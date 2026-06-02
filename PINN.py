import torch
import torch.nn as nn
import numpy as np
import args_parse
from pathlib import Path

args = args_parse.args_parser()

# Spectral transformer block for T_OTCM and T_TTCM
class SpectralEncoderTransformer(nn.Module):
    def __init__(self, n_wavelengths=args.n_wavelengths, d_spec=args.SET_d_spec, n_heads=args.SET_n_head, n_layers=args.SET_n_layers):
        super().__init__()
        self.d_spec = d_spec

        # Every wavelength → vector of d_spec
        self.input_proj = nn.Linear(1, d_spec)

        # Positional embedding (47 tokens lang, d_spec breed)
        self.pos_emb = nn.Parameter(torch.randn(1, n_wavelengths, d_spec) * 0.02)

        # Mini transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_spec,
            nhead=n_heads,
            dim_feedforward=d_spec * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Pooling to one vector per spectrum
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, 47)  (één spectrum)
        B = x.shape[0]

        # Make tokens: (B, 47, 1) → (B, 47, d_spec)
        x = x.unsqueeze(-1)
        x = self.input_proj(x)

        # Give positional embeddings
        x = x + self.pos_emb  # (B, 47, d_spec)

        # Through encoder
        x_enc = self.transformer(x)  # (B, 47, d_spec)

        # Pooling: average over tokens (wavelengths)
        x_pooled = x_enc.mean(dim=1)  # (B, d_spec)

        return x_pooled


# PINNs --> PMLP-OTCM, T_OTCM, PMLP-PMLP_TTCM, T_TTCM

# PMLP-OTCM
class PMLP_OTCM(nn.Module):
    def __init__(
        self,
        input_size=(args.n_wavelengths * args.n_angles),
        hidden_size=args.hidden_layer_size,
        num_hidden_layers=args.n_hidden_layer,
        activation=args.activation_function,
        use_dropout=args.dropout,
        dropout_prob=args.dropout_size
    ):
        super(PMLP_OTCM, self).__init__()


        # Define min and max values for each target variable
        self.output_min = torch.tensor([args.BvF_min, args.S_min, args.Rv_min, (1 - args.BvF_max + args.MvF_max), args.MvF_min, args.a2_min, args.b2_min, args.e_min, args.e_min, args.e_min, args.e_min])  # Example min values
        self.output_max = torch.tensor([args.BvF_max, args.S_max, args.Rv_max, (1 - args.BvF_min + args.MvF_min), args.MvF_max, args.a2_max, args.b2_max, args.e_max, args.e_max, args.e_max, args.e_max])  # Example max values
        self.delta_output = self.output_max - self.output_min
        # Absorption coefficient matrix
        ROOT = Path(__file__).resolve().parent

        self.data_absorption = np.load(
            ROOT /
            "Absorptioncoefficients" /
            "absorption_coefficient_matrix.npy"
        )

        # Select activation function
        if activation.lower() == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Create branches
        self.branches = nn.ModuleList()
        for _ in range(11):
            layers = []
            # Input layer
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(self.activation_fn)
            if use_dropout:
                layers.append(nn.Dropout(dropout_prob))

            # Hidden layers
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(self.activation_fn)
                if use_dropout:
                    layers.append(nn.Dropout(dropout_prob))

            # Output layer
            layers.append(nn.Linear(hidden_size, 1))
            layers.append(nn.Sigmoid())  # Keep output in [0,1] for scaling

            branch = nn.Sequential(*layers)
            self.branches.append(branch)

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        outputs = torch.stack(outputs, dim=0).squeeze()
        output_min = self.output_min.unsqueeze(1)
        output_max = self.output_max.unsqueeze(1)
        scaled_outputs = output_min + outputs * (output_max - output_min)
        # print(scaled_outputs.shape)
        return scaled_outputs

    def loss_dt(self, x, outputs, lamda1=args.w1, lamda2=args.w2, lamda3=args.w3, lamda4=args.w4):

        # Define parameters from outputs
        BvF = outputs[0]
        Hb_sat = outputs[1]
        Rv = outputs[2]
        SvF = outputs[3]
        MvF = outputs[4]
        a2 = outputs[5]
        b2 = outputs[6]
        e1 = outputs[7]
        e2 = outputs[8]
        e3 = outputs[9]
        e4 = outputs[10]

        # Define constant values for wavelength range and absorption coefficients
        wavelengths = torch.linspace(args.min_wavelength, args.max_wavelength, args.n_wavelengths)

        # Define absorption coefficients for different wavelengths
        ua_skin_baseline = self.data_absorption[0]
        ua_melanin = self.data_absorption[5]
        ua_oxyhaem = self.data_absorption[1]
        ua_deoxyhaem = self.data_absorption[2]
        refractive_index_skin = self.data_absorption[6]
        refractive_index_air = self.data_absorption[7]

        # Transform to Tensors
        ua_skin_baseline = torch.tensor(ua_skin_baseline)
        ua_melanin = torch.tensor(ua_melanin)
        ua_oxyhaem = torch.tensor(ua_oxyhaem)
        ua_deoxyhaem = torch.tensor(ua_deoxyhaem)
        refractive_index_skin = torch.tensor(refractive_index_skin)
        refractive_index_air = torch.tensor(refractive_index_air)

        # Calculating individual contributions of compartments
        # Calculating individual absorption
        ua_skin_baseline_absorption = torch.outer(SvF, ua_skin_baseline)
        # Melanin
        ua_melanin_absorption = torch.outer(MvF, ua_melanin)
        # Oxygenated blood
        ua_oxy_absorption = torch.outer(Hb_sat, ua_oxyhaem)
        # Deoxygenated blood
        ua_deoxy_absorption = torch.outer((1 - Hb_sat), ua_deoxyhaem)
        # Absorption coefficient whole blood
        ua_bl = (ua_oxy_absorption + ua_deoxy_absorption)

        # Calculating to correct for difference in radius of vessel
        # Calculating the transmission factor
        transmissionfactor = 2 * ua_bl * Rv.view(-1, 1)
        # Calculating the numerator
        C_diff_numerator = 1 - torch.exp(-transmissionfactor)
        # Calculating the dominator
        C_diff_dominator = 2 * Rv
        # Calculating the Diffusion factor
        C_diff = C_diff_numerator / C_diff_dominator.view(-1, 1)

        # Calculating the absorption coefficient of whole blood
        ua_bl_absorption = BvF.view(-1, 1) * args.c_hb * C_diff

        # Absorption total
        ua = torch.abs(ua_bl_absorption + ua_skin_baseline_absorption + ua_melanin_absorption)

        # Scattering
        us = a2[:, np.newaxis] * (wavelengths / 500) ** (-b2[:, np.newaxis])

        # Calculating the albedo (which is the proportionality of absorption coefficient and scattering coefficient
        a = us / (us + ua)

        # Calculating k ( Total internal reflection part)
        n_rel = refractive_index_skin / refractive_index_air
        rd = (-1.44 * n_rel ** -2) + (0.71 * n_rel ** -1) + (0.668 + 0.0636 * n_rel)
        k = (1 + rd) / (1 - rd)

        # Calculating Rdif
        dominator = (1 + 2 * k * (1 - a) + (1 + 2 * k / 3) * torch.sqrt(3 * (1 - a)))
        Rdif = a / dominator

        # Surface reflectance
        # Fresnel
        fr = ((refractive_index_skin - refractive_index_air) / (refractive_index_skin + refractive_index_air)) ** 2
        fr_normalized = fr/fr[0]

        # Calculate each reflectance
        Rsurf1 = torch.outer(e1, fr_normalized)
        Rsurf2 = torch.outer(e2, fr_normalized)
        Rsurf3 = torch.outer(e3, fr_normalized)
        Rsurf4 = torch.outer(e4, fr_normalized)

        # Data
        Rdata1 = x[:, :args.n_wavelengths]
        Rdata2 = x[:, args.n_wavelengths:(2*args.n_wavelengths)]
        Rdata3 = x[:, (2*args.n_wavelengths):(3*args.n_wavelengths)]
        Rdata4 = x[:, (3*args.n_wavelengths):(4*args.n_wavelengths)]

        # Predicted data
        Rpred1 = Rdif + Rsurf1
        Rpred2 = Rdif + Rsurf2
        Rpred3 = Rdif + Rsurf3
        Rpred4 = Rdif + Rsurf4

        # Loss calculations
        # Data loss
        # Angle one
        Loss1_lamda1 = ((Rdata1[:, :16] - (Rdif[:, :16] + Rsurf1[:, :16])) ** 2) * lamda1
        Loss1_lamda2 = ((Rdata1[:, 16:25] - (Rdif[:, 16:25] + Rsurf1[:, 16:25])) ** 2) * lamda2
        Loss1_lamda3 = ((Rdata1[:, 25:47] - (Rdif[:, 25:47] + Rsurf1[:, 25:47])) ** 2) * lamda3
        total_loss_angle1 = (Loss1_lamda1.sum() + Loss1_lamda2.sum() + Loss1_lamda3.sum())
        # Angle two
        Loss2_lamda1 = ((Rdata2[:, :16] - (Rdif[:, :16] + Rsurf2[:, :16])) ** 2) * lamda1
        Loss2_lamda2 = ((Rdata2[:, 16:25] - (Rdif[:, 16:25] + Rsurf2[:, 16:25])) ** 2) * lamda2
        Loss2_lamda3 = ((Rdata2[:, 25:47] - (Rdif[:, 25:47] + Rsurf2[:, 25:47])) ** 2) * lamda3
        total_loss_angle2 = (Loss2_lamda1.sum() + Loss2_lamda2.sum() + Loss2_lamda3.sum())
        # Angle three
        Loss3_lamda1 = ((Rdata3[:, :16] - (Rdif[:, :16] + Rsurf3[:, :16])) ** 2) * lamda1
        Loss3_lamda2 = ((Rdata3[:, 16:25] - (Rdif[:, 16:25] + Rsurf3[:, 16:25])) ** 2) * lamda2
        Loss3_lamda3 = ((Rdata3[:, 25:47] - (Rdif[:, 25:47] + Rsurf3[:, 25:47])) ** 2) * lamda3
        total_loss_angle3 = (Loss3_lamda1.sum() + Loss3_lamda2.sum() + Loss3_lamda3.sum())
        # Angle four
        Loss4_lamda1 = ((Rdata4[:, :16] - (Rdif[:, :16] + Rsurf4[:, :16])) ** 2) * lamda1
        Loss4_lamda2 = ((Rdata4[:, 16:25] - (Rdif[:, 16:25] + Rsurf4[:, 16:25])) ** 2) * lamda2
        Loss4_lamda3 = ((Rdata4[:, 25:47] - (Rdif[:, 25:47] + Rsurf4[:, 25:47])) ** 2) * lamda3
        total_loss_angle4 = (Loss4_lamda1.sum() + Loss4_lamda2.sum() + Loss4_lamda3.sum())

        # Physics loss
        Loss_fractions = ((1 - (BvF + SvF + MvF)) ** 2) * lamda4

        # Total loss
        loss = total_loss_angle1 + total_loss_angle2 + total_loss_angle3 + total_loss_angle4 + Loss_fractions.sum()

        return loss, Rdata1[0, :], Rdata2[0, :], Rdata3[0, :], Rdata4[0, :], Rpred1[0, :], Rpred2[0, :], Rpred3[0, :], Rpred4[0, :]

# T_OTCM
class T_OTCM(nn.Module):
    def __init__(self, n_angles=args.n_angles, n_wavelengths=args.n_wavelengths, d_spec_set=args.SET_d_spec, n_heads_set = args.SET_n_head, n_layers_set = args.SET_n_layers, d_model=args.AET_d_model, n_heads=args.AET_n_head, n_layers=args.AET_n_layers, hidden_mlp=args.AET_hl_MLP, out_params=11):
        super().__init__()

        # Define min and max values for each target variable
        self.output_min = torch.tensor([args.BvF_min, args.S_min, args.Rv_min, (1 - args.BvF_max + args.MvF_max), args.MvF_min, args.a2_min, args.b2_min, args.e_min, args.e_min, args.e_min, args.e_min])  # Example min values
        self.output_max = torch.tensor([args.BvF_max, args.S_max, args.Rv_max, (1 - args.BvF_min + args.MvF_min), args.MvF_max, args.a2_max, args.b2_max, args.e_max, args.e_max, args.e_max, args.e_max])  # Example max values
        # Absorption coefficient matrix
        ROOT = Path(__file__).resolve().parent

        self.data_absorption = np.load(
            ROOT /
            "Absorptioncoefficients" /
            "absorption_coefficient_matrix.npy"
        )

        # Use mini-transformer
        self.spectral_encoder = SpectralEncoderTransformer(n_wavelengths=n_wavelengths, d_spec=d_spec_set, n_heads = n_heads_set, n_layers=n_layers_set)

        self.angle_emb = nn.Parameter(torch.randn(n_angles, d_model) * 0.02)
        self.proj = nn.Linear(d_spec_set, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=d_model*4,
                                                   activation='gelu',
                                                   batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_mlp),
            nn.GELU(),
            nn.Linear(hidden_mlp, out_params)
        )

    def forward(self, x):
        # x: (B, 4, 47)
        spectra_enc = []
        for i in range(x.shape[1]):  # 4 angles
            spec_i = x[:, i, :]  # (B, 47)
            enc_i = self.spectral_encoder(spec_i)  # (B, d_spec)
            spectra_enc.append(enc_i)
        spectra_enc = torch.stack(spectra_enc, dim=1)  # (B, 4, d_spec)

        # Project to d_model + add angular embeddings
        x_proj = self.proj(spectra_enc) + self.angle_emb.unsqueeze(0)  # (B, 4, d_model)

        # Transformer of angles
        x_enc = self.transformer(x_proj)  # (B, 4, d_model)

        # Pool over angles
        x_pooled = x_enc.mean(dim=1)  # (B, d_model)

        # MLP to 11 parameters
        outputs = self.mlp(x_pooled)

        # Constrain to [0,1]
        outputs = torch.sigmoid(outputs)  # shape (B, 11)

        # Scale to desired range
        scaled_outputs = self.output_min + outputs * (self.output_max - self.output_min)

        return scaled_outputs

    def loss_dt(self, x, outputs):

        # Define parameters from outputs
        BvF = outputs[:, 0]
        Hb_sat = outputs[:, 1]
        Rv = outputs[:, 2]
        SvF = outputs[:, 3]
        MvF = outputs[:, 4]
        a2 = outputs[:, 5]
        b2 = outputs[:, 6]
        e1 = outputs[:, 7]
        e2 = outputs[:, 8]
        e3 = outputs[:, 9]
        e4 = outputs[:, 10]

        # Define constant values for wavelength range and absorption coefficients
        wavelengths = torch.linspace(args.min_wavelength, args.max_wavelength, args.n_wavelengths)

        # Define absorption coefficients for different wavelengths
        ua_skin_baseline = self.data_absorption[0]
        ua_melanin = self.data_absorption[5]
        ua_oxyhaem = self.data_absorption[1]
        ua_deoxyhaem = self.data_absorption[2]
        refractive_index_skin = self.data_absorption[6]
        refractive_index_air = self.data_absorption[7]

        # Transform to Tensors
        ua_skin_baseline = torch.tensor(ua_skin_baseline)
        ua_melanin = torch.tensor(ua_melanin)
        ua_oxyhaem = torch.tensor(ua_oxyhaem)
        ua_deoxyhaem = torch.tensor(ua_deoxyhaem)
        refractive_index_skin = torch.tensor(refractive_index_skin)
        refractive_index_air = torch.tensor(refractive_index_air)

        # Calculating individual contributions of compartments
        # Calculating individual absorption
        ua_skin_baseline_absorption = torch.outer(SvF, ua_skin_baseline)
        # Melanin
        ua_melanin_absorption = torch.outer(MvF, ua_melanin)
        # Oxygenated blood
        ua_oxy_absorption = torch.outer(Hb_sat, ua_oxyhaem)
        # Deoxygenated blood
        ua_deoxy_absorption = torch.outer((1 - Hb_sat), ua_deoxyhaem)
        # Absorption coefficient whole blood
        ua_bl = (ua_oxy_absorption + ua_deoxy_absorption)

        # Calculating to correct for difference in radius of vessel
        # Calculating the transmission factor
        transmissionfactor = 2 * ua_bl * Rv.view(-1, 1)
        # Calculating the numerator
        C_diff_numerator = 1 - torch.exp(-transmissionfactor)
        # Calculating the dominator
        C_diff_dominator = 2 * Rv
        # Calculating the Diffusion factor
        C_diff = C_diff_numerator / C_diff_dominator.view(-1, 1)

        # Calculating the absorption coefficient of whole blood
        ua_bl_absorption = BvF.view(-1, 1) * args.c_hb * C_diff

        # Absorption total
        ua = torch.abs(ua_bl_absorption + ua_skin_baseline_absorption + ua_melanin_absorption)

        # Scattering
        us = a2[:, np.newaxis] * (wavelengths / 500) ** (-b2[:, np.newaxis])

        # Calculating the albedo (which is the proportionality of absorption coefficient and scattering coefficient
        a = us / (us + ua)

        # Calculating k ( Total internal reflection part)
        n_rel = refractive_index_skin / refractive_index_air
        rd = (-1.44 * n_rel ** -2) + (0.71 * n_rel ** -1) + (0.668 + 0.0636 * n_rel)
        k = (1 + rd) / (1 - rd)

        # Calculating Rdif
        dominator = (1 + 2 * k * (1 - a) + (1 + 2 * k / 3) * torch.sqrt(3 * (1 - a)))
        Rdif = a / dominator

        # Surface reflectance
        # Fresnel
        fr = ((refractive_index_skin - refractive_index_air) / (refractive_index_skin + refractive_index_air)) ** 2
        fr_normalized = fr/fr[0]

        # Calculate each reflectance
        Rsurf1 = torch.outer(e1, fr_normalized)
        Rsurf2 = torch.outer(e2, fr_normalized)
        Rsurf3 = torch.outer(e3, fr_normalized)
        Rsurf4 = torch.outer(e4, fr_normalized)

        # Data
        Rdata1 = x[:, 0, :]
        Rdata2 = x[:, 1, :]
        Rdata3 = x[:, 2, :]
        Rdata4 = x[:, 3, :]

        # Predicted data
        Rpred1 = Rdif + Rsurf1
        Rpred2 = Rdif + Rsurf2
        Rpred3 = Rdif + Rsurf3
        Rpred4 = Rdif + Rsurf4

        # Loss calculations
        # Data loss
        Loss_angle1 = (Rdata1-Rpred1)**2
        Loss_angle2 = (Rdata2-Rpred2)**2
        Loss_angle3 = (Rdata3-Rpred3)**2
        Loss_angle4 = (Rdata4-Rpred4)**2
        total_loss_data = (Loss_angle1.sum() + Loss_angle2.sum() + Loss_angle3.sum() + Loss_angle4.sum())

        # Physics loss
        Loss_fractions = ((1 - (BvF + SvF + MvF)) ** 2)
        # Total loss
        loss = total_loss_data + Loss_fractions.sum()

        return loss, Rdata1[0, :], Rdata2[0, :], Rdata3[0, :], Rdata4[0, :], Rpred1[0, :], Rpred2[0, :], Rpred3[0, :], Rpred4[0, :]

# PMLP-TTCM
class PMLP_TTCM(nn.Module):
    def __init__(
        self,
        input_size=(args.n_wavelengths * args.n_angles),
        hidden_size=args.hidden_layer_size,
        num_hidden_layers=args.n_hidden_layer,
        activation=args.activation_function,
        use_dropout=args.dropout,
        dropout_prob=args.dropout_size
    ):
        super(PMLP_TTCM, self).__init__()

        # Define min and max values for each target variable
        self.output_min = torch.tensor([args.MvF_min, (1-args.MvF_max), args.d1, args.a1_min, args.b1_min, args.BvF_min, args.S_min, args.Rv_min, (1-args.BvF_max), args.a2_min, args.b2_min, args.e_min, args.e_min, args.e_min, args.e_min])
        self.output_max = torch.tensor([args.MvF_max, (1-args.MvF_min), args.d1, args.a1_max, args.b1_max, args.BvF_max, args.S_max, args.Rv_max, (1-args.BvF_min), args.a2_max, args.b2_max, args.e_max, args.e_max, args.e_max, args.e_max])
        self.delta_output = self.output_max - self.output_min
        # Absorption coefficient matrix
        ROOT = Path(__file__).resolve().parent

        self.data_absorption = np.load(
            ROOT /
            "Absorptioncoefficients" /
            "absorption_coefficient_matrix.npy"
        )

        # Select activation function
        if activation.lower() == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Create branches
        self.branches = nn.ModuleList()
        for _ in range(15):
            layers = []
            # Input layer
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(self.activation_fn)
            if use_dropout:
                layers.append(nn.Dropout(dropout_prob))

            # Hidden layers
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(self.activation_fn)
                if use_dropout:
                    layers.append(nn.Dropout(dropout_prob))

            # Output layer
            layers.append(nn.Linear(hidden_size, 1))
            layers.append(nn.Sigmoid())  # Keep output in [0,1] for scaling

            branch = nn.Sequential(*layers)
            self.branches.append(branch)

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        outputs = torch.stack(outputs, dim=0).squeeze()
        output_min = self.output_min.unsqueeze(1)
        output_max = self.output_max.unsqueeze(1)
        scaled_outputs = output_min + outputs * (output_max - output_min)
        return scaled_outputs

    def loss_dt(self, x, outputs, lamda1=args.w1, lamda2=args.w2, lamda3=args.w3, lamda4=args.w4):
        # Define parameters from outputs
        MvF = outputs[0]
        SvF1 = outputs[1]
        d1 = outputs[2]
        a1 = outputs[3]
        b1 = outputs[4]
        BvF = outputs[5]
        Hb_sat = outputs[6]
        Rv = outputs[7]
        SvF2 = outputs[8]
        a2 = outputs[9]
        b2 = outputs[10]
        e1 = outputs[11]
        e2 = outputs[12]
        e3 = outputs[13]
        e4 = outputs[14]

        # Define constant values for wavelength range and absorption coefficients
        wavelengths = torch.linspace(args.min_wavelength, args.max_wavelength, args.n_wavelengths)
        # Define absorption coeficients for different wavelengths
        ua_skin_baseline = self.data_absorption[0]
        ua_melanin = self.data_absorption[5]
        ua_oxyhaem = self.data_absorption[1]
        ua_deoxyhaem = self.data_absorption[2]
        refractive_index_skin = self.data_absorption[6]
        refractive_index_air = self.data_absorption[7]
        # Transform to Tensors
        ua_skin_baseline = torch.tensor(ua_skin_baseline)
        ua_melanin = torch.tensor(ua_melanin)
        ua_oxyhaem = torch.tensor(ua_oxyhaem)
        ua_deoxyhaem = torch.tensor(ua_deoxyhaem)
        refractive_index_skin = torch.tensor(refractive_index_skin)

        # Epidermis
        # Absorption skin baseline 1
        ua_skin_baseline_absorption1 = torch.outer(SvF1, ua_skin_baseline)
        # Absorption melanin
        ua_melanin_absorption = torch.outer(MvF, ua_melanin)
        # Total absorption epidermis
        ua1 = torch.abs(ua_skin_baseline_absorption1 + ua_melanin_absorption)
        # Scatterings-coefficient Epidermis
        us1 = a1[:, np.newaxis] * (wavelengths / 500) ** (-b1[:, np.newaxis])
        # transport coefficient epidermal layer
        ueff1 = ua1 + us1
        # Diffusion coefficient D epidermal layer
        D1 = 1 / (3 * ueff1)
        D1_squared = D1 ** 2

        # Optical penetration depth epidermal layer
        delta1 = torch.sqrt(1 / (3 * ueff1 * ua1))
        delta1_squared = delta1 ** 2
        delta1_cube_squared = delta1 ** 3

        # Dermis
        # Absorption skin baseline 2
        ua_skin_baseline_absorption2 = torch.outer(SvF2, ua_skin_baseline)
        # Oxygenated blood
        ua_oxy_absorption = torch.outer(Hb_sat, ua_oxyhaem)
        # Deoxygenated blood
        ua_deoxy_absorption = torch.outer((1 - Hb_sat), ua_deoxyhaem)
        # Absorption coefficient whole blood
        ua_bl = (ua_oxy_absorption + ua_deoxy_absorption)

        # Calculating to correct for difference in radius of vessel
        # Calculating the transmission factor
        transmissionfactor = 2 * ua_bl * Rv.view(-1, 1)
        # Calculating the numerator
        C_diff_numerator = 1 - torch.exp(-transmissionfactor)
        # Calculating the dominator
        C_diff_dominator = 2 * Rv
        # Calculating the Diffusion factor
        C_diff = C_diff_numerator / C_diff_dominator.view(-1, 1)
        # Calculating the absorption coefficient of whole blood
        ua_bl_absorption = BvF.view(-1, 1) * args.c_hb * C_diff
        # Calculating the total absorption dermis
        ua2 = (ua_skin_baseline_absorption2 + ua_bl_absorption)

        # Scatterings-coefficient Dermis
        us2 = a2[:, np.newaxis] * (wavelengths / 500) ** (-b2[:, np.newaxis])
        # transport coefficient epidermal layer
        ueff2 = ua2 + us2
        # Diffusion coefficient D epidermal layer
        D2 = 1 / (3 * ueff2)
        # Optical penetration depth epidermal layer
        delta2 = torch.sqrt(1 / (3 * ueff2 * ua2))
        delta2_squared = delta2 ** 2

        # Total internal coefficient A
        r_eff = args.r_eff
        # Calculate k
        A = (1 - r_eff) / (2 * (1 + r_eff))

        # Calculating the reflection coefficient
        y_A = delta1 * us1 * A
        # From spott et al, Svaasand et al
        y_B = (1 + (delta2 / (3 * D2))) * torch.abs((((delta1_squared * delta2) / 3) - (delta1_squared * D2)) * torch.cosh(d1[:, np.newaxis] / delta1) + (((delta1_cube_squared * D2) / (3 * D1)) - (delta1 * delta2 * D1)) * torch.sinh(d1[:, np.newaxis] / delta1))
        y_C = torch.abs(delta2_squared * D1 * (us2 / us1) * ((delta1_squared / (9 * D1_squared)) - 1) + (delta1_squared * (D2 - (delta2_squared / (9 * D2)))) * torch.exp(-d1[:, np.newaxis] / (3 * D1)))
        y_D = ((delta1_squared / (9 * D1_squared)) + 1) * ((delta2 / (3 * D2) + 1)) * torch.abs((D1 * delta1 * (D2 + (delta2 * A)) * torch.cosh(d1[:, np.newaxis] / delta1)) + (((D1_squared * delta2) + (D2 * delta1_squared * A)) * torch.sinh(d1[:, np.newaxis] / delta1)))
        Rdif = y_A * ((y_B + y_C) / y_D)

        # Surface reflectance
        # Fresnel
        fr = ((refractive_index_skin - refractive_index_air) / (refractive_index_skin + refractive_index_air)) ** 2
        fr_normalized = fr / fr[0]

        # Calculate each reflectance
        Rsurf1 = torch.outer(e1, fr_normalized)
        Rsurf2 = torch.outer(e2, fr_normalized)
        Rsurf3 = torch.outer(e3, fr_normalized)
        Rsurf4 = torch.outer(e4, fr_normalized)

        # Data
        Rdata1 = x[:, :args.n_wavelengths]
        Rdata2 = x[:, args.n_wavelengths:(2*args.n_wavelengths)]
        Rdata3 = x[:, (2*args.n_wavelengths):(3*args.n_wavelengths)]
        Rdata4 = x[:, (3*args.n_wavelengths):(4*args.n_wavelengths)]

        # Predicted data
        Rpred1 = Rdif + Rsurf1
        Rpred2 = Rdif + Rsurf2
        Rpred3 = Rdif + Rsurf3
        Rpred4 = Rdif + Rsurf4

        # Loss calculations
        # Data loss
        # Angle one
        Loss1_lamda1 = ((Rdata1[:, :16] - (Rdif[:, :16] + Rsurf1[:, :16])) ** 2) * lamda1
        Loss1_lamda2 = ((Rdata1[:, 16:25] - (Rdif[:, 16:25] + Rsurf1[:, 16:25])) ** 2) * lamda2
        Loss1_lamda3 = ((Rdata1[:, 25:47] - (Rdif[:, 25:47] + Rsurf1[:, 25:47])) ** 2) * lamda3
        total_loss_angle1 = (Loss1_lamda1.sum() + Loss1_lamda2.sum() + Loss1_lamda3.sum())
        # Angle two
        Loss2_lamda1 = ((Rdata2[:, :16] - (Rdif[:, :16] + Rsurf2[:, :16])) ** 2) * lamda1
        Loss2_lamda2 = ((Rdata2[:, 16:25] - (Rdif[:, 16:25] + Rsurf2[:, 16:25])) ** 2) * lamda2
        Loss2_lamda3 = ((Rdata2[:, 25:47] - (Rdif[:, 25:47] + Rsurf2[:, 25:47])) ** 2) * lamda3
        total_loss_angle2 = (Loss2_lamda1.sum() + Loss2_lamda2.sum() + Loss2_lamda3.sum())
        # Angle three
        Loss3_lamda1 = ((Rdata3[:, :16] - (Rdif[:, :16] + Rsurf3[:, :16])) ** 2) * lamda1
        Loss3_lamda2 = ((Rdata3[:, 16:25] - (Rdif[:, 16:25] + Rsurf3[:, 16:25])) ** 2) * lamda2
        Loss3_lamda3 = ((Rdata3[:, 25:47] - (Rdif[:, 25:47] + Rsurf3[:, 25:47])) ** 2) * lamda3
        total_loss_angle3 = (Loss3_lamda1.sum() + Loss3_lamda2.sum() + Loss3_lamda3.sum())
        # Angle four
        Loss4_lamda1 = ((Rdata4[:, :16] - (Rdif[:, :16] + Rsurf4[:, :16])) ** 2) * lamda1
        Loss4_lamda2 = ((Rdata4[:, 16:25] - (Rdif[:, 16:25] + Rsurf4[:, 16:25])) ** 2) * lamda2
        Loss4_lamda3 = ((Rdata4[:, 25:47] - (Rdif[:, 25:47] + Rsurf4[:, 25:47])) ** 2) * lamda3
        total_loss_angle4 = (Loss4_lamda1.sum() + Loss4_lamda2.sum() + Loss4_lamda3.sum())

        # Physics loss
        # TvF of the first layer
        Loss_fractions1 = ((1 - (SvF1 + MvF)) ** 2) * lamda4
        # TvF of the second layer
        Loss_fractions2 = ((1 - (SvF2 + BvF)) ** 2) * lamda4

        # Total loss
        loss = total_loss_angle1 + total_loss_angle2 + total_loss_angle3 + total_loss_angle4 + Loss_fractions1.sum() + Loss_fractions2.sum()

        return loss, Rdata1[0, :], Rdata2[0, :], Rdata3[0, :], Rdata4[0, :], Rpred1[0, :], Rpred2[0, :], Rpred3[0, :], Rpred4[0, :]

# T_TTCM
class T_TTCM(nn.Module):
    def __init__(self, n_angles=args.n_angles, n_wavelengths=args.n_wavelengths, d_spec_set=args.SET_d_spec, n_heads_set = args.SET_n_head, n_layers_set = args.SET_n_layers, d_model=args.AET_d_model, n_heads=args.AET_n_head, n_layers=args.AET_n_layers, hidden_mlp=args.AET_hl_MLP, out_params=15):
        super().__init__()

        # Define min and max values for each target variable
        self.output_min = torch.tensor([args.MvF_min, (1-args.MvF_max), args.d1, args.a1_min, args.b1_min, args.BvF_min, args.S_min, args.Rv_min, (1-args.BvF_max), args.a2_min, args.b2_min, args.e_min, args.e_min, args.e_min, args.e_min])
        self.output_max = torch.tensor([args.MvF_max, (1-args.MvF_min), args.d1, args.a1_max, args.b1_max, args.BvF_max, args.S_max, args.Rv_max, (1-args.BvF_min), args.a2_max, args.b2_max, args.e_max, args.e_max, args.e_max, args.e_max])
        self.delta_output = self.output_max - self.output_min
        # Absorption coefficient matrix
        ROOT = Path(__file__).resolve().parent

        self.data_absorption = np.load(
            ROOT /
            "Absorptioncoefficients" /
            "absorption_coefficient_matrix.npy"
        )

        # Use mini-transformer
        self.spectral_encoder = SpectralEncoderTransformer(n_wavelengths=n_wavelengths, d_spec=d_spec_set, n_heads = n_heads_set, n_layers=n_layers_set)

        self.angle_emb = nn.Parameter(torch.randn(n_angles, d_model) * 0.02)
        self.proj = nn.Linear(d_spec_set, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=d_model*4,
                                                   activation='gelu',
                                                   batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_mlp),
            nn.GELU(),
            nn.Linear(hidden_mlp, out_params)
        )

    def forward(self, x):
        # x: (B, 4, 47)
        spectra_enc = []
        for i in range(x.shape[1]):  # over 4 angles
            spec_i = x[:, i, :]  # (B, 47)
            enc_i = self.spectral_encoder(spec_i)  # (B, d_spec)
            spectra_enc.append(enc_i)
        spectra_enc = torch.stack(spectra_enc, dim=1)  # (B, 4, d_spec)

        # Project to d_model + add angular embeddings
        x_proj = self.proj(spectra_enc) + self.angle_emb.unsqueeze(0)  # (B, 4, d_model)

        # Transformer over angles
        x_enc = self.transformer(x_proj)  # (B, 4, d_model)

        # Pool over angles
        x_pooled = x_enc.mean(dim=1)  # (B, d_model)

        # MLP to 11 parameters
        outputs = self.mlp(x_pooled)

        # Constrain to [0,1]
        outputs = torch.sigmoid(outputs)  # shape (B, 11)

        # Scale to desired range
        scaled_outputs = self.output_min + outputs * (self.output_max - self.output_min)

        return scaled_outputs

    def loss_dt(self, x, outputs):
        # Define parameters from outputs
        MvF = outputs[:, 0]
        SvF1 = outputs[:, 1]
        d1 = outputs[:, 2]
        a1 = outputs[:, 3]
        b1 = outputs[:, 4]
        BvF = outputs[:, 5]
        Hb_sat = outputs[:, 6]
        Rv = outputs[:, 7]
        SvF2 = outputs[:, 8]
        a2 = outputs[:, 9]
        b2 = outputs[:, 10]
        e1 = outputs[:, 11]
        e2 = outputs[:, 12]
        e3 = outputs[:, 13]
        e4 = outputs[:, 14]

        # Define constant values for wavelength range and absorption coefficients
        wavelengths = torch.linspace(args.min_wavelength, args.max_wavelength, args.n_wavelengths)
        # Define absorption coefficients for different wavelengths
        ua_skin_baseline = self.data_absorption[0]
        ua_melanin = self.data_absorption[5]
        ua_oxyhaem = self.data_absorption[1]
        ua_deoxyhaem = self.data_absorption[2]
        refractive_index_skin = self.data_absorption[6]
        refractive_index_air = self.data_absorption[7]
        # Transform to Tensors
        ua_skin_baseline = torch.tensor(ua_skin_baseline)
        ua_melanin = torch.tensor(ua_melanin)
        ua_oxyhaem = torch.tensor(ua_oxyhaem)
        ua_deoxyhaem = torch.tensor(ua_deoxyhaem)
        refractive_index_skin = torch.tensor(refractive_index_skin)

        # Epidermis
        # Absorption skin baseline 1
        ua_skin_baseline_absorption1 = torch.outer(SvF1, ua_skin_baseline)
        # Absorption melanin
        ua_melanin_absorption = torch.outer(MvF, ua_melanin)
        # Total absorption epidermis
        ua1 = torch.abs(ua_skin_baseline_absorption1 + ua_melanin_absorption)
        # Scatterings-coefficient Epidermis
        us1 = a1[:, np.newaxis] * (wavelengths / 500) ** (-b1[:, np.newaxis])
        # transport coefficient epidermal layer
        ueff1 = ua1 + us1
        # Diffusion coefficient D epidermal layer
        D1 = 1 / (3 * ueff1)
        D1_squared = D1 ** 2

        # Optical penetration depth epidermal layer
        delta1 = torch.sqrt(1 / (3 * ueff1 * ua1))
        delta1_squared = delta1 ** 2
        delta1_cube_squared = delta1 ** 3

        # Dermis
        # Absorption skin baseline 2
        ua_skin_baseline_absorption2 = torch.outer(SvF2, ua_skin_baseline)
        # Oxygenated blood
        ua_oxy_absorption = torch.outer(Hb_sat, ua_oxyhaem)
        # Deoxygenated blood
        ua_deoxy_absorption = torch.outer((1 - Hb_sat), ua_deoxyhaem)
        # Absorption coefficient whole blood
        ua_bl = (ua_oxy_absorption + ua_deoxy_absorption)

        # Calculating to correct for difference in radius of vessel
        # Calculating the transmission factor
        transmissionfactor = 2 * ua_bl * Rv.view(-1, 1)
        # Calculating the numerator
        C_diff_numerator = 1 - torch.exp(-transmissionfactor)
        # Calculating the dominator
        C_diff_dominator = 2 * Rv
        # Calculating the Diffusion factor
        C_diff = C_diff_numerator / C_diff_dominator.view(-1, 1)
        # Calulcating the absorption coefficient of whole blood
        ua_bl_absorption = BvF.view(-1, 1) * args.c_hb * C_diff
        # Calculating the total absorption dermis
        ua2 = (ua_skin_baseline_absorption2 + ua_bl_absorption)

        # Scatterings-coefficient Dermis
        us2 = a2[:, np.newaxis] * (wavelengths / 500) ** (-b2[:, np.newaxis])
        # transport coefficient epidermal layer
        ueff2 = ua2 + us2
        # Diffusion coefficient D epidermal layer
        D2 = 1 / (3 * ueff2)
        # Optical penetration depth epidermal layer
        delta2 = torch.sqrt(1 / (3 * ueff2 * ua2))
        delta2_squared = delta2 ** 2

        # Total internal coefficient A
        r_eff = args.r_eff
        # Calculate k
        A = (1 - r_eff) / (2 * (1 + r_eff))

        # Calculating the reflection coefficient
        y_A = delta1 * us1 * A
        # From spott et al, Svaasand et al
        y_B = (1 + (delta2 / (3 * D2))) * torch.abs((((delta1_squared * delta2) / 3) - (delta1_squared * D2)) * torch.cosh(d1[:, np.newaxis] / delta1) + (((delta1_cube_squared * D2) / (3 * D1)) - (delta1 * delta2 * D1)) * torch.sinh(d1[:, np.newaxis] / delta1))
        y_C = torch.abs(delta2_squared * D1 * (us2 / us1) * ((delta1_squared / (9 * D1_squared)) - 1) + (delta1_squared * (D2 - (delta2_squared / (9 * D2)))) * torch.exp(-d1[:, np.newaxis] / (3 * D1)))
        y_D = ((delta1_squared / (9 * D1_squared)) + 1) * ((delta2 / (3 * D2) + 1)) * torch.abs((D1 * delta1 * (D2 + (delta2 * A)) * torch.cosh(d1[:, np.newaxis] / delta1)) + (((D1_squared * delta2) + (D2 * delta1_squared * A)) * torch.sinh(d1[:, np.newaxis] / delta1)))
        Rdif = y_A * ((y_B + y_C) / y_D)

        # Surface reflectance
        # Fresnel
        fr = ((refractive_index_skin - refractive_index_air) / (refractive_index_skin + refractive_index_air)) ** 2
        fr_normalized = fr / fr[0]

        # Calculate each reflectance
        Rsurf1 = torch.outer(e1, fr_normalized)
        Rsurf2 = torch.outer(e2, fr_normalized)
        Rsurf3 = torch.outer(e3, fr_normalized)
        Rsurf4 = torch.outer(e4, fr_normalized)

        # Data
        Rdata1 = x[:, 0, :]
        Rdata2 = x[:, 1, :]
        Rdata3 = x[:, 2, :]
        Rdata4 = x[:, 3, :]

        # Predicted data
        Rpred1 = Rdif + Rsurf1
        Rpred2 = Rdif + Rsurf2
        Rpred3 = Rdif + Rsurf3
        Rpred4 = Rdif + Rsurf4

        # Loss calculations
        # Data loss
        Loss_angle1 = (Rdata1-Rpred1)**2
        Loss_angle2 = (Rdata2-Rpred2)**2
        Loss_angle3 = (Rdata3-Rpred3)**2
        Loss_angle4 = (Rdata4-Rpred4)**2
        total_loss_data = (Loss_angle1.sum() + Loss_angle2.sum() + Loss_angle3.sum() + Loss_angle4.sum())

        # Physics loss
        Loss_fractions1 = ((1 - (MvF + SvF1)) ** 2)
        Loss_fractions2 = ((1 - (BvF + SvF2)) ** 2)
        # Total loss
        loss = total_loss_data + Loss_fractions1.sum() + Loss_fractions2.sum()

        return loss, Rdata1[0, :], Rdata2[0, :], Rdata3[0, :], Rdata4[0, :], Rpred1[0, :], Rpred2[0, :], Rpred3[0, :], Rpred4[0, :]

