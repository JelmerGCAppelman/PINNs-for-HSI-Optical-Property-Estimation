import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import time
from PINN import PMLP_OTCM, PMLP_TTCM, T_OTCM, T_TTCM
import torch
import args_parse
from pathlib import Path

args = args_parse.args_parser()

class Analysis:
    def __init__(self, model_dates):

        self.model_dates = model_dates

        # Absorption coefficient matrix
        ROOT = Path(__file__).resolve().parent

        self.data_absorption = np.load(
            ROOT /
            "Absorptioncoefficients" /
            "absorption_coefficient_matrix.npy"
        )

        self.std_approximated_noise = np.load(
            ROOT /
            "Noise std" /
            "Model" /
            "estimated_std_noise.npy"
        )

        self.wavelengths = np.linspace(args.min_wavelength, args.max_wavelength, args.n_wavelengths)
        # Create base directory to save
        self.base_save_dir = ROOT / "Evaluation"
        self.base_save_dir.mkdir(parents=True, exist_ok=True)

        # Helper function to load data
        def load_test_dataset(model_name):
            base = (
                    ROOT /
                    "PINN" /
                    model_name /
                    "Dataset final model" /
                    "Testing"
            )

            spectra = np.load(
                base / "Spectra" / "spectra.npy",
                allow_pickle=True
            )

            parameters = np.load(
                base / "Parameters" / "parameters.npy",
                allow_pickle=True
            )

            return spectra, parameters

        # Load the spectra and the parameters
        PMLP_OTCM_data_spectra, PMLP_OTCM_data_parameters = load_test_dataset("PMLP_OTCM")
        T_OTCM_data_spectra, T_OTCM_data_parameters = load_test_dataset("T_OTCM")
        PMLP_TTCM_data_spectra, PMLP_TTCM_data_parameters = load_test_dataset("PMLP_TTCM")
        T_TTCM_data_spectra, T_TTCM_data_parameters = load_test_dataset("T_TTCM")

        # Evaluation data
        self.PMLP_OTCM_spectra = torch.Tensor(np.squeeze(PMLP_OTCM_data_spectra))
        self.PMLP_OTCM_parameters = np.squeeze(PMLP_OTCM_data_parameters)

        self.T_OTCM_spectra = torch.Tensor(np.squeeze(T_OTCM_data_spectra))
        self.T_OTCM_parameters = np.squeeze(T_OTCM_data_parameters)

        self.PMLP_TTCM_spectra = torch.Tensor(np.squeeze(PMLP_TTCM_data_spectra))
        self.PMLP_TTCM_parameters = np.squeeze(PMLP_TTCM_data_parameters)

        self.T_TTCM_spectra = torch.Tensor(np.squeeze(T_TTCM_data_spectra))
        self.T_TTCM_parameters = np.squeeze(T_TTCM_data_parameters)


        # ==========================
        # Load trained models
        # ==========================

        self.loaded_models = {}

        model_classes = {
            "PMLP_OTCM": lambda: PMLP_OTCM(
                hidden_size=128,
                num_hidden_layers=3,
                use_dropout=True
            ),

            "PMLP_TTCM": lambda: PMLP_TTCM(
                hidden_size=128,
                num_hidden_layers=3,
                use_dropout=True
            ),

            "T_OTCM": lambda: T_OTCM(
                d_model=64,
                n_heads=4,
                n_layers=2,
                d_spec_set=64,
                n_heads_set=2,
                n_layers_set=2
            ),

            "T_TTCM": lambda: T_TTCM(
                d_model=64,
                n_heads=4,
                n_layers=2,
                d_spec_set=64,
                n_heads_set=2,
                n_layers_set=2
            )
        }

        for model_name, date in self.model_dates.items():
            checkpoint_path = (
                    ROOT /
                    "PINN" /
                    model_name /
                    "Results" /
                    "Final model" /
                    date /
                    "checkpoint final model" /
                    "model.pkl"
            )

            model = model_classes[model_name]()

            model.load_state_dict(
                torch.load(
                    checkpoint_path,
                    map_location="cpu"
                )
            )

            model.eval()

            self.loaded_models[model_name] = model

    def sample_from_gaussian(self):
        mean = np.zeros_like(self.std_approximated_noise)  # Mean is set to zero
        sampled_values = np.random.normal(mean, self.std_approximated_noise)
        return sampled_values

    def calculate_relative_absolute_error(self, vector1, vector2):

        vector1 = np.array(vector1)
        vector2 = np.array(vector2)

        # Ensure both vectors are of the same length
        if vector1.shape != vector2.shape:
            raise ValueError("Vectors must have the same length.")

        # Calculate absolute error for each point
        relative_absolute_errors = (np.abs(vector1 - vector2) / vector1) * 100

        return relative_absolute_errors

    # PMLP_OTCM spectra and PMLP_OTCM spectra + noise
    def OTCM_spectra_synthesize(self, BvF=0.02, Hb_sat=1, R_vessel=0.003, SvF=0.97,
                                MvF=0.001, a1=45.3, b1=2.453, c_hb=4.7,
                                e1=0.05, e2=0.01, e3=0.02, e4=0.14):

        # Define constant
        wavelengths = np.linspace(450, 818, 47)
        # Define absorption coeficients for different wavelengths
        ua_skin_baseline = self.data_absorption[0]
        ua_melanin = self.data_absorption[5]
        ua_oxyhaem = self.data_absorption[1]
        ua_deoxyhaem = self.data_absorption[2]
        refractive_index_skin = self.data_absorption[6]
        refractive_index_air = self.data_absorption[7]
        ua_whole_blood = ua_oxyhaem * Hb_sat + ua_deoxyhaem * (1 - Hb_sat)

        # Determining vessel factor
        C_diff = (1 - np.exp(-2 * ua_whole_blood * R_vessel)) / (2 * R_vessel)

        # Absorption coefficient
        ua = abs((BvF * args.c_hb * C_diff) + (ua_skin_baseline * SvF) + (ua_melanin * MvF))

        # Scattering coefficient
        us = a1 * ((wavelengths / 500) ** -b1)

        # Defining alpha
        a = us / (us + ua)

        # Relative refractive index
        n_rel = refractive_index_skin / refractive_index_air

        # Calculate rd
        rd = (-1.44 * n_rel ** -2) + (0.71 * n_rel ** -1) + (0.668 + 0.0636 * n_rel)

        # Calculate k
        k = (1 + rd) / (1 - rd)

        # Determine Diffusion theory part
        Rdif = a / (1 + 2 * k * (1 - a) + (1 + 2 * k / 3) * np.sqrt(3 * (1 - a)))

        # Fresnel
        fr = ((refractive_index_skin - refractive_index_air) / (refractive_index_skin + refractive_index_air)) ** 2
        normalized_fr = fr / fr[0]

        # Surface reflectance per angle
        Rsurf1 = e1 * normalized_fr
        Rsurf2 = e2 * normalized_fr
        Rsurf3 = e3 * normalized_fr
        Rsurf4 = e4 * normalized_fr

        # Make the predictions
        pred1 = Rdif + Rsurf1
        pred2 = Rdif + Rsurf2
        pred3 = Rdif + Rsurf3
        pred4 = Rdif + Rsurf4

        return pred1, pred2, pred3, pred4

    def OTCM_spectra_and_noise(self, BvF=0.02, Hb_sat=0.8, SvF=0.97, MvF=0.001,
                               R_vessel=0.003, a1=45.3, b1=2.453, c_hb=4.7,
                               e1=0.05, e2=0.01, e3=0.02, e4=0.14):

        # Define constant
        wavelengths = np.linspace(450, 818, 47)

        # Defining the right parameters
        ua_skinbaseline = self.data_absorption[0]
        ua_oxyhaem = self.data_absorption[1]
        ua_deoxyhaem = self.data_absorption[2]
        ua_melanin = self.data_absorption[5]
        refractive_index_skin = self.data_absorption[6]
        refractive_index_air = self.data_absorption[7]
        ua_whole_blood = ua_oxyhaem * Hb_sat + ua_deoxyhaem * (1 - Hb_sat)

        # Determining vessel factor
        C_diff = (1 - np.exp(-2 * ua_whole_blood * R_vessel)) / (2 * R_vessel)

        # Absorption coefficient
        ua = abs((BvF * args.c_hb * C_diff) + (ua_skinbaseline * SvF) + (ua_melanin * MvF))

        # Scattering coefficient
        us = a1 * ((wavelengths / 500) ** -b1)

        # Defining alpha
        a = us / (us + ua)

        # Relative refractive index
        n_rel = refractive_index_skin / refractive_index_air

        # Calculate rd
        rd = (-1.44 * n_rel ** -2) + (0.71 * n_rel ** -1) + (0.668 + 0.0636 * n_rel)

        # Calculate k
        k = (1 + rd) / (1 - rd)

        # Determine Diffusion theory part
        Rdif = a / (1 + 2 * k * (1 - a) + (1 + 2 * k / 3) * np.sqrt(3 * (1 - a)))

        # Fresnel
        fr = ((refractive_index_skin - refractive_index_air) / (refractive_index_skin + refractive_index_air)) ** 2
        normalized_fr = fr / fr[0]

        # Surface reflectance per angle
        Rsurf1 = e1 * normalized_fr
        Rsurf2 = e2 * normalized_fr
        Rsurf3 = e3 * normalized_fr
        Rsurf4 = e4 * normalized_fr

        # Noise
        noise_pred1 = self.sample_from_gaussian()
        noise_pred2 = self.sample_from_gaussian()
        noise_pred3 = self.sample_from_gaussian()
        noise_pred4 = self.sample_from_gaussian()

        # Make the predictions
        pred1 = Rdif + Rsurf1 + noise_pred1
        pred2 = Rdif + Rsurf2 + noise_pred2
        pred3 = Rdif + Rsurf3 + noise_pred3
        pred4 = Rdif + Rsurf4 + noise_pred4

        return pred1, pred2, pred3, pred4

    # TTCM spectra and TTCM spectra + noise
    def TTCM_spectra_synthesize(self, MvF=0.01, SvF1=0.99, d1=0.025, a1=68.7, b1=1.161,
                                BvF=0.008, Hb_sat=0.9, R_vessel=0.002, SvF2=0.992,
                                a2=45.3, b2=2.453, e1=0.1, e2=0.05, e3=0.1, e4=0.15):

        # Define constant
        wavelengths = np.linspace(450, 818, 47)
        # Define absorption coeficients for different wavelengths
        ua_skin_baseline = self.data_absorption[0]
        ua_melanin = self.data_absorption[5]
        ua_oxyhaem = self.data_absorption[1]
        ua_deoxyhaem = self.data_absorption[2]
        refractive_index_skin = self.data_absorption[6]
        refractive_index_air = self.data_absorption[7]

        # Define the absorption coefficient of whole blood
        ua_whole_blood = ua_oxyhaem * Hb_sat + ua_deoxyhaem * (1 - Hb_sat)
        # Define the epidermal layer
        # Absorptioncoefficient epidermal layer
        ua1 = (SvF1 * ua_skin_baseline) + (MvF * ua_melanin)
        # Scatteringcoefficient epidermal layer
        us1 = a1 * ((wavelengths / 500) ** -b1)
        # transport coefficient epidermal leyer
        ueff1 = ua1 + us1
        # Diffusion coefficient D epidermal layer
        D1 = 1 / (3 * ueff1)
        D1_squared = D1 ** 2
        # Optical penetration depth epidermal layer
        delta1 = np.sqrt(1 / (3 * ueff1 * ua1))
        delta1_squared = delta1 ** 2
        delta1_cube_squared = delta1 ** 3

        # Define the dermal layer
        # Determining vessel factor
        C_diff = (1 - np.exp(-2 * ua_whole_blood * R_vessel)) / (2 * R_vessel)
        # Absorptioncoefficient epidermal layer
        ua2 = (SvF2 * ua_skin_baseline) + (BvF * args.c_hb * C_diff)
        # Scatteringcoefficient epidermal layer
        us2 = a2 * ((wavelengths / 500) ** -b2)
        # transport coefficient epidermal leyer
        ueff2 = ua2 + us2
        # Diffusion coefficient D epidermal layer
        D2 = 1 / (3 * ueff2)
        # Optical penetration depth dermis
        delta2 = np.sqrt(1 / (3 * ueff2 * ua2))
        delta2_squared = delta2 ** 2

        # Total internal coefficient A
        r_eff = args.r_eff
        # Calculate k
        A = (1 - r_eff) / (2 * (1 + r_eff))
        # Calculating the reflection coefficient
        y_A = delta1 * us1 * A
        # From spott et al, Svaasand et al
        y_B = (1 + (delta2 / (3 * D2))) * np.abs(
            (((delta1_squared * delta2) / 3) - (delta1_squared * D2)) * np.cosh(d1 / delta1) + (
                    ((delta1_cube_squared * D2) / (3 * D1)) - (delta1 * delta2 * D1)) * np.sinh(d1 / delta1))
        y_C = np.abs(delta2_squared * D1 * (us2 / us1) * ((delta1_squared / (9 * D1_squared)) - 1) + (
                delta1_squared * (D2 - (delta2_squared / (9 * D2)))) * np.exp(-d1 / (3 * D1)))
        y_D = ((delta1_squared / (9 * D1_squared)) + 1) * ((delta2 / (3 * D2) + 1)) * np.abs(
            (D1 * delta1 * (D2 + (delta2 * A)) * np.cosh(d1 / delta1)) + (
                    ((D1_squared * delta2) + (D2 * delta1_squared * A)) * np.sinh(d1 / delta1)))
        Rdif = y_A * ((y_B + y_C) / y_D)

        # Surface reflectance
        # Fresnel
        fr = ((refractive_index_skin - refractive_index_air) / (refractive_index_skin + refractive_index_air)) ** 2
        normalized_fr = fr / fr[0]

        # Surface reflectance per angle
        Rsurf1 = e1 * normalized_fr
        Rsurf2 = e2 * normalized_fr
        Rsurf3 = e3 * normalized_fr
        Rsurf4 = e4 * normalized_fr

        # Make the predictions
        pred1 = Rdif + Rsurf1
        pred2 = Rdif + Rsurf2
        pred3 = Rdif + Rsurf3
        pred4 = Rdif + Rsurf4

        return pred1, pred2, pred3, pred4

    def TTCM_spectra_and_noise(self, BvF=0.015, Hb_sat=0.8, R_vessel=0.003, c_hb=4.7,
                               MvF=0.02, a1=68.7, b1=1.161, a2=45.3,
                               b2=2.453, d1=0.01, e1=0, e2=0, e3=0, e4=0):

        # Define constant
        wavelengths = np.linspace(450, 818, 47)

        # Defining the right parameters
        ua_skinbaseline = self.data_absorption[0]
        ua_oxyhaem = self.data_absorption[1]
        ua_deoxyhaem = self.data_absorption[2]
        ua_melanin = self.data_absorption[5]
        refractive_index_skin = self.data_absorption[6]
        refractive_index_air = self.data_absorption[7]

        # Define the absorption coefficient of whole blood
        ua_whole_blood = ua_oxyhaem * Hb_sat + ua_deoxyhaem * (1 - Hb_sat)
        # Define the epidermal layer
        # Skin volume fraction epidermus
        SvF1 = 1 - MvF
        # Absorption-coefficient epidermal layer
        ua1 = (SvF1 * ua_skinbaseline) + (MvF * ua_melanin)
        # Scattering-coefficient epidermal layer
        us1 = a1 * ((wavelengths / 500) ** -b1)
        # transport coefficient epidermal layer
        ueff1 = ua1 + us1
        # Diffusion coefficient D epidermal layer
        D1 = 1 / (3 * ueff1)
        D1_squared = D1 ** 2
        # Optical penetration depth epidermal layer
        delta1 = np.sqrt(1 / (3 * ueff1 * ua1))
        delta1_squared = delta1 ** 2
        delta1_cube_squared = delta1 ** 3

        # Define the dermal layer
        # Skin volume fraction dermis
        SvF2 = 1 - BvF
        # Determining vessel factor
        C_diff = (1 - np.exp(-2 * ua_whole_blood * R_vessel)) / (2 * R_vessel)
        # Absorption-coefficient epidermal layer
        ua2 = (SvF2 * ua_skinbaseline) + (BvF * args.c_hb * C_diff)
        # Scattering-coefficient epidermal layer
        us2 = a2 * ((wavelengths / 500) ** -b2)
        # transport coefficient epidermal leyer
        ueff2 = ua2 + us2
        # Diffusion coefficient D epidermal layer
        D2 = 1 / (3 * ueff2)
        # Optical penetration depth dermis
        delta2 = np.sqrt(1 / (3 * ueff2 * ua2))
        delta2_squared = delta2 ** 2

        # Total internal coefficient A
        r_eff = 0.439
        # Calculate k
        A = (1 - r_eff) / (2 * (1 + r_eff))
        # Calculating the reflection coefficient
        y_A = delta1 * us1 * A
        # From spott et al, Svaasand et al
        y_B = (1 + (delta2 / (3 * D2))) * np.abs(
            (((delta1_squared * delta2) / 3) - (delta1_squared * D2)) * np.cosh(d1 / delta1) + (
                    ((delta1_cube_squared * D2) / (3 * D1)) - (delta1 * delta2 * D1)) * np.sinh(d1 / delta1))
        y_C = np.abs(delta2_squared * D1 * (us2 / us1) * ((delta1_squared / (9 * D1_squared)) - 1) + (
                delta1_squared * (D2 - (delta2_squared / (9 * D2)))) * np.exp(-d1 / (3 * D1)))
        y_D = ((delta1_squared / (9 * D1_squared)) + 1) * ((delta2 / (3 * D2) + 1)) * np.abs(
            (D1 * delta1 * (D2 + (delta2 * A)) * np.cosh(d1 / delta1)) + (
                    ((D1_squared * delta2) + (D2 * delta1_squared * A)) * np.sinh(d1 / delta1)))
        y = y_A * ((y_B + y_C) / y_D)

        # Surface reflectance
        # Fresnel
        fr = ((refractive_index_skin - refractive_index_air) / (refractive_index_skin + refractive_index_air)) ** 2
        normalized_fr = fr / fr[0]

        # Surface reflectance per angle
        Rsurf1 = e1 * normalized_fr
        Rsurf2 = e2 * normalized_fr
        Rsurf3 = e3 * normalized_fr
        Rsurf4 = e4 * normalized_fr

        # Noise
        noise_pred1 = self.sample_from_gaussian()
        noise_pred2 = self.sample_from_gaussian()
        noise_pred3 = self.sample_from_gaussian()
        noise_pred4 = self.sample_from_gaussian()

        # Make the predictions
        pred1 = y + Rsurf1 + noise_pred1
        pred2 = y + Rsurf2 + noise_pred2
        pred3 = y + Rsurf3 + noise_pred3
        pred4 = y + Rsurf4 + noise_pred4

        return pred1, pred2, pred3, pred4


    # Synthetic data analysis

    # Hexabin density plots
    def hexabin_plot(self, current_model="PMLP_TTCM", parameter="b2"):

        if current_model not in self.loaded_models:
            raise ValueError(
                f"{current_model} is not loaded. Add it to model_dates when creating Analysis()."
            )

        param_latex = {
            "MvF": {"base": r"f_{\mathrm{mel}}", "true": r"f_{\mathrm{mel,true}}", "pred": r"f_{\mathrm{mel,pred}}"},
            "BvF": {"base": r"f_{\mathrm{blood}}", "true": r"f_{\mathrm{blood,true}}",
                    "pred": r"f_{\mathrm{blood,pred}}"},
            "S": {"base": r"S", "true": r"S_{\mathrm{true}}", "pred": r"S_{\mathrm{pred}}"},
            "Rv": {"base": r"R_v", "true": r"R_{v,\mathrm{true}}", "pred": r"R_{v,\mathrm{pred}}"},
            "b2": {"base": r"\beta_2", "true": r"\beta_{2,\mathrm{true}}", "pred": r"\beta_{2,\mathrm{pred}}"},
            "e1": {"base": r"\epsilon", "true": r"\epsilon_{\mathrm{true}}", "pred": r"\epsilon_{\mathrm{pred}}"},
            "e2": {"base": r"\epsilon", "true": r"\epsilon_{\mathrm{true}}", "pred": r"\epsilon_{\mathrm{pred}}"},
            "e3": {"base": r"\epsilon", "true": r"\epsilon_{\mathrm{true}}", "pred": r"\epsilon_{\mathrm{pred}}"},
            "e4": {"base": r"\epsilon", "true": r"\epsilon_{\mathrm{true}}", "pred": r"\epsilon_{\mathrm{pred}}"},
        }

        model_info = {
            "PMLP_OTCM": {
                "title": "PMLP-OTCM",
                "color": "Blues",
                "spectra": self.PMLP_OTCM_spectra,
                "parameters": self.PMLP_OTCM_parameters,
                "indices": {
                    "BvF": 0,
                    "S": 1,
                    "Rv": 2,
                    "MvF": 4,
                    "b2": 6,
                    "e1": 7,
                    "e2": 8,
                    "e3": 9,
                    "e4": 10,
                },
                "output_axis": "parameter_first",
            },

            "PMLP_TTCM": {
                "title": "PMLP-TTCM",
                "color": "Reds",
                "spectra": self.PMLP_TTCM_spectra,
                "parameters": self.PMLP_TTCM_parameters,
                "indices": {
                    "MvF": 0,
                    "BvF": 5,
                    "S": 6,
                    "Rv": 7,
                    "b2": 10,
                    "e1": 11,
                    "e2": 12,
                    "e3": 13,
                    "e4": 14,
                },
                "output_axis": "parameter_first",
            },

            "T_OTCM": {
                "title": "T-OTCM",
                "color": "Oranges",
                "spectra": self.T_OTCM_spectra,
                "parameters": self.T_OTCM_parameters,
                "indices": {
                    "BvF": 0,
                    "S": 1,
                    "Rv": 2,
                    "MvF": 4,
                    "b2": 6,
                    "e1": 7,
                    "e2": 8,
                    "e3": 9,
                    "e4": 10,
                },
                "output_axis": "sample_first",
            },

            "T_TTCM": {
                "title": "T-TTCM",
                "color": "Greens",
                "spectra": self.T_TTCM_spectra,
                "parameters": self.T_TTCM_parameters,
                "indices": {
                    "MvF": 0,
                    "BvF": 5,
                    "S": 6,
                    "Rv": 7,
                    "b2": 10,
                    "e1": 11,
                    "e2": 12,
                    "e3": 13,
                    "e4": 14,
                },
                "output_axis": "sample_first",
            },
        }

        if parameter not in param_latex:
            raise ValueError(f"Unknown parameter: {parameter}")

        info = model_info[current_model]

        outputs = (
            self.loaded_models[current_model](info["spectra"])
            .detach()
            .cpu()
            .numpy()
        )

        parameter_index = info["indices"][parameter]

        true_parameter = info["parameters"][:, parameter_index]

        if info["output_axis"] == "parameter_first":
            predicted_parameter = outputs[parameter_index, :]
        else:
            predicted_parameter = outputs[:, parameter_index]

        abs_error = np.abs(predicted_parameter - true_parameter)

        if parameter == "Rv":
            mean_error = np.mean(abs_error) * 10000
            std_error = np.std(abs_error) * 10000
            print(f"{parameter} {current_model}: mean absolute error = {mean_error:.3f} um ± {std_error:.3f} um")
        elif parameter in ["MvF", "BvF", "S"]:
            mean_error = np.mean(abs_error) * 100
            std_error = np.std(abs_error) * 100
            print(f"{parameter} {current_model}: mean absolute error = {mean_error:.3f}% ± {std_error:.3f}%")
        else:
            mean_error = np.mean(abs_error)
            std_error = np.std(abs_error)
            print(f"{parameter} {current_model}: mean absolute error = {mean_error:.3f} ± {std_error:.3f}")

        save_path = (
                self.base_save_dir /
                "Synthetic" /
                "Plots" /
                "Hexabin" /
                parameter /
                f"true_predicted_{parameter}_{current_model}.png"
        )

        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(14, 12))

        plt.plot(
            true_parameter,
            true_parameter,
            color="black",
            linewidth=1.5,
            label="True Line"
        )

        hb = plt.hexbin(
            true_parameter,
            predicted_parameter,
            gridsize=100,
            cmap=info["color"],
            alpha=0.7,
            norm=LogNorm(vmin=1, vmax=100)
        )

        cbar = plt.colorbar(hb)
        cbar.set_label("Density", fontsize=24)
        cbar.ax.tick_params(labelsize=18)

        plt.xlabel(rf"${param_latex[parameter]['true']}$", fontsize=32)
        plt.ylabel(rf"${param_latex[parameter]['pred']}$", fontsize=32)
        plt.title(
            rf"{info['title']} — ${param_latex[parameter]['base']}$",
            fontsize=32,
            pad=20
        )

        plt.grid(False)
        plt.legend(fontsize=24, loc="best")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    # Reconstructed spectra
    def reconstruction_synthetic(self):

        required_models = [
            "PMLP_OTCM",
            "PMLP_TTCM",
            "T_OTCM",
            "T_TTCM"
        ]

        missing = [
            model_name for model_name in required_models
            if model_name not in self.loaded_models
        ]

        if missing:
            raise ValueError(
                f"reconstruction_synthetic requires these models to be loaded: {missing}"
            )

        save_path_OTCM = (
                self.base_save_dir /
                "Synthetic" /
                "Plots" /
                "Reconstruction" /
                "OTCM_reconstruction_synthetic.png"
        )

        save_path_TTCM = (
                self.base_save_dir /
                "Synthetic" /
                "Plots" /
                "Reconstruction" /
                "example_reconstruction.png"
        )

        save_path_OTCM.parent.mkdir(parents=True, exist_ok=True)
        save_path_TTCM.parent.mkdir(parents=True, exist_ok=True)

        angle1_OTCM, angle2_OTCM, angle3_OTCM, angle4_OTCM = self.OTCM_spectra_and_noise(
            BvF=0.01,
            Hb_sat=0.8,
            SvF=0.98,
            MvF=0.01,
            R_vessel=0.0017,
            a1=45.3,
            b1=2.453,
            c_hb=4.7,
            e1=0.08,
            e2=0.03,
            e3=0.02,
            e4=0.05
        )

        angle1_TTCM, angle2_TTCM, angle3_TTCM, angle4_TTCM = self.TTCM_spectra_and_noise(
            BvF=0.01,
            Hb_sat=0.8,
            R_vessel=0.0017,
            c_hb=4.7,
            MvF=0.01,
            a1=68.7,
            b1=1.161,
            a2=45.3,
            b2=2.453,
            d1=0.01,
            e1=0.08,
            e2=0.03,
            e3=0.02,
            e4=0.05
        )

        phi1_OTCM = torch.tensor(angle1_OTCM, dtype=torch.float32)
        phi2_OTCM = torch.tensor(angle2_OTCM, dtype=torch.float32)
        phi3_OTCM = torch.tensor(angle3_OTCM, dtype=torch.float32)
        phi4_OTCM = torch.tensor(angle4_OTCM, dtype=torch.float32)

        stacked_OTCM = torch.stack(
            [phi1_OTCM, phi2_OTCM, phi3_OTCM, phi4_OTCM],
            dim=0
        )

        T_OTCM_final_tensor = stacked_OTCM.unsqueeze(0)

        OTCM_final_tensor = T_OTCM_final_tensor.view(1, -1)

        extra_row = torch.full(
            (1, OTCM_final_tensor.shape[1]),
            0.1,
            dtype=OTCM_final_tensor.dtype
        )

        OTCM_final_tensor = torch.cat(
            [OTCM_final_tensor, extra_row],
            dim=0
        )

        phi1_TTCM = torch.tensor(angle1_TTCM, dtype=torch.float32)
        phi2_TTCM = torch.tensor(angle2_TTCM, dtype=torch.float32)
        phi3_TTCM = torch.tensor(angle3_TTCM, dtype=torch.float32)
        phi4_TTCM = torch.tensor(angle4_TTCM, dtype=torch.float32)

        stacked_TTCM = torch.stack(
            [phi1_TTCM, phi2_TTCM, phi3_TTCM, phi4_TTCM],
            dim=0
        )

        T_TTCM_final_tensor = stacked_TTCM.unsqueeze(0)

        TTCM_final_tensor = T_TTCM_final_tensor.view(1, -1)

        extra_row = torch.full(
            (1, TTCM_final_tensor.shape[1]),
            0.1,
            dtype=TTCM_final_tensor.dtype
        )

        TTCM_final_tensor = torch.cat(
            [TTCM_final_tensor, extra_row],
            dim=0
        )

        # ========================
        # Predictions PMLP_OTCM
        # ========================
        color_OTCM = "#1f77b4"

        outputs_OTCM = (
            self.loaded_models["PMLP_OTCM"](
                OTCM_final_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )

        pred_MvF_OTCM = outputs_OTCM[4, 0]
        pred_BvF_OTCM = outputs_OTCM[0, 0]
        pred_S_OTCM = outputs_OTCM[1, 0]
        pred_Rv_OTCM = outputs_OTCM[2, 0]
        pred_SvF1_OTCM = outputs_OTCM[3, 0]
        pred_e1_OTCM = outputs_OTCM[7, 0]
        pred_e2_OTCM = outputs_OTCM[8, 0]
        pred_e3_OTCM = outputs_OTCM[9, 0]
        pred_e4_OTCM = outputs_OTCM[10, 0]

        predicted_spectrum_OTCM, _, _, _ = self.OTCM_spectra_synthesize(
            BvF=pred_BvF_OTCM,
            Hb_sat=pred_S_OTCM,
            R_vessel=pred_Rv_OTCM,
            SvF=pred_SvF1_OTCM,
            MvF=pred_MvF_OTCM,
            a1=45.3,
            b1=2.453,
            c_hb=4.7,
            e1=pred_e1_OTCM,
            e2=pred_e2_OTCM,
            e3=pred_e3_OTCM,
            e4=pred_e4_OTCM
        )

        # ========================
        # Predictions PMLP_TTCM
        # ========================
        color_TTCM = "#d62728"

        outputs_TTCM = (
            self.loaded_models["PMLP_TTCM"](
                TTCM_final_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )

        pred_MvF_TTCM = outputs_TTCM[0, 0]
        pred_SvF1_TTCM = outputs_TTCM[1, 0]
        pred_BvF_TTCM = outputs_TTCM[5, 0]
        pred_S_TTCM = outputs_TTCM[6, 0]
        pred_Rv_TTCM = outputs_TTCM[7, 0]
        pred_SvF2_TTCM = outputs_TTCM[8, 0]
        pred_e1_TTCM = outputs_TTCM[11, 0]
        pred_e2_TTCM = outputs_TTCM[12, 0]
        pred_e3_TTCM = outputs_TTCM[13, 0]
        pred_e4_TTCM = outputs_TTCM[14, 0]

        predicted_spectrum_TTCM, _, _, _ = self.TTCM_spectra_synthesize(
            MvF=pred_MvF_TTCM,
            SvF1=pred_SvF1_TTCM,
            d1=0.01,
            a1=68.7,
            b1=1.161,
            BvF=pred_BvF_TTCM,
            Hb_sat=pred_S_TTCM,
            R_vessel=pred_Rv_TTCM,
            SvF2=pred_SvF2_TTCM,
            a2=45.3,
            b2=2.453,
            e1=pred_e1_TTCM,
            e2=pred_e2_TTCM,
            e3=pred_e3_TTCM,
            e4=pred_e4_TTCM
        )

        # ========================
        # Predictions T_OTCM
        # ========================
        color_T_OTCM = "#ff7f0e"

        outputs_T_OTCM = np.squeeze(
            self.loaded_models["T_OTCM"](
                T_OTCM_final_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )

        pred_MvF_T_OTCM = outputs_T_OTCM[4]
        pred_BvF_T_OTCM = outputs_T_OTCM[0]
        pred_S_T_OTCM = outputs_T_OTCM[1]
        pred_Rv_T_OTCM = outputs_T_OTCM[2]
        pred_SvF1_T_OTCM = outputs_T_OTCM[3]
        pred_e1_T_OTCM = outputs_T_OTCM[7]
        pred_e2_T_OTCM = outputs_T_OTCM[8]
        pred_e3_T_OTCM = outputs_T_OTCM[9]
        pred_e4_T_OTCM = outputs_T_OTCM[10]

        predicted_spectrum_T_OTCM, _, _, _ = self.OTCM_spectra_synthesize(
            BvF=pred_BvF_T_OTCM,
            Hb_sat=pred_S_T_OTCM,
            R_vessel=pred_Rv_T_OTCM,
            SvF=pred_SvF1_T_OTCM,
            MvF=pred_MvF_T_OTCM,
            a1=45.3,
            b1=2.453,
            c_hb=4.7,
            e1=pred_e1_T_OTCM,
            e2=pred_e2_T_OTCM,
            e3=pred_e3_T_OTCM,
            e4=pred_e4_T_OTCM
        )

        # ========================
        # Predictions T_TTCM
        # ========================
        color_T_TTCM = "#32a852"

        outputs_T_TTCM = np.squeeze(
            self.loaded_models["T_TTCM"](
                T_TTCM_final_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )

        pred_MvF_T_TTCM = outputs_T_TTCM[0]
        pred_SvF1_T_TTCM = outputs_T_TTCM[1]
        pred_BvF_T_TTCM = outputs_T_TTCM[5]
        pred_S_T_TTCM = outputs_T_TTCM[6]
        pred_Rv_T_TTCM = outputs_T_TTCM[7]
        pred_SvF2_T_TTCM = outputs_T_TTCM[8]
        pred_e1_T_TTCM = outputs_T_TTCM[11]
        pred_e2_T_TTCM = outputs_T_TTCM[12]
        pred_e3_T_TTCM = outputs_T_TTCM[13]
        pred_e4_T_TTCM = outputs_T_TTCM[14]

        predicted_spectrum_T_TTCM, _, _, _ = self.TTCM_spectra_synthesize(
            MvF=pred_MvF_T_TTCM,
            SvF1=pred_SvF1_T_TTCM,
            d1=0.01,
            a1=68.7,
            b1=1.161,
            BvF=pred_BvF_T_TTCM,
            Hb_sat=pred_S_T_TTCM,
            R_vessel=pred_Rv_T_TTCM,
            SvF2=pred_SvF2_T_TTCM,
            a2=45.3,
            b2=2.453,
            e1=pred_e1_T_TTCM,
            e2=pred_e2_T_TTCM,
            e3=pred_e3_T_TTCM,
            e4=pred_e4_T_TTCM,
        )

        # ========================
        # OTCM plot
        # ========================
        plt.figure(figsize=(14, 12))

        plt.plot(
            self.wavelengths,
            angle1_OTCM,
            label="Synthetic OTCM",
            color="black",
            linewidth=3.0
        )

        plt.plot(
            self.wavelengths,
            predicted_spectrum_OTCM,
            label="PMLP-OTCM",
            color=color_OTCM,
            linewidth=3.0
        )

        plt.plot(
            self.wavelengths,
            predicted_spectrum_T_OTCM,
            label="T-OTCM",
            color=color_T_OTCM,
            linewidth=3.0
        )

        plt.xlabel(r"$\lambda$(nm)", fontsize=32)
        plt.ylabel("Reflectance", fontsize=32)
        plt.title("OTCM", fontsize=32, pad=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylim(0, 1)
        plt.legend(fontsize=24)
        plt.tight_layout()
        plt.savefig(save_path_OTCM, dpi=300, bbox_inches="tight")
        plt.close()

        # ========================
        # TTCM plot
        # ========================
        plt.figure(figsize=(14, 12))

        plt.plot(
            self.wavelengths,
            angle1_TTCM,
            label="Synthetic TTCM",
            color="black",
            linewidth=3.0
        )

        plt.plot(
            self.wavelengths,
            predicted_spectrum_TTCM,
            label="PMLP-TTCM",
            color=color_TTCM,
            linewidth=3.0
        )

        plt.plot(
            self.wavelengths,
            predicted_spectrum_T_TTCM,
            label="T-TTCM",
            color=color_T_TTCM,
            linewidth=3.0
        )

        plt.xlabel(r"$\lambda$(nm)", fontsize=32)
        plt.ylabel("Reflectance", fontsize=32)
        plt.title("TTCM", fontsize=32, pad=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylim(0, 1)
        plt.legend(fontsize=24)
        plt.tight_layout()
        plt.savefig(save_path_TTCM, dpi=300, bbox_inches="tight")
        plt.close()

    # MRAE over the spectra
    def MRAE_synthetic(self):

        required_models = [
            "PMLP_OTCM",
            "PMLP_TTCM",
            "T_OTCM",
            "T_TTCM"
        ]

        missing = [
            model_name for model_name in required_models
            if model_name not in self.loaded_models
        ]

        if missing:
            raise ValueError(
                f"standard_deviation_synthetic requires these models to be loaded: {missing}"
            )

        # ========================
        # Predictions
        # ========================

        start = time.perf_counter()
        outputs_OTCM = (
            self.loaded_models["PMLP_OTCM"](self.PMLP_OTCM_spectra)
            .detach()
            .cpu()
            .numpy()
        )
        time_OTCM = time.perf_counter() - start
        print(f"PMLP_OTCM time: {time_OTCM:.6f} s")

        start = time.perf_counter()
        outputs_TTCM = (
            self.loaded_models["PMLP_TTCM"](self.PMLP_TTCM_spectra)
            .detach()
            .cpu()
            .numpy()
        )
        time_TTCM = time.perf_counter() - start
        print(f"PMLP_TTCM time: {time_TTCM:.6f} s")

        start = time.perf_counter()
        outputs_T_OTCM = (
            self.loaded_models["T_OTCM"](self.T_OTCM_spectra)
            .detach()
            .cpu()
            .numpy()
        )
        time_T_OTCM = time.perf_counter() - start
        print(f"T_OTCM time: {time_T_OTCM:.6f} s")

        start = time.perf_counter()
        outputs_T_TTCM = (
            self.loaded_models["T_TTCM"](self.T_TTCM_spectra)
            .detach()
            .cpu()
            .numpy()
        )
        time_T_TTCM = time.perf_counter() - start
        print(f"T_TTCM time: {time_T_TTCM:.6f} s")

        # ========================
        # Error lists
        # ========================

        OTCM_relative_absolute_error_list = []
        TTCM_relative_absolute_error_list = []
        T_OTCM_relative_absolute_error_list = []
        T_TTCM_relative_absolute_error_list = []

        n_samples = self.PMLP_OTCM_parameters.shape[0]

        for i in range(n_samples):
            # ========================
            # True parameters PMLP_OTCM
            # ========================

            true_BvF_OTCM = self.PMLP_OTCM_parameters[i, 0]
            true_S_OTCM = self.PMLP_OTCM_parameters[i, 1]
            true_Rv_OTCM = self.PMLP_OTCM_parameters[i, 2]
            true_SvF1_OTCM = self.PMLP_OTCM_parameters[i, 3]
            true_MvF_OTCM = self.PMLP_OTCM_parameters[i, 4]
            true_e1_OTCM = self.PMLP_OTCM_parameters[i, 7]
            true_e2_OTCM = self.PMLP_OTCM_parameters[i, 8]
            true_e3_OTCM = self.PMLP_OTCM_parameters[i, 9]
            true_e4_OTCM = self.PMLP_OTCM_parameters[i, 10]

            # ========================
            # True parameters PMLP_TTCM
            # ========================

            true_MvF_TTCM = self.PMLP_TTCM_parameters[i, 0]
            true_SvF1_TTCM = self.PMLP_TTCM_parameters[i, 1]
            true_BvF_TTCM = self.PMLP_TTCM_parameters[i, 5]
            true_S_TTCM = self.PMLP_TTCM_parameters[i, 6]
            true_Rv_TTCM = self.PMLP_TTCM_parameters[i, 7]
            true_SvF2_TTCM = self.PMLP_TTCM_parameters[i, 8]
            true_e1_TTCM = self.PMLP_TTCM_parameters[i, 11]
            true_e2_TTCM = self.PMLP_TTCM_parameters[i, 12]
            true_e3_TTCM = self.PMLP_TTCM_parameters[i, 13]
            true_e4_TTCM = self.PMLP_TTCM_parameters[i, 14]

            # ========================
            # Predicted parameters PMLP_OTCM
            # output shape: [parameters, samples]
            # ========================

            pred_BvF_OTCM = outputs_OTCM[0, i]
            pred_S_OTCM = outputs_OTCM[1, i]
            pred_Rv_OTCM = outputs_OTCM[2, i]
            pred_SvF1_OTCM = outputs_OTCM[3, i]
            pred_MvF_OTCM = outputs_OTCM[4, i]
            pred_e1_OTCM = outputs_OTCM[7, i]
            pred_e2_OTCM = outputs_OTCM[8, i]
            pred_e3_OTCM = outputs_OTCM[9, i]
            pred_e4_OTCM = outputs_OTCM[10, i]

            # ========================
            # Predicted parameters PMLP_TTCM
            # output shape: [parameters, samples]
            # ========================

            pred_MvF_TTCM = outputs_TTCM[0, i]
            pred_SvF1_TTCM = outputs_TTCM[1, i]
            pred_BvF_TTCM = outputs_TTCM[5, i]
            pred_S_TTCM = outputs_TTCM[6, i]
            pred_Rv_TTCM = outputs_TTCM[7, i]
            pred_SvF2_TTCM = outputs_TTCM[8, i]
            pred_e1_TTCM = outputs_TTCM[11, i]
            pred_e2_TTCM = outputs_TTCM[12, i]
            pred_e3_TTCM = outputs_TTCM[13, i]
            pred_e4_TTCM = outputs_TTCM[14, i]

            # ========================
            # Predicted parameters T_OTCM
            # output shape: [samples, parameters]
            # ========================

            pred_BvF_T_OTCM = outputs_T_OTCM[i, 0]
            pred_S_T_OTCM = outputs_T_OTCM[i, 1]
            pred_Rv_T_OTCM = outputs_T_OTCM[i, 2]
            pred_SvF1_T_OTCM = outputs_T_OTCM[i, 3]
            pred_MvF_T_OTCM = outputs_T_OTCM[i, 4]
            pred_e1_T_OTCM = outputs_T_OTCM[i, 7]
            pred_e2_T_OTCM = outputs_T_OTCM[i, 8]
            pred_e3_T_OTCM = outputs_T_OTCM[i, 9]
            pred_e4_T_OTCM = outputs_T_OTCM[i, 10]

            # ========================
            # Predicted parameters T_TTCM
            # output shape: [samples, parameters]
            # ========================

            pred_MvF_T_TTCM = outputs_T_TTCM[i, 0]
            pred_SvF1_T_TTCM = outputs_T_TTCM[i, 1]
            pred_BvF_T_TTCM = outputs_T_TTCM[i, 5]
            pred_S_T_TTCM = outputs_T_TTCM[i, 6]
            pred_Rv_T_TTCM = outputs_T_TTCM[i, 7]
            pred_SvF2_T_TTCM = outputs_T_TTCM[i, 8]
            pred_e1_T_TTCM = outputs_T_TTCM[i, 11]
            pred_e2_T_TTCM = outputs_T_TTCM[i, 12]
            pred_e3_T_TTCM = outputs_T_TTCM[i, 13]
            pred_e4_T_TTCM = outputs_T_TTCM[i, 14]

            # ========================
            # True spectra
            # ========================

            true_spectrum_OTCM, _, _, _ = self.OTCM_spectra_synthesize(
                BvF=true_BvF_OTCM,
                Hb_sat=true_S_OTCM,
                R_vessel=true_Rv_OTCM,
                SvF=true_SvF1_OTCM,
                MvF=true_MvF_OTCM,
                a1=45.3,
                b1=2.453,
                c_hb=4.7,
                e1=true_e1_OTCM,
                e2=true_e2_OTCM,
                e3=true_e3_OTCM,
                e4=true_e4_OTCM
            )

            true_spectrum_TTCM, _, _, _ = self.TTCM_spectra_synthesize(
                MvF=true_MvF_TTCM,
                SvF1=true_SvF1_TTCM,
                d1=0.01,
                a1=68.7,
                b1=1.161,
                BvF=true_BvF_TTCM,
                Hb_sat=true_S_TTCM,
                R_vessel=true_Rv_TTCM,
                SvF2=true_SvF2_TTCM,
                a2=45.3,
                b2=2.453,
                e1=true_e1_TTCM,
                e2=true_e2_TTCM,
                e3=true_e3_TTCM,
                e4=true_e4_TTCM,
            )

            # ========================
            # Predicted spectra
            # ========================

            pred_spectrum_OTCM, _, _, _ = self.OTCM_spectra_synthesize(
                BvF=pred_BvF_OTCM,
                Hb_sat=pred_S_OTCM,
                R_vessel=pred_Rv_OTCM,
                SvF=pred_SvF1_OTCM,
                MvF=pred_MvF_OTCM,
                a1=45.3,
                b1=2.453,
                c_hb=4.7,
                e1=pred_e1_OTCM,
                e2=pred_e2_OTCM,
                e3=pred_e3_OTCM,
                e4=pred_e4_OTCM
            )

            pred_spectrum_TTCM, _, _, _ = self.TTCM_spectra_synthesize(
                MvF=pred_MvF_TTCM,
                SvF1=pred_SvF1_TTCM,
                d1=0.01,
                a1=68.7,
                b1=1.161,
                BvF=pred_BvF_TTCM,
                Hb_sat=pred_S_TTCM,
                R_vessel=pred_Rv_TTCM,
                SvF2=pred_SvF2_TTCM,
                a2=45.3,
                b2=2.453,
                e1=pred_e1_TTCM,
                e2=pred_e2_TTCM,
                e3=pred_e3_TTCM,
                e4=pred_e4_TTCM,
            )

            pred_spectrum_T_OTCM, _, _, _ = self.OTCM_spectra_synthesize(
                BvF=pred_BvF_T_OTCM,
                Hb_sat=pred_S_T_OTCM,
                R_vessel=pred_Rv_T_OTCM,
                SvF=pred_SvF1_T_OTCM,
                MvF=pred_MvF_T_OTCM,
                a1=45.3,
                b1=2.453,
                c_hb=4.7,
                e1=pred_e1_T_OTCM,
                e2=pred_e2_T_OTCM,
                e3=pred_e3_T_OTCM,
                e4=pred_e4_T_OTCM
            )

            pred_spectrum_T_TTCM, _, _, _ = self.TTCM_spectra_synthesize(
                MvF=pred_MvF_T_TTCM,
                SvF1=pred_SvF1_T_TTCM,
                d1=0.01,
                a1=68.7,
                b1=1.161,
                BvF=pred_BvF_T_TTCM,
                Hb_sat=pred_S_T_TTCM,
                R_vessel=pred_Rv_T_TTCM,
                SvF2=pred_SvF2_T_TTCM,
                a2=45.3,
                b2=2.453,
                e1=pred_e1_T_TTCM,
                e2=pred_e2_T_TTCM,
                e3=pred_e3_T_TTCM,
                e4=pred_e4_T_TTCM,
            )

            # ========================
            # Relative absolute errors
            # ========================

            OTCM_RAE = self.calculate_relative_absolute_error(
                true_spectrum_OTCM,
                pred_spectrum_OTCM
            )

            TTCM_RAE = self.calculate_relative_absolute_error(
                true_spectrum_TTCM,
                pred_spectrum_TTCM
            )

            T_OTCM_RAE = self.calculate_relative_absolute_error(
                true_spectrum_OTCM,
                pred_spectrum_T_OTCM
            )

            T_TTCM_RAE = self.calculate_relative_absolute_error(
                true_spectrum_TTCM,
                pred_spectrum_T_TTCM
            )

            OTCM_relative_absolute_error_list.append(OTCM_RAE)
            TTCM_relative_absolute_error_list.append(TTCM_RAE)
            T_OTCM_relative_absolute_error_list.append(T_OTCM_RAE)
            T_TTCM_relative_absolute_error_list.append(T_TTCM_RAE)

        # ========================
        # Return arrays
        # ========================

        OTCM_relative_absolute_error = np.array(OTCM_relative_absolute_error_list)
        TTCM_relative_absolute_error = np.array(TTCM_relative_absolute_error_list)
        T_OTCM_relative_absolute_error = np.array(T_OTCM_relative_absolute_error_list)
        T_TTCM_relative_absolute_error = np.array(T_TTCM_relative_absolute_error_list)

        return (
            OTCM_relative_absolute_error,
            TTCM_relative_absolute_error,
            T_OTCM_relative_absolute_error,
            T_TTCM_relative_absolute_error
        )

    # MRAE plot
    def MRAE_plot_synthetic(self):

        save_path_MRAE = (
                self.base_save_dir /
                "Synthetic" /
                "Plots" /
                "MRAE" /
                "MRAE_synthetic.svg"
        )

        save_path_MRAE.parent.mkdir(parents=True, exist_ok=True)

        (
            OTCM_absolute_error,
            TTCM_absolute_error,
            T_OTCM_absolute_error,
            T_TTCM_absolute_error
        ) = self.MRAE_synthetic()

        MRAE_OTCM = np.mean(OTCM_absolute_error, axis=0)
        MRAE_TTCM = np.mean(TTCM_absolute_error, axis=0)
        MRAE_T_OTCM = np.mean(T_OTCM_absolute_error, axis=0)
        MRAE_T_TTCM = np.mean(T_TTCM_absolute_error, axis=0)

        OTCM_std = np.std(OTCM_absolute_error, axis=0)
        TTCM_std = np.std(TTCM_absolute_error, axis=0)
        T_OTCM_std = np.std(T_OTCM_absolute_error, axis=0)
        T_TTCM_std = np.std(T_TTCM_absolute_error, axis=0)

        colors = {
            "PMLP_OTCM": "#1f77b4",
            "PMLP_TTCM": "#d62728",
            "T_OTCM": "#ff7f0e",
            "T_TTCM": "#32a852",
        }

        plt.figure(figsize=(14, 12))

        plt.plot(self.wavelengths, MRAE_OTCM, label="PMLP-OTCM", color=colors["PMLP_OTCM"], linewidth=3.0)
        plt.fill_between(self.wavelengths, MRAE_OTCM - OTCM_std, MRAE_OTCM + OTCM_std, color=colors["PMLP_OTCM"],
                         alpha=0.3)

        plt.plot(self.wavelengths, MRAE_TTCM, label="PMLP-TTCM", color=colors["PMLP_TTCM"], linewidth=3.0)
        plt.fill_between(self.wavelengths, MRAE_TTCM - TTCM_std, MRAE_TTCM + TTCM_std, color=colors["PMLP_TTCM"],
                         alpha=0.3)

        plt.plot(self.wavelengths, MRAE_T_OTCM, label="T-OTCM", color=colors["T_OTCM"], linewidth=3.0)
        plt.fill_between(self.wavelengths, MRAE_T_OTCM - T_OTCM_std, MRAE_T_OTCM + T_OTCM_std, color=colors["T_OTCM"],
                         alpha=0.3)

        plt.plot(self.wavelengths, MRAE_T_TTCM, label="T-TTCM", color=colors["T_TTCM"], linewidth=3.0)
        plt.fill_between(self.wavelengths, MRAE_T_TTCM - T_TTCM_std, MRAE_T_TTCM + T_TTCM_std, color=colors["T_TTCM"],
                         alpha=0.3)

        plt.xlabel(r"$\lambda$ (nm)", fontsize=24)
        plt.ylabel("MRAE (%)", fontsize=24)
        plt.ylim(0, 10)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=24)
        plt.tight_layout()

        plt.savefig(save_path_MRAE, dpi=300, bbox_inches="tight")
        plt.close()
