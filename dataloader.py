import os
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import torch
import matplotlib.pyplot as plt
import args_parse
from pathlib import Path

args = args_parse.args_parser()

class Dataloader:

    def __init__(self):

        ROOT = Path(__file__).resolve().parent

        print(ROOT)
        base = ROOT / "PINN" / args.model

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

        # =====================================================
        # Grid search
        # =====================================================

        self.path_spectra_matrix_grid_search = (
                base /
                "Synthesized data" /
                "Spectra matrix" /
                "spectra.npy"
        )

        self.path_parameters_grid_search = (
                base /
                "Synthesized data" /
                "Parameter matrix" /
                "parameters.npy"
        )

        # =====================================================
        # Final model
        # =====================================================

        self.path_spectra_matrix_final_model = (
                base /
                "Synthesized data final model" /
                "Spectra matrix" /
                "spectra.npy"
        )

        self.path_parameters_final_model = (
                base /
                "Synthesized data final model" /
                "Parameter matrix" /
                "parameters.npy"
        )

        # =====================================================
        # Grid search datasets
        # =====================================================

        self.path_training_spectra = (
                base /
                "Dataset gridsearch" /
                "Training" /
                "Spectra" /
                "spectra"
        )

        self.path_training_parameters = (
                base /
                "Dataset gridsearch" /
                "Training" /
                "Parameters" /
                "parameters"
        )

        self.path_validation_spectra = (
                base /
                "Dataset gridsearch" /
                "Validation" /
                "Spectra" /
                "spectra"
        )

        self.path_validation_parameters = (
                base /
                "Dataset gridsearch" /
                "Validation" /
                "Parameters" /
                "parameters"
        )

        self.path_test_spectra = (
                base /
                "Dataset gridsearch" /
                "Testing" /
                "Spectra" /
                "spectra"
        )

        self.path_test_parameters = (
                base /
                "Dataset gridsearch" /
                "Testing" /
                "Parameters" /
                "parameters"
        )

        # =====================================================
        # Final model datasets
        # =====================================================

        self.path_training_spectra_final_model = (
                base /
                "Dataset final model" /
                "Training" /
                "Spectra" /
                "spectra"
        )

        self.path_training_parameters_final_model = (
                base /
                "Dataset final model" /
                "Training" /
                "Parameters" /
                "parameters"
        )

        self.path_validation_spectra_final_model = (
                base /
                "Dataset final model" /
                "Validation" /
                "Spectra" /
                "spectra"
        )

        self.path_validation_parameters_final_model = (
                base /
                "Dataset final model" /
                "Validation" /
                "Parameters" /
                "parameters"
        )

        self.path_test_spectra_final_model = (
                base /
                "Dataset final model" /
                "Testing" /
                "Spectra" /
                "spectra"
        )

        self.path_test_parameters_final_model = (
                base /
                "Dataset final model" /
                "Testing" /
                "Parameters" /
                "parameters"
        )

        if args.mode == "grid_search":
            self.samples = args.samples_grid_search
        else:
            self.samples = args.samples_final_model

    def sample_from_gaussian(self):
        mean = np.zeros_like(self.std_approximated_noise)  # Mean is set to zero
        sampled_values = np.random.normal(mean, self.std_approximated_noise)
        return sampled_values

    def spectrum_OTCM(self, make_plot=False):

        spectra_matrix = np.zeros((self.samples, args.n_wavelengths, args.n_angles))
        spectra_matrix_no_noise = np.zeros((self.samples, args.n_wavelengths, args.n_angles))
        parameters_matrix = np.zeros((self.samples, 11))

        for i in range(self.samples):
            BvF = args.BvF_min + (args.BvF_max - args.BvF_min) * np.random.rand()
            Hb_sat = args.S_min + (args.S_max - args.S_min) * np.random.rand()
            Rv = args.Rv_min + (args.Rv_max - args.Rv_min) * np.random.rand()
            MvF = args.MvF_min + (args.MvF_max - args.MvF_min) * np.random.rand()
            SvF = 1 - (BvF + MvF)
            a2 = args.a2_min + (args.a2_max - args.a2_min) * np.random.rand()
            b2 = args.b2_min + (args.b2_max - args.b2_min) * np.random.rand()
            e1 = args.e_min + (args.e_max - args.e_min) * np.random.rand()
            e2 = args.e_min + (args.e_max - args.e_min) * np.random.rand()
            e3 = args.e_min + (args.e_max - args.e_min) * np.random.rand()
            e4 = args.e_min + (args.e_max - args.e_min) * np.random.rand()

            # Define constant
            wavelengths = np.linspace(args.min_wavelength, args.max_wavelength, args.n_wavelengths)
            # Define absorption coefficients for different wavelengths
            ua_skin_baseline = self.data_absorption[0]
            ua_melanin = self.data_absorption[5]
            ua_oxyhaem = self.data_absorption[1]
            ua_deoxyhaem = self.data_absorption[2]
            refractive_index_skin = self.data_absorption[6]
            refractive_index_air = self.data_absorption[7]

            # Calculating the absorption coefficient of whole blood
            ua_bl = ua_oxyhaem * Hb_sat + ua_deoxyhaem * (1 - Hb_sat)

            # Defining factor correction factor for diffuse light C_diff ( Not same as in paper van Veen divded ua_blood out of equation)
            C_diff = (1 - np.exp(-2 * ua_bl * Rv)) / (2 * Rv)

            # Defining total ua
            ua = abs((BvF * args.c_hb * C_diff) + (ua_skin_baseline * SvF) + (ua_melanin * MvF))

            # Scattering coefficient
            us = a2 * ((wavelengths / 500) ** -b2)

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

            # Surface reflectance
            # Fresnel
            fr = ((refractive_index_skin - refractive_index_air) / (refractive_index_skin + refractive_index_air)) ** 2
            normalized_fr = fr / fr[0]

            # Surface reflectance per angle
            Rsurf1 = e1 * normalized_fr
            Rsurf2 = e2 * normalized_fr
            Rsurf3 = e3 * normalized_fr
            Rsurf4 = e4 * normalized_fr

            # Determine noise
            noise_pred1 = self.sample_from_gaussian()
            noise_pred2 = self.sample_from_gaussian()
            noise_pred3 = self.sample_from_gaussian()
            noise_pred4 = self.sample_from_gaussian()

            # Make the predictions
            pred1 = Rdif + Rsurf1 + noise_pred1
            pred2 = Rdif + Rsurf2 + noise_pred2
            pred3 = Rdif + Rsurf3 + noise_pred3
            pred4 = Rdif + Rsurf4 + noise_pred4
            # Without noise
            pred1_no_noise = Rdif + Rsurf1
            pred2_no_noise = Rdif + Rsurf2
            pred3_no_noise = Rdif + Rsurf3
            pred4_no_noise = Rdif + Rsurf4

            print(f"Spectra {i} is successfully created.")

            # Illustrate plot a i =100
            if i == 100 and make_plot == True:
                # Plot each prediction on the same graph
                plt.figure(figsize=(10, 6))

                plt.plot(wavelengths, pred1, label=r'$\phi_1$', linestyle='-', color='b')
                plt.plot(wavelengths, pred2, label=r'$\phi_2$', linestyle='-', color='g')
                plt.plot(wavelengths, pred3, label=r'$\phi_3$', linestyle='-', color='r')
                plt.plot(wavelengths, pred4, label=r'$\phi_4$', linestyle='-', color='m')

                # Add titles and labels
                plt.ylim(0.1, 0.8)
                plt.xlabel(r"$\lambda$(nm)")  # Customize based on actual x-axis meaning
                plt.ylabel('Reflectance (%)')  # Customize based on actual data meaning
                # plt.title('Predicted Reflectance for Different Angles')
                plt.legend()
                plt.grid(True)

                # Show plot
                plt.show()

            # Saving spectra
            spectra_matrix[i, :, 0] = pred1
            spectra_matrix[i, :, 1] = pred2
            spectra_matrix[i, :, 2] = pred3
            spectra_matrix[i, :, 3] = pred4
            # Saving spectra no noise
            spectra_matrix_no_noise[i, :, 0] = pred1_no_noise
            spectra_matrix_no_noise[i, :, 1] = pred2_no_noise
            spectra_matrix_no_noise[i, :, 2] = pred3_no_noise
            spectra_matrix_no_noise[i, :, 3] = pred4_no_noise
            # Saving parameters
            parameters_matrix[i, 0] = BvF
            parameters_matrix[i, 1] = Hb_sat
            parameters_matrix[i, 2] = Rv
            parameters_matrix[i, 3] = SvF
            parameters_matrix[i, 4] = MvF
            parameters_matrix[i, 5] = a2
            parameters_matrix[i, 6] = b2
            parameters_matrix[i, 7] = e1
            parameters_matrix[i, 8] = e2
            parameters_matrix[i, 9] = e3
            parameters_matrix[i, 10] = e4

        return spectra_matrix, parameters_matrix, spectra_matrix_no_noise

    def spectrum_TTCM(self, make_plot=False):
        # Initialize spectra and parameters matrix
        spectra_matrix = np.zeros((self.samples, args.n_wavelengths, args.n_angles))
        spectra_matrix_no_noise = np.zeros((self.samples, args.n_wavelengths, args.n_angles))
        parameters_matrix = np.zeros((self.samples, 15))

        for i in range(self.samples):
            # Second layer (dermis)
            BvF = args.BvF_min + (args.BvF_max - args.BvF_min) * np.random.rand()
            Hb_sat = args.S_min + (args.S_max - args.S_min) * np.random.rand()
            Rv = args.Rv_min + (args.Rv_max - args.Rv_min) * np.random.rand()
            SvF2 = 1 - BvF
            a2 = args.a2_min + (args.a2_max - args.a2_min) * np.random.rand()
            b2 = args.b2_min + (args.b2_max - args.b2_min) * np.random.rand()
            # First layer (epidermis)
            MvF = args.MvF_min + (args.MvF_max - args.MvF_min) * np.random.rand()
            SvF1 = 1 - MvF
            d1 = args.d1
            a1 = args.a1_min + (args.a1_max - args.a1_min) * np.random.rand()
            b1 = args.b1_min + (args.b1_max - args.b1_min) * np.random.rand()
            e1 = args.e_min + (args.e_max - args.e_min) * np.random.rand()
            e2 = args.e_min + (args.e_max - args.e_min) * np.random.rand()
            e3 = args.e_min + (args.e_max - args.e_min) * np.random.rand()
            e4 = args.e_min + (args.e_max - args.e_min) * np.random.rand()

            # Define constant
            wavelengths = np.linspace(args.min_wavelength, args.max_wavelength, args.n_wavelengths)
            # Define absorption coefficients for different wavelengths
            ua_skin_baseline = self.data_absorption[0]
            ua_melanin = self.data_absorption[5]
            ua_oxyhaem = self.data_absorption[1]
            ua_deoxyhaem = self.data_absorption[2]
            refractive_index_skin = self.data_absorption[6]
            refractive_index_air = self.data_absorption[7]

            # Define the absorption coefficient of whole blood
            ua_whole_blood = ua_oxyhaem * Hb_sat + ua_deoxyhaem * (1 - Hb_sat)
            # Define the epidermal layer
            # Absorption-coefficient epidermal layer
            ua1 = (SvF1 * ua_skin_baseline) + (MvF * ua_melanin)
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
            # Determining vessel factor
            C_diff = (1 - np.exp(-2 * ua_whole_blood * Rv)) / (2 * Rv)
            # Absorption-coefficient epidermal layer
            ua2 = (SvF2 * ua_skin_baseline) + (BvF * args.c_hb * C_diff)
            # Scattering-coefficient epidermal layer
            us2 = a2 * ((wavelengths / 500) ** -b2)
            # transport coefficient epidermal layer
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

            # Determine the noise
            noise_pred1 = self.sample_from_gaussian()
            noise_pred2 = self.sample_from_gaussian()
            noise_pred3 = self.sample_from_gaussian()
            noise_pred4 = self.sample_from_gaussian()

            # Make the predictions
            pred1 = Rdif + Rsurf1 + noise_pred1
            pred2 = Rdif + Rsurf2 + noise_pred2
            pred3 = Rdif + Rsurf3 + noise_pred3
            pred4 = Rdif + Rsurf4 + noise_pred4

            # Without noise
            pred1_no_noise = Rdif + Rsurf1
            pred2_no_noise = Rdif + Rsurf2
            pred3_no_noise = Rdif + Rsurf3
            pred4_no_noise = Rdif + Rsurf4

            # Add noise to the spectra

            print(f"Spectra {i} is successfully created.")

            # Illustrate plot a i =100
            if i == 100 and make_plot == True:
                # Plot each prediction on the same graph
                plt.figure(figsize=(10, 6))
                print(pred1)
                plt.plot(wavelengths, pred1, label=r'$\phi_1$', linestyle='-', color='b')
                plt.plot(wavelengths, pred2, label=r'$\phi_2$', linestyle='-', color='g')
                plt.plot(wavelengths, pred3, label=r'$\phi_3$', linestyle='-', color='r')
                plt.plot(wavelengths, pred4, label=r'$\phi_4$', linestyle='-', color='m')

                # Add titles and labels
                plt.ylim(0, 1.0)
                plt.xlabel(r"$\lambda$(nm)")  # Customize based on actual x-axis meaning
                plt.ylabel('Reflectance (%)')  # Customize based on actual data meaning
                # plt.title('Predicted Reflectance for Different Angles')
                plt.legend()
                plt.grid(True)

                # Show plot
                plt.show()

            # Saving spectra
            spectra_matrix[i, :, 0] = pred1
            spectra_matrix[i, :, 1] = pred2
            spectra_matrix[i, :, 2] = pred3
            spectra_matrix[i, :, 3] = pred4
            # Saving spectra no noise
            spectra_matrix_no_noise[i, :, 0] = pred1_no_noise
            spectra_matrix_no_noise[i, :, 1] = pred2_no_noise
            spectra_matrix_no_noise[i, :, 2] = pred3_no_noise
            spectra_matrix_no_noise[i, :, 3] = pred4_no_noise
            # Saving parameters
            parameters_matrix[i, 0] = MvF
            parameters_matrix[i, 1] = SvF1
            parameters_matrix[i, 2] = d1
            parameters_matrix[i, 3] = a1
            parameters_matrix[i, 4] = b1
            parameters_matrix[i, 5] = BvF
            parameters_matrix[i, 6] = Hb_sat
            parameters_matrix[i, 7] = Rv
            parameters_matrix[i, 8] = SvF2
            parameters_matrix[i, 9] = a2
            parameters_matrix[i, 10] = b2
            parameters_matrix[i, 11] = e1
            parameters_matrix[i, 12] = e2
            parameters_matrix[i, 13] = e3
            parameters_matrix[i, 14] = e4

        return spectra_matrix, parameters_matrix, spectra_matrix_no_noise

    def data_loader_synthetic(self, batch_size=args.batch_size, data_synthesizer=args.synthesize_data):

        features_no_noise = None

        # ========================
        # PMLP_OTCM - synthesize
        # ========================
        if args.model == "PMLP_OTCM" and data_synthesizer:

            spectra, labels, spectra_no_noise = self.spectrum_OTCM()

            features = spectra.transpose(0, 2, 1)
            features = features.reshape(spectra.shape[0], -1)

            features_no_noise = spectra_no_noise.transpose(0, 2, 1)
            features_no_noise = features_no_noise.reshape(spectra_no_noise.shape[0], -1)

            x = torch.tensor(features, dtype=torch.float32)
            x_no_noise = torch.tensor(features_no_noise, dtype=torch.float32)
            t = torch.tensor(labels, dtype=torch.float32)

        # ========================
        # PMLP_OTCM - load existing
        # ========================
        elif args.model == "PMLP_OTCM" and data_synthesizer == False:

            if args.mode == "grid_search" and (
                    self.path_spectra_matrix_grid_search.endswith(".npy")
                    and os.path.exists(self.path_spectra_matrix_grid_search)
                    and self.path_parameters_grid_search.endswith(".npy")
                    and os.path.exists(self.path_parameters_grid_search)
            ):
                path_spectra = self.path_spectra_matrix_grid_search
                path_parameters = self.path_parameters_grid_search

                spectra = np.load(path_spectra)
                labels = np.load(path_parameters)

                features = spectra.transpose(0, 2, 1)
                features = features.reshape(spectra.shape[0], -1)

                x = torch.tensor(features, dtype=torch.float32)
                t = torch.tensor(labels, dtype=torch.float32)

            elif (
                    self.path_spectra_matrix_final_model.endswith(".npy")
                    and os.path.exists(self.path_spectra_matrix_final_model)
                    and self.path_parameters_final_model.endswith(".npy")
                    and os.path.exists(self.path_parameters_final_model)
            ):
                path_spectra = self.path_spectra_matrix_final_model
                path_parameters = self.path_parameters_final_model

                spectra = np.load(path_spectra)
                labels = np.load(path_parameters)

                features = spectra.transpose(0, 2, 1)
                features = features.reshape(spectra.shape[0], -1)

                x = torch.tensor(features, dtype=torch.float32)
                t = torch.tensor(labels, dtype=torch.float32)

            else:
                spectra, labels, spectra_no_noise = self.spectrum_OTCM()

                features = spectra.transpose(0, 2, 1)
                features = features.reshape(spectra.shape[0], -1)

                features_no_noise = spectra_no_noise.transpose(0, 2, 1)
                features_no_noise = features_no_noise.reshape(spectra_no_noise.shape[0], -1)

                x = torch.tensor(features, dtype=torch.float32)
                x_no_noise = torch.tensor(features_no_noise, dtype=torch.float32)
                t = torch.tensor(labels, dtype=torch.float32)

        # ========================
        # T_OTCM - synthesize
        # ========================
        elif args.model == "T_OTCM" and data_synthesizer:

            spectra, labels, spectra_no_noise = self.spectrum_OTCM()

            features = spectra.transpose(0, 2, 1)
            features_no_noise = spectra_no_noise.transpose(0, 2, 1)

            x = torch.tensor(features, dtype=torch.float32)
            x_no_noise = torch.tensor(features_no_noise, dtype=torch.float32)
            t = torch.tensor(labels, dtype=torch.float32)

        # ========================
        # T_OTCM - load existing
        # ========================
        elif args.model == "T_OTCM" and data_synthesizer == False:

            if args.mode == "grid_search" and (
                    self.path_spectra_matrix_grid_search.endswith(".npy")
                    and os.path.exists(self.path_spectra_matrix_grid_search)
                    and self.path_parameters_grid_search.endswith(".npy")
                    and os.path.exists(self.path_parameters_grid_search)
            ):
                path_spectra = self.path_spectra_matrix_grid_search
                path_parameters = self.path_parameters_grid_search

                spectra = np.load(path_spectra)
                labels = np.load(path_parameters)

                features = spectra.transpose(0, 2, 1)

                x = torch.tensor(features, dtype=torch.float32)
                t = torch.tensor(labels, dtype=torch.float32)

            elif (
                    self.path_spectra_matrix_final_model.endswith(".npy")
                    and os.path.exists(self.path_spectra_matrix_final_model)
                    and self.path_parameters_final_model.endswith(".npy")
                    and os.path.exists(self.path_parameters_final_model)
            ):
                path_spectra = self.path_spectra_matrix_final_model
                path_parameters = self.path_parameters_final_model

                spectra = np.load(path_spectra)
                labels = np.load(path_parameters)

                features = spectra.transpose(0, 2, 1)

                x = torch.tensor(features, dtype=torch.float32)
                t = torch.tensor(labels, dtype=torch.float32)

            else:
                spectra, labels, spectra_no_noise = self.spectrum_OTCM()

                features = spectra.transpose(0, 2, 1)
                features_no_noise = spectra_no_noise.transpose(0, 2, 1)

                x = torch.tensor(features, dtype=torch.float32)
                x_no_noise = torch.tensor(features_no_noise, dtype=torch.float32)
                t = torch.tensor(labels, dtype=torch.float32)

        # ========================
        # PMLP_TTCM - synthesize
        # ========================
        elif args.model == "PMLP_TTCM" and data_synthesizer:

            spectra, labels, spectra_no_noise = self.spectrum_TTCM()

            features = spectra.transpose(0, 2, 1)
            features = features.reshape(spectra.shape[0], -1)

            features_no_noise = spectra_no_noise.transpose(0, 2, 1)
            features_no_noise = features_no_noise.reshape(spectra_no_noise.shape[0], -1)

            x = torch.tensor(features, dtype=torch.float32)
            x_no_noise = torch.tensor(features_no_noise, dtype=torch.float32)
            t = torch.tensor(labels, dtype=torch.float32)

        # ========================
        # PMLP_TTCM - load existing
        # ========================
        elif args.model == "PMLP_TTCM" and data_synthesizer == False:

            if args.mode == "grid_search" and (
                    self.path_spectra_matrix_grid_search.endswith(".npy")
                    and os.path.exists(self.path_spectra_matrix_grid_search)
                    and self.path_parameters_grid_search.endswith(".npy")
                    and os.path.exists(self.path_parameters_grid_search)
            ):
                path_spectra = self.path_spectra_matrix_grid_search
                path_parameters = self.path_parameters_grid_search

                spectra = np.load(path_spectra)
                labels = np.load(path_parameters)

                features = spectra.transpose(0, 2, 1)
                features = features.reshape(spectra.shape[0], -1)

                x = torch.tensor(features, dtype=torch.float32)
                t = torch.tensor(labels, dtype=torch.float32)

            elif (
                    self.path_spectra_matrix_final_model.endswith(".npy")
                    and os.path.exists(self.path_spectra_matrix_final_model)
                    and self.path_parameters_final_model.endswith(".npy")
                    and os.path.exists(self.path_parameters_final_model)
            ):
                path_spectra = self.path_spectra_matrix_final_model
                path_parameters = self.path_parameters_final_model

                spectra = np.load(path_spectra)
                labels = np.load(path_parameters)

                features = spectra.transpose(0, 2, 1)
                features = features.reshape(spectra.shape[0], -1)

                x = torch.tensor(features, dtype=torch.float32)
                t = torch.tensor(labels, dtype=torch.float32)

            else:
                spectra, labels, spectra_no_noise = self.spectrum_TTCM()

                features = spectra.transpose(0, 2, 1)
                features = features.reshape(spectra.shape[0], -1)

                features_no_noise = spectra_no_noise.transpose(0, 2, 1)
                features_no_noise = features_no_noise.reshape(spectra_no_noise.shape[0], -1)

                x = torch.tensor(features, dtype=torch.float32)
                x_no_noise = torch.tensor(features_no_noise, dtype=torch.float32)
                t = torch.tensor(labels, dtype=torch.float32)

        # ========================
        # T_TTCM - synthesize
        # ========================
        elif args.model == "T_TTCM" and data_synthesizer:

            spectra, labels, spectra_no_noise = self.spectrum_TTCM()

            features = spectra.transpose(0, 2, 1)
            features_no_noise = spectra_no_noise.transpose(0, 2, 1)

            x = torch.tensor(features, dtype=torch.float32)
            x_no_noise = torch.tensor(features_no_noise, dtype=torch.float32)
            t = torch.tensor(labels, dtype=torch.float32)

        # ========================
        # T_TTCM - load existing
        # ========================
        elif args.model == "T_TTCM" and data_synthesizer == False:

            if args.mode == "grid_search" and (
                    self.path_spectra_matrix_grid_search.endswith(".npy")
                    and os.path.exists(self.path_spectra_matrix_grid_search)
                    and self.path_parameters_grid_search.endswith(".npy")
                    and os.path.exists(self.path_parameters_grid_search)
            ):
                path_spectra = self.path_spectra_matrix_grid_search
                path_parameters = self.path_parameters_grid_search

                spectra = np.load(path_spectra)
                labels = np.load(path_parameters)

                features = spectra.transpose(0, 2, 1)

                x = torch.tensor(features, dtype=torch.float32)
                t = torch.tensor(labels, dtype=torch.float32)

            elif (
                    self.path_spectra_matrix_final_model.endswith(".npy")
                    and os.path.exists(self.path_spectra_matrix_final_model)
                    and self.path_parameters_final_model.endswith(".npy")
                    and os.path.exists(self.path_parameters_final_model)
            ):
                path_spectra = self.path_spectra_matrix_final_model
                path_parameters = self.path_parameters_final_model

                spectra = np.load(path_spectra)
                labels = np.load(path_parameters)

                features = spectra.transpose(0, 2, 1)

                x = torch.tensor(features, dtype=torch.float32)
                t = torch.tensor(labels, dtype=torch.float32)

            else:
                spectra, labels, spectra_no_noise = self.spectrum_TTCM()

                features = spectra.transpose(0, 2, 1)
                features_no_noise = spectra_no_noise.transpose(0, 2, 1)

                x = torch.tensor(features, dtype=torch.float32)
                x_no_noise = torch.tensor(features_no_noise, dtype=torch.float32)
                t = torch.tensor(labels, dtype=torch.float32)

        else:
            raise ValueError(f"Unknown model selected: {args.model}")

        # ========================
        # Dataset
        # ========================
        if features_no_noise is not None:
            dataset = TensorDataset(x, t, x_no_noise)
        else:
            dataset = TensorDataset(x, t)

        total_size = len(dataset)
        train_size = int(total_size * 0.6)
        val_size = int(total_size * 0.2)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

        print(
            f"Dataset sizes: Train={len(train_dataset)}, "
            f"Validation={len(val_dataset)}, Test={len(test_dataset)}"
        )

        # ========================
        # Extract batches
        # ========================
        if features_no_noise is not None:

            train_features = [batch[0] for batch in train_loader]
            train_labels = [batch[1] for batch in train_loader]
            train_features_no_noise = [batch[2] for batch in train_loader]

            val_features = [batch[0] for batch in val_loader]
            val_labels = [batch[1] for batch in val_loader]
            val_features_no_noise = [batch[2] for batch in val_loader]

            test_features = [batch[0] for batch in test_loader]
            test_labels = [batch[1] for batch in test_loader]
            test_features_no_noise = [batch[2] for batch in test_loader]

        else:

            train_features = [batch[0] for batch in train_loader]
            train_labels = [batch[1] for batch in train_loader]

            val_features = [batch[0] for batch in val_loader]
            val_labels = [batch[1] for batch in val_loader]

            test_features = [batch[0] for batch in test_loader]
            test_labels = [batch[1] for batch in test_loader]

        # ========================
        # Save original noisy features and labels
        # ========================
        if args.mode == "grid_search":

            np.save(self.path_training_spectra, train_features)
            np.save(self.path_training_parameters, train_labels)

            np.save(self.path_validation_spectra, val_features)
            np.save(self.path_validation_parameters, val_labels)

            np.save(self.path_test_spectra, test_features)
            np.save(self.path_test_parameters, test_labels)

        else:

            np.save(self.path_training_spectra_final_model, train_features)
            np.save(self.path_training_parameters_final_model, train_labels)

            np.save(self.path_validation_spectra_final_model, val_features)
            np.save(self.path_validation_parameters_final_model, val_labels)

            np.save(self.path_test_spectra_final_model, test_features)
            np.save(self.path_test_parameters_final_model, test_labels)

        # ========================
        # Save no-noise features with same structure as features
        # ========================
        if features_no_noise is not None:

            def add_no_noise_suffix(path):
                path = Path(path)
                return path.parent / f"{path.name}_no_noise.npy"

            if args.mode == "grid_search":

                train_no_noise_path = add_no_noise_suffix(
                    self.path_training_spectra
                )

                val_no_noise_path = add_no_noise_suffix(
                    self.path_validation_spectra
                )

                test_no_noise_path = add_no_noise_suffix(
                    self.path_test_spectra
                )

            else:

                train_no_noise_path = add_no_noise_suffix(
                    self.path_training_spectra_final_model
                )

                val_no_noise_path = add_no_noise_suffix(
                    self.path_validation_spectra_final_model
                )

                test_no_noise_path = add_no_noise_suffix(
                    self.path_test_spectra_final_model
                )

            np.save(train_no_noise_path, train_features_no_noise)
            np.save(val_no_noise_path, val_features_no_noise)
            np.save(test_no_noise_path, test_features_no_noise)

            print(f"Saved no-noise training spectra to: {train_no_noise_path}")
            print(f"Saved no-noise validation spectra to: {val_no_noise_path}")
            print(f"Saved no-noise test spectra to: {test_no_noise_path}")