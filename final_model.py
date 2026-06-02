import numpy as np
import torch
import os
from datetime import datetime
import pandas as pd
from torch.optim import Adam
from PINN import PMLP_OTCM, PMLP_TTCM, T_OTCM, T_TTCM
import matplotlib.pyplot as plt
import args_parse
from dataloader import Dataloader
from pathlib import Path

args = args_parse.args_parser()

class FinalModel:
    def __init__(self):
        self.wavelengths = np.linspace(args.min_wavelength, args.max_wavelength, args.n_wavelengths)

        # Synthesize data
        Dataloader().data_loader_synthetic()

        ROOT = Path(__file__).resolve().parent
        base = ROOT / "PINN" / args.model

        # Extract data
        training_path = (
                base /
                "Dataset final model" /
                "Training" /
                "Spectra" /
                "spectra.npy"
        )

        validation_path = (
                base /
                "Dataset final model" /
                "Validation" /
                "Spectra" /
                "spectra.npy"
        )

        self.train_features_batches = np.load(training_path, allow_pickle=True)
        self.validation_features_batches = np.load(validation_path, allow_pickle=True)

        # Define model
        model_map = {
            "PMLP_OTCM": PMLP_OTCM,
            "T_OTCM": T_OTCM,
            "PMLP_TTCM": PMLP_TTCM,
            "T_TTCM": T_TTCM,
        }

        if args.model not in model_map:
            raise ValueError(f"Unknown model selected: {args.model}")

        self.model = model_map[args.model]()

        # Base directories to save files
        self.base_directory = base / "Results" / "Final model"
        self.base_directory.mkdir(parents=True, exist_ok=True)

    def train_mlp(self):
        # Device detecteren
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print("Using device:", device)
        self.model = self.model.to(device)

        # Define number of batches
        number_train_batches = self.train_features_batches.shape[0]
        number_validation_batches = self.validation_features_batches.shape[0]

        # Define the torch tensors of the data
        train_features_batches = torch.tensor(self.train_features_batches, device=device)
        validation_features_batches = torch.tensor(self.validation_features_batches, device=device)

        # Optimizer
        optimizer = Adam(lr=args.learning_rate, params=self.model.parameters())

        # Defining empty lists
        training_loss_list = []
        validation_loss_list = []

        # Defining starting value of validation loss
        best_val_loss = float('inf')

        # Counter epochs
        counter_epochs = 0
        x_axis_plot_epochs = 0

        # Making directory
        today_str = datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(self.base_directory, today_str)
        os.makedirs(save_dir, exist_ok=True)

        # Defining the model path
        model_path = os.path.join(save_dir, "checkpoint final model", "model.pkl")

        # Make sure nested folders exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Starting the training loop
        for epoch in range(args.epochs):
            running_loss = 0.0
            running_loss_validation = 0.0
            print(epoch)
            # Iterate over batches in the training DataLoader
            for batch in range(number_train_batches):
                optimizer.zero_grad()
                batch_data = train_features_batches[batch, :, :]
                outputs = torch.squeeze(self.model(batch_data))  # Full dataset passed to model
                loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = self.model.loss_dt(batch_data,
                                                                                                          outputs,
                                                                                                          lamda1=args.w1,
                                                                                                          lamda2=args.w2,
                                                                                                          lamda4=args.w4)
                loss.backward()
                optimizer.step()
                running_loss += loss
            print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / number_train_batches}")
            # Saving the losses in list to plot
            running_loss_training = running_loss / number_train_batches
            training_loss_list.append(running_loss_training.item())

            # Validation
            self.model.eval()
            val_loss = 0.0
            batch_loss_validation = 0.0
            with torch.no_grad():
                # Iterate over batches in the training DataLoader
                for batch in range(number_validation_batches):
                    batch_data = validation_features_batches[batch, :, :]
                    outputs_val = torch.squeeze(self.model(batch_data))
                    val_loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = self.model.loss_dt(
                        batch_data, outputs_val, lamda1=args.w1, lamda2=args.w2, lamda4=args.w4)
                    batch_loss_validation += val_loss
            validation_loss = batch_loss_validation / number_validation_batches
            print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {validation_loss}")
            x_axis_plot_epochs += 1
            # Filling list validation for the plotting
            validation_loss_list.append(validation_loss.item())
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                torch.save(self.model.state_dict(), model_path)  # Save the model
                print(f'Saving the model with validation loss: ', validation_loss.item())
                counter_epochs = 0
            else:
                counter_epochs += 1
            print(f"Current counter epochs:{counter_epochs}")
            if counter_epochs == 20:
                break

        # Plotting the training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(x_axis_plot_epochs), training_loss_list, label='Training Loss')
        plt.plot(range(x_axis_plot_epochs), validation_loss_list, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        # Apply log scale to the y-axis
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        # Define save path
        save_path_loss_curve = os.path.join(save_dir, "Plot results", "example_loss_curve.png")
        os.makedirs(os.path.dirname(save_path_loss_curve), exist_ok=True)
        plt.savefig(save_path_loss_curve)
        plt.close()
        print(f"Loss curve saved at {save_path_loss_curve}")

        # Define x-axis values (e.g., wavelengths or indices)
        x_axis = self.wavelengths
        # Plot true reflectance spectra (all angles overlayed)
        plt.figure(figsize=(8, 6), dpi=300)

        markers = ['o', 's', '^', 'x']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        labels = [r"$\phi_1$", r"$\phi_2$", r"$\phi_3$", r"$\phi_4$"]
        Rdata_list = [Rdata1, Rdata2, Rdata3, Rdata4]

        for rdata, m, c, label in zip(Rdata_list, markers, colors, labels):
            plt.plot(x_axis, rdata.detach().cpu().numpy(),
                     marker=m, color=c, linewidth=1.5, markersize=4,
                     label=label)

        plt.xlabel(r"$\lambda$ (nm)", fontsize=14)
        plt.ylabel("Reflectance", fontsize=14)
        # plt.title("True Reflectance Spectra", fontsize=16, fontweight="bold")
        plt.ylim(0, 1)
        plt.xlim(float(x_axis.min()), float(x_axis.max()))
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend(fontsize=12, frameon=False, loc="best")
        plt.tight_layout()
        # Making path to save
        # Define the path where you want to save the plot
        save_path_true = os.path.join(save_dir, "Plot results", "true_spectra.png")
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path_true), exist_ok=True)
        # Save the plot to the specified path
        plt.savefig(save_path_true)
        # Optionally, close the plot if running in a script to free up memory
        plt.close()
        print(f"Plot saved at {save_path_true}")

        # Plot predicted reflectance spectra (all angles overlayed)
        plt.figure(figsize=(8, 6), dpi=300)
        markers = ['o', 's', '^', 'x']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        labels = [r"$\phi_1$", r"$\phi_2$", r"$\phi_3$", r"$\phi_4$"]
        Rpred_list = [Rpred1, Rpred2, Rpred3, Rpred4]

        for rpred, m, c, label in zip(Rpred_list, markers, colors, labels):
            plt.plot(x_axis, rpred.detach().cpu().numpy(),
                     marker=m, color=c, linewidth=1.5, markersize=4,
                     label=label)

        plt.xlabel(r"$\lambda$ (nm)", fontsize=14)
        plt.ylabel("Reflectance", fontsize=14)
        # plt.title("Predicted Reflectance Spectra", fontsize=16, fontweight="bold")
        plt.ylim(0, 1)
        plt.xlim(float(x_axis.min()), float(x_axis.max()))
        plt.grid(False)
        plt.legend(fontsize=12, frameon=False, loc="best")
        plt.tight_layout()
        save_path_pred = os.path.join(save_dir, "Plot results", "predicted_spectra.png")
        os.makedirs(os.path.dirname(save_path_pred), exist_ok=True)
        plt.savefig(save_path_pred)
        plt.close()
        print(f"Predicted data plot saved at {save_path_pred}")

        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300)  # dpi for high resolution
        # fig.suptitle("True vs. Predicted Reflectance Spectra", fontsize=16, fontweight="bold")
        # Common settings
        true_style = dict(marker='o', color='tab:blue', markersize=4, linewidth=1.2)
        pred_style = dict(marker='x', color='tab:red', markersize=4, linewidth=1.2, linestyle='--')

        titles = [r"$\phi_1$", r"$\phi_2$", r"$\phi_3$", r"$\phi_4$"]
        data_list = [(Rdata1, Rpred1), (Rdata2, Rpred2), (Rdata3, Rpred3), (Rdata4, Rpred4)]

        for ax, title, (rdata, rpred) in zip(axs.ravel(), titles, data_list):
            ax.plot(x_axis, rdata.detach().cpu().numpy(), label="True", **true_style)
            ax.plot(x_axis, rpred.detach().cpu().numpy(), label="Predicted", **pred_style)

            ax.set_title(title, fontsize=14)
            ax.set_xlabel(r"$\lambda$ (nm)", fontsize=12)
            ax.set_ylabel("Reflectance", fontsize=12)
            ax.set_ylim(0, 1)  # <-- Fix y-axis from 0 to 1
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.legend(fontsize=10, frameon=False, loc="best")
            ax.grid(True, linestyle=":", alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust so suptitle fits
        # Define save path
        save_path_angles = os.path.join(save_dir, "Plot results", "reconstruction_spectra.png")
        os.makedirs(os.path.dirname(save_path_angles), exist_ok=True)
        plt.savefig(save_path_angles)
        plt.close()
        print(f"Comparison plot saved at {save_path_angles}")
        # Save values in Excel file
        # Assume results_dict is already created like in your training function
        results_dict = {
            "parameters": {
                "BvF_min": args.BvF_min,
                "BvF_max": args.BvF_max,
                "S_min": args.S_min,
                "S_max": args.S_max,
                "Rv_min": args.Rv_min,
                "Rv_max": args.Rv_max,
                "MvF_min": args.MvF_min,
                "MvF_max": args.MvF_max,
                "d1": args.d1,
                "a1_min": args.a1_min,
                "a1_max": args.a1_max,
                "b1_min": args.b1_min,
                "b1_max": args.b1_max,
                "a2_min": args.a2_min,
                "a2_max": args.a2_max,
                "b2_min": args.b2_min,
                "b2_max": args.b2_max,
                "e_min": args.e_min,
                "e_max": args.e_max,
                "c_hb": args.c_hb,
                "r_eff": args.r_eff,
            },

            "model_architecture": {
                "hidden_layer_size": args.hidden_layer_size,
                "n_hidden_layer": args.n_hidden_layer,
                "activation_function": args.activation_function,
                "dropout": args.dropout,
                "dropout_size": args.dropout_size,
                "w1": args.w1,
                "w2": args.w2,
                "w3": args.w3,
                "w4": args.w4,
            },

            "remaining": {
                "n_wavelengths": args.n_wavelengths,
                "min_wavelength": args.min_wavelength,
                "max_wavelength": args.max_wavelength,
                "n_angles": args.n_angles,
                "gridsearch": args.gridsearch,
                "synthesize_data": args.synthesize_data,
                "samples_grid_search": args.samples_grid_search,
                "samples_final_model": args.samples_final_model,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
            },

            "results": {
                "best_validation_loss": best_val_loss.item() if torch.is_tensor(best_val_loss) else best_val_loss,
                "final_training_loss": training_loss_list[-1] if len(training_loss_list) > 0 else None,
                "final_validation_loss": validation_loss_list[-1] if len(validation_loss_list) > 0 else None,
            }
        }

        # Helper function to convert nested dict to list of rows
        def flatten_to_rows(category, d):
            rows = []
            for k, v in d.items():
                rows.append({
                    "category": category,
                    "name": k,
                    "value": v
                })
            return rows

        # Build all rows
        rows = []
        rows.extend(flatten_to_rows("parameters", results_dict["parameters"]))
        rows.extend(flatten_to_rows("model_architecture", results_dict["model_architecture"]))
        rows.extend(flatten_to_rows("remaining", results_dict["remaining"]))
        rows.extend(flatten_to_rows("results", results_dict["results"]))

        # Create DataFrame in long format
        df_results = pd.DataFrame(rows)

        # Save to Excel
        save_path_excel = os.path.join(save_dir, "training_results.xlsx")
        df_results.to_excel(save_path_excel, index=False)

        print(f"Training results saved to {save_path_excel}")

    def train_transformer(self):
        # Device detected
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print("Using device:", device)
        self.model = self.model.to(device)

        # Define number of batches
        number_train_batches = self.train_features_batches.shape[0]
        number_validation_batches = self.validation_features_batches.shape[0]

        # Define the torch tensors of the data
        train_features_batches = torch.tensor(self.train_features_batches, device=device)
        validation_features_batches = torch.tensor(self.validation_features_batches, device=device)

        # Optimizer
        optimizer = Adam(lr=args.learning_rate, params=self.model.parameters())

        # Defining empty lists
        training_loss_list = []
        validation_loss_list = []

        # Defining starting value of validation loss
        best_val_loss = float('inf')

        # Counter epochs
        counter_epochs = 0
        x_axis_plot_epochs = 0

        # Making directory
        today_str = datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(self.base_directory, today_str)
        os.makedirs(save_dir, exist_ok=True)

        # Defining the model path
        model_path = os.path.join(save_dir, "checkpoint final model", "model.pkl")

        # Make sure nested folders exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        for epoch in range(args.epochs):
            running_loss = 0.0
            running_loss_validation = 0.0
            print(epoch)
            # Iterate over batches in the training DataLoader
            for batch in range(number_train_batches):
                optimizer.zero_grad()
                batch_data = train_features_batches[batch, :, :, :]
                outputs = torch.squeeze(self.model(batch_data))  # Full dataset passed to model
                loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = self.model.loss_dt(batch_data,
                                                                                                          outputs)
                loss.backward()
                optimizer.step()
                running_loss += loss
            print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / number_train_batches}")
            # Saving the losses in list to plot
            running_loss_training = running_loss / number_train_batches
            training_loss_list.append(running_loss_training.item())
            # Validation
            self.model.eval()
            val_loss = 0.0
            batch_loss_validation = 0.0
            with torch.no_grad():
                # Iterate over batches in the training DataLoader
                for batch in range(number_validation_batches):
                    batch_data = validation_features_batches[batch, :, :, :]
                    # print(batch_data.shape)
                    outputs_val = torch.squeeze(self.model(batch_data))
                    val_loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = self.model.loss_dt(
                        batch_data, outputs_val)
                    batch_loss_validation += val_loss
            validation_loss = batch_loss_validation / number_validation_batches
            print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {validation_loss}")
            x_axis_plot_epochs += 1
            # Filling list validation for the plotting
            validation_loss_list.append(validation_loss.item())
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                torch.save(self.model.state_dict(), model_path)  # Save the model
                print(f'Saving the model with validation loss: ', validation_loss.item())
                counter_epochs = 0
            else:
                counter_epochs += 1
            print(f"Current counter epochs:{counter_epochs}")
            if counter_epochs == 20:
                break

        # Plotting the training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(x_axis_plot_epochs), training_loss_list, label='Training Loss')
        plt.plot(range(x_axis_plot_epochs), validation_loss_list, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        # Apply log scale to the y-axis
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        # Define save path
        save_path_loss_curve = os.path.join(save_dir, "Plot results", "example_loss_curve.png")
        os.makedirs(os.path.dirname(save_path_loss_curve), exist_ok=True)
        plt.savefig(save_path_loss_curve)
        plt.close()
        print(f"Loss curve saved at {save_path_loss_curve}")

        # Define x-axis values (e.g., wavelengths or indices)
        x_axis = self.wavelengths
        # Plot true reflectance spectra (all angles overlayed)
        plt.figure(figsize=(8, 6), dpi=300)

        markers = ['o', 's', '^', 'x']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        labels = [r"$\phi_1$", r"$\phi_2$", r"$\phi_3$", r"$\phi_4$"]
        Rdata_list = [Rdata1, Rdata2, Rdata3, Rdata4]

        for rdata, m, c, label in zip(Rdata_list, markers, colors, labels):
            plt.plot(x_axis, rdata.detach().cpu().numpy(),
                     marker=m, color=c, linewidth=1.5, markersize=4,
                     label=label)

        plt.xlabel(r"$\lambda$ (nm)", fontsize=14)
        plt.ylabel("Reflectance", fontsize=14)
        # plt.title("True Reflectance Spectra", fontsize=16, fontweight="bold")
        plt.ylim(0, 1)
        plt.xlim(float(x_axis.min()), float(x_axis.max()))
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend(fontsize=12, frameon=False, loc="best")
        plt.tight_layout()
        # Making path to save
        # Define the path where you want to save the plot
        save_path_true = os.path.join(save_dir, "Plot results", "true_spectra.png")
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path_true), exist_ok=True)
        # Save the plot to the specified path
        plt.savefig(save_path_true)
        # Optionally, close the plot if running in a script to free up memory
        plt.close()
        print(f"Plot saved at {save_path_true}")

        # Plot predicted reflectance spectra (all angles overlayed)
        plt.figure(figsize=(8, 6), dpi=300)
        markers = ['o', 's', '^', 'x']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        labels = [r"$\phi_1$", r"$\phi_2$", r"$\phi_3$", r"$\phi_4$"]
        Rpred_list = [Rpred1, Rpred2, Rpred3, Rpred4]

        for rpred, m, c, label in zip(Rpred_list, markers, colors, labels):
            plt.plot(x_axis, rpred.detach().cpu().numpy(),
                     marker=m, color=c, linewidth=1.5, markersize=4,
                     label=label)

        plt.xlabel(r"$\lambda$ (nm)", fontsize=14)
        plt.ylabel("Reflectance", fontsize=14)
        # plt.title("Predicted Reflectance Spectra", fontsize=16, fontweight="bold")
        plt.ylim(0, 1)
        plt.xlim(float(x_axis.min()), float(x_axis.max()))
        plt.grid(False)
        plt.legend(fontsize=12, frameon=False, loc="best")
        plt.tight_layout()
        save_path_pred = os.path.join(save_dir, "Plot results", "predicted_spectra.png")
        os.makedirs(os.path.dirname(save_path_pred), exist_ok=True)
        plt.savefig(save_path_pred)
        plt.close()
        print(f"Predicted data plot saved at {save_path_pred}")

        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300)  # dpi for high resolution
        # fig.suptitle("True vs. Predicted Reflectance Spectra", fontsize=16, fontweight="bold")
        # Common settings
        true_style = dict(marker='o', color='tab:blue', markersize=4, linewidth=1.2)
        pred_style = dict(marker='x', color='tab:red', markersize=4, linewidth=1.2, linestyle='--')

        titles = [r"$\phi_1$", r"$\phi_2$", r"$\phi_3$", r"$\phi_4$"]
        data_list = [(Rdata1, Rpred1), (Rdata2, Rpred2), (Rdata3, Rpred3), (Rdata4, Rpred4)]

        for ax, title, (rdata, rpred) in zip(axs.ravel(), titles, data_list):
            ax.plot(x_axis, rdata.detach().cpu().numpy(), label="True", **true_style)
            ax.plot(x_axis, rpred.detach().cpu().numpy(), label="Predicted", **pred_style)

            ax.set_title(title, fontsize=14)
            ax.set_xlabel(r"$\lambda$ (nm)", fontsize=12)
            ax.set_ylabel("Reflectance", fontsize=12)
            ax.set_ylim(0, 1)  # <-- Fix y-axis from 0 to 1
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.legend(fontsize=10, frameon=False, loc="best")
            ax.grid(True, linestyle=":", alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust so suptitle fits
        # Define save path
        save_path_angles = os.path.join(save_dir, "Plot results", "reconstruction_spectra.png")
        os.makedirs(os.path.dirname(save_path_angles), exist_ok=True)
        plt.savefig(save_path_angles)
        plt.close()
        print(f"Comparison plot saved at {save_path_angles}")

        # Save values in excel file
        # Assume results_dict is already created like in your training function
        results_dict = {
            "parameters": {
                "BvF_min": args.BvF_min,
                "BvF_max": args.BvF_max,
                "S_min": args.S_min,
                "S_max": args.S_max,
                "Rv_min": args.Rv_min,
                "Rv_max": args.Rv_max,
                "MvF_min": args.MvF_min,
                "MvF_max": args.MvF_max,
                "d1": args.d1,
                "a1_min": args.a1_min,
                "a1_max": args.a1_max,
                "b1_min": args.b1_min,
                "b1_max": args.b1_max,
                "a2_min": args.a2_min,
                "a2_max": args.a2_max,
                "b2_min": args.b2_min,
                "b2_max": args.b2_max,
                "e_min": args.e_min,
                "e_max": args.e_max,
                "c_hb": args.c_hb,
                "r_eff": args.r_eff,
            },

            "model_architecture": {
                # Spectral Encoder Transformer (SET)
                "SET_d_spec": args.SET_d_spec,
                "SET_n_head": args.SET_n_head,
                "SET_n_layers": args.SET_n_layers,

                # Angular Encoder Transformer (AET)
                "AET_d_model": args.AET_d_model,
                "AET_n_head": args.AET_n_head,
                "AET_n_layers": args.AET_n_layers,
                "AET_hl_MLP": args.AET_hl_MLP,
            },

            "remaining": {
                "n_wavelengths": args.n_wavelengths,
                "min_wavelength": args.min_wavelength,
                "max_wavelength": args.max_wavelength,
                "n_angles": args.n_angles,
                "gridsearch": args.gridsearch,
                "synthesize_data": args.synthesize_data,
                "samples_grid_search": args.samples_grid_search,
                "samples_final_model": args.samples_final_model,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
            },

            "results": {
                "best_validation_loss": best_val_loss.item() if torch.is_tensor(best_val_loss) else best_val_loss,
                "final_training_loss": training_loss_list[-1] if len(training_loss_list) > 0 else None,
                "final_validation_loss": validation_loss_list[-1] if len(validation_loss_list) > 0 else None,
            }
        }

        # Helper function to convert nested dict to list of rows
        def flatten_to_rows(category, d):
            rows = []
            for k, v in d.items():
                rows.append({
                    "category": category,
                    "name": k,
                    "value": v
                })
            return rows

        # Build all rows
        rows = []
        rows.extend(flatten_to_rows("parameters", results_dict["parameters"]))
        rows.extend(flatten_to_rows("model_architecture", results_dict["model_architecture"]))
        rows.extend(flatten_to_rows("remaining", results_dict["remaining"]))
        rows.extend(flatten_to_rows("results", results_dict["results"]))

        # Create DataFrame in long format
        df_results = pd.DataFrame(rows)

        # Save to Excel
        save_path_excel = os.path.join(save_dir, "training_results.xlsx")
        df_results.to_excel(save_path_excel, index=False)

        print(f"Training results saved to {save_path_excel}")
