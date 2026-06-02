import sys
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

class GridSearch:

    def __init__(self):
        self.wavelengths = np.linspace(args.min_wavelength, args.max_wavelength, args.n_wavelengths)
        # Making empty lists of mre of parameters
        self.MvF_mre = []
        self.SvF1_mre = []
        self.d1_mre = []
        self.a1_mre = []
        self.b1_mre = []
        self.BvF_mre = []
        self.S_mre = []
        self.Rv_mre = []
        self.SvF2_mre = []
        self.a2_mre = []
        self.b2_mre = []
        self.e1_mre = []
        self.e2_mre = []
        self.e3_mre = []
        self.e4_mre = []

        # MRE lists OTCM
        self.mre_lists_OTCM = [
            self.BvF_mre,
            self.S_mre,
            self.Rv_mre,
            self.SvF1_mre,
            self.MvF_mre,
            self.a2_mre,
            self.b2_mre,
            self.e1_mre,
            self.e2_mre,
            self.e3_mre,
            self.e4_mre
        ]

        # MRE lists TTCM
        self.mre_lists_TTCM = [
            self.MvF_mre,
            self.SvF1_mre,
            self.d1_mre,
            self.a1_mre,
            self.b1_mre,
            self.BvF_mre,
            self.S_mre,
            self.Rv_mre,
            self.SvF2_mre,
            self.a2_mre,
            self.b2_mre,
            self.e1_mre,
            self.e2_mre,
            self.e3_mre,
            self.e4_mre
        ]

        ROOT = Path(__file__).resolve().parent

        self.base_directory = (
                ROOT /
                "PINN" /
                args.model /
                "Results" /
                "Gridsearch"
        )

        self.base_directory.mkdir(parents=True, exist_ok=True)

    # Function to calculate MRE (result)
    def mre_parameter(self, y_true, y_pred):
        epsilon = 1e-8  # to avoid division by zero
        return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon)*100)

    # MLP
    def model_mlp(self, c_batch_size = 100, c_lr_rate = 0.001):

        # List of models to iterate over
        hidden_size = [64, 128]
        layers = [2, 3]
        dropout = [True, False]
        best_validation_loss_list = []

        # Making directory
        today_str = datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(self.base_directory, today_str)
        os.makedirs(save_dir, exist_ok=True)
        # Make empty lists for appending
        hidden_layer_states = []
        n_layer_states = []
        dropout_states = []
        counter_models = 0
        save_path_grid_search = os.path.join(save_dir, "model", "results_gs", f"model_gs.xlsx")
        os.makedirs(os.path.dirname(save_path_grid_search), exist_ok=True)

        # Initialize model / load gridsearch datasets
        Dataloader().data_loader_synthetic(batch_size=c_batch_size)
        ROOT = Path(__file__).resolve().parent
        base = ROOT / "PINN" / args.model

        training_path = (
                base /
                "Dataset gridsearch" /
                "Training" /
                "Spectra" /
                "spectra.npy"
        )

        validation_path = (
                base /
                "Dataset gridsearch" /
                "Validation" /
                "Spectra" /
                "spectra.npy"
        )

        test_spectra_path = (
                base /
                "Dataset gridsearch" /
                "Testing" /
                "Spectra" /
                "spectra.npy"
        )

        test_parameter_path = (
                base /
                "Dataset gridsearch" /
                "Testing" /
                "Parameters" /
                "parameters.npy"
        )

        train_features_batches = np.load(training_path, allow_pickle=True)
        validation_features_batches = np.load(validation_path, allow_pickle=True)
        test_features_batches = np.load(test_spectra_path, allow_pickle=True).squeeze()
        test_labels_batches = np.load(test_parameter_path, allow_pickle=True).squeeze()

        # Making the datasets
        amount_train_batches = train_features_batches.shape[0]
        amount_validation_batches = validation_features_batches.shape[0]
        # Making tensors
        train_features_batches = torch.tensor(train_features_batches)
        validation_features_batches = torch.tensor(validation_features_batches)
        test_features_batches = torch.tensor(test_features_batches)
        # Save results to lists
        for i in hidden_size:
            for j in layers:
                for k in dropout:
                    print("Creating model with HS:", i, "NL:", j, "Dropout:", k)
                    print(" ")
                    print(" ")
                    print(" ")
                    counter_epochs = 0
                    counter_models += 1
                    if args.model == "PMLP_OTCM":
                        model = PMLP_OTCM(hidden_size=i, num_hidden_layers=j, use_dropout=k)
                    elif args.model == "PMLP_TTCM":
                        model = PMLP_TTCM(hidden_size=i, num_hidden_layers=j, use_dropout=k)
                    else:
                        print("No PMLP_OTCM or PMLP_TTCM is selected")
                        sys.exit()

                    # Append in the lists
                    hidden_layer_states.append(i)
                    n_layer_states.append(j)
                    dropout_states.append(k)

                    # Defining the model path
                    model_path = os.path.join(save_dir, "model", "checkpoints", f"gridsearch_model_{counter_models}.pkl")
                    save_path_loss_curve = os.path.join(save_dir, "model", "loss curve", f"state_{counter_models}.png")

                    # Make sure nested folders exist
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    os.makedirs(os.path.dirname(save_path_loss_curve), exist_ok=True)

                    best_val_loss = float('inf')
                    optimizer = Adam(lr=c_lr_rate, params=model.parameters())
                    training_loss_list = []
                    validation_loss_list = []
                    counter_total_epochs = 0

                    # Example of how to use full dataset for training
                    for epoch in range(args.epochs):
                        counter_total_epochs += 1
                        running_loss = 0.0
                        # Iterate over batches in the training DataLoader
                        for batch in range(amount_train_batches):
                            optimizer.zero_grad()
                            batch_data = train_features_batches[batch, :, :]
                            outputs = torch.squeeze(model(batch_data))  # Full dataset passed to model
                            loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = model.loss_dt(batch_data,outputs, lamda1=1, lamda2=1, lamda4=1)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss
                        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / amount_train_batches}")
                        # Saving the losses in list to plot
                        running_loss_training = running_loss / amount_train_batches
                        training_loss_list.append(running_loss_training.item())
                        model.eval()
                        batch_loss_validation = 0.0
                        with torch.no_grad():
                            # Iterate over batches in the training DataLoader
                            for batch in range(amount_validation_batches):
                                batch_data = validation_features_batches[batch, :, :]
                                outputs_val = torch.squeeze(model(batch_data))
                                val_loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = model.loss_dt(batch_data, outputs_val, lamda1=1, lamda2=1, lamda4=1)
                                batch_loss_validation += val_loss
                        validation_loss = batch_loss_validation / amount_validation_batches
                        print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {validation_loss}")
                        # Filling list validation for the plotting
                        validation_loss_list.append(validation_loss.item())
                        if validation_loss < best_val_loss:
                            best_val_loss = validation_loss
                            torch.save(model.state_dict(), model_path)  # Save the model
                            print(f'Saving the model with validation loss: ', validation_loss.item())
                            counter_epochs = 0
                        else:
                            counter_epochs += 1
                        print(f"Current counter epochs:{counter_epochs}")
                        if counter_epochs == 20:
                            break

                    best_validation_loss_list.append(best_val_loss.item())
                    # Plotting the training and validation loss
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(counter_total_epochs), training_loss_list, label='Training Loss')
                    plt.plot(range(counter_total_epochs), validation_loss_list, label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Loss over Epochs')
                    # Apply log scale to the y-axis
                    plt.yscale('log')
                    plt.legend()
                    plt.grid(True)
                    # Save the plot
                    plt.savefig(save_path_loss_curve)
                    plt.close()
                    print(f"Loss curve saved at {save_path_loss_curve}")

                    # Create figure with 2x2 subplots
                    x_axis = self.wavelengths
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
                    save_path_angles = os.path.join(save_dir, "model", "reconstruction", f"reconstruction_spectra_{counter_models}.png")
                    os.makedirs(os.path.dirname(save_path_angles), exist_ok=True)
                    plt.savefig(save_path_angles)
                    plt.close()
                    print(f"Comparison plot saved at {save_path_angles}")

                    # Getting the results MRE of the gird search model
                    model.eval()
                    with torch.no_grad():
                        outputs_test = torch.squeeze(model(test_features_batches))  # Full dataset passed to model
                        outputs_predicted = np.array(outputs_test)
                        if args.model == "PMLP_OTCM":
                            for l in range(test_labels_batches.shape[1]):
                                parameter_predicted = outputs_predicted[l, :]
                                parameter_true = test_labels_batches[:, l]
                                mre_parameter = self.mre_parameter(parameter_true, parameter_predicted)
                                self.mre_lists_OTCM[l].append(mre_parameter)
                        elif args.model =="PMLP_TTCM":
                            for l in range(test_labels_batches.shape[1]):
                                parameter_predicted = outputs_predicted[l, :]
                                parameter_true = test_labels_batches[:, l]
                                mre_parameter = self.mre_parameter(parameter_true, parameter_predicted)
                                self.mre_lists_TTCM[l].append(mre_parameter)

        # Take the mean of the e fractions
        e_total = np.mean([self.e1_mre, self.e2_mre, self.e3_mre, self.e4_mre], axis= 0)
        # Save the results in the xlsx file
        states = [f"State{m+1}" for m in range(len(best_validation_loss_list))]
        grid_search_model_dataframe = pd.DataFrame({"States": states, "HL":hidden_layer_states, "NL":n_layer_states, "dropout":dropout_states, "Best validation": best_validation_loss_list,
                                                    "MvF": self.MvF_mre, "BvF": self.BvF_mre, "S": self.S_mre, "Rv": self.Rv_mre, "e": e_total})
        grid_search_model_dataframe.to_excel(save_path_grid_search)

    def learning_parameters_mlp(self):

        # List of models to iterate over
        batch_size = [50, 100, 200, 400]
        learning_rate = [0.001, 0.0001]
        best_validation_loss_list = []

        # Making directory
        today_str = datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(self.base_directory, today_str)
        os.makedirs(save_dir, exist_ok=True)
        # Make empty lists for appending
        batch_size_states = []
        learning_rate_states = []
        counter_models = 0
        save_path_grid_search = os.path.join(save_dir, "learning parameters", "results_gs", f"learning_parameters_gs.xlsx")
        os.makedirs(os.path.dirname(save_path_grid_search), exist_ok=True)

        for i in batch_size:
            for j in learning_rate:
                counter_epochs = 0
                counter_models += 1
                # Create model
                if args.model == "PMLP_OTCM":
                    model = PMLP_OTCM(
                        hidden_size=128,
                        num_hidden_layers=3,
                        use_dropout=True
                    )

                elif args.model == "PMLP_TTCM":
                    model = PMLP_TTCM(
                        hidden_size=128,
                        num_hidden_layers=3,
                        use_dropout=False
                    )

                else:
                    raise ValueError(f"Unknown model selected: {args.model}")

                # Generate datasets
                Dataloader().data_loader_synthetic(batch_size=i)

                # Define dataset paths
                ROOT = Path(__file__).resolve().parent
                base = ROOT / "PINN" / args.model

                training_path = (
                        base /
                        "Dataset gridsearch" /
                        "Training" /
                        "Spectra" /
                        "spectra.npy"
                )

                validation_path = (
                        base /
                        "Dataset gridsearch" /
                        "Validation" /
                        "Spectra" /
                        "spectra.npy"
                )

                test_spectra_path = (
                        base /
                        "Dataset gridsearch" /
                        "Testing" /
                        "Spectra" /
                        "spectra.npy"
                )

                test_parameter_path = (
                        base /
                        "Dataset gridsearch" /
                        "Testing" /
                        "Parameters" /
                        "parameters.npy"
                )

                # Load datasets
                train_features_batches = np.load(training_path, allow_pickle=True)
                validation_features_batches = np.load(validation_path, allow_pickle=True)
                test_features_batches = np.load(test_spectra_path, allow_pickle=True).squeeze()
                test_labels_batches = np.load(test_parameter_path, allow_pickle=True).squeeze()

                print("Creating model with BS:", i, "LR:", j)
                print(" ")
                print(" ")
                print(" ")
                # Append in the lists
                batch_size_states.append(i)
                learning_rate_states.append(j)

                # Defining the model path
                model_path = os.path.join(save_dir, "learning parameters", "checkpoints", f"gridsearch_model_{counter_models}.pkl")
                save_path_loss_curve = os.path.join(save_dir, "learning parameters", "loss curve", f"state_{counter_models}.png")

                # Make sure nested folders exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                os.makedirs(os.path.dirname(save_path_loss_curve), exist_ok=True)

                # Making the datasets
                amount_train_batches = train_features_batches.shape[0]
                amount_validation_batches = validation_features_batches.shape[0]
                # Making tensors
                train_features_batches = torch.tensor(train_features_batches)
                validation_features_batches = torch.tensor(validation_features_batches)
                test_features_batches = torch.tensor(test_features_batches)

                best_val_loss = float('inf')
                optimizer = Adam(lr=j, params=model.parameters())
                training_loss_list = []
                validation_loss_list = []
                counter_total_epochs = 0

                # Example of how to use full dataset for training
                for epoch in range(args.epochs):
                    counter_total_epochs += 1
                    running_loss = 0.0
                    # Iterate over batches in the training DataLoader
                    for batch in range(amount_train_batches):
                        optimizer.zero_grad()
                        batch_data = train_features_batches[batch, :, :]
                        outputs = torch.squeeze(model(batch_data))  # Full dataset passed to model
                        loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = model.loss_dt(batch_data,
                                                                                                             outputs,
                                                                                                             lamda1=100,
                                                                                                             lamda2=100,
                                                                                                             lamda4=1)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss
                    print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / amount_train_batches}")
                    # Saving the losses in list to plot
                    running_loss_training = running_loss / amount_train_batches
                    training_loss_list.append(running_loss_training.item())
                    model.eval()
                    batch_loss_validation = 0.0
                    with torch.no_grad():
                        # Iterate over batches in the training DataLoader
                        for batch in range(amount_validation_batches):
                            batch_data = validation_features_batches[batch, :, :]
                            outputs_val = torch.squeeze(model(batch_data))
                            val_loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = model.loss_dt(batch_data, outputs_val, lamda1=100, lamda2=100, lamda4=1)
                            batch_loss_validation += val_loss
                    validation_loss = batch_loss_validation / amount_validation_batches
                    print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {validation_loss}")
                    # Filling list validation for the plotting
                    validation_loss_list.append(validation_loss.item())
                    if validation_loss < best_val_loss:
                        best_val_loss = validation_loss
                        torch.save(model.state_dict(), model_path)  # Save the model
                        print(f'Saving the model with validation loss: ', validation_loss.item())
                        counter_epochs = 0
                    else:
                        counter_epochs += 1
                    print(f"Current counter epochs:{counter_epochs}")
                    if counter_epochs == 20:
                        break

                best_validation_loss_list.append(best_val_loss.item())
                # Plotting the training and validation loss
                plt.figure(figsize=(10, 6))
                plt.plot(range(counter_total_epochs), training_loss_list, label='Training Loss')
                plt.plot(range(counter_total_epochs), validation_loss_list, label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss over Epochs')
                # Apply log scale to the y-axis
                plt.yscale('log')
                plt.legend()
                plt.grid(True)
                # Save the plot
                plt.savefig(save_path_loss_curve)
                plt.close()
                print(f"Loss curve saved at {save_path_loss_curve}")

                # Create figure with 2x2 subplots
                x_axis = self.wavelengths
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
                save_path_angles = os.path.join(save_dir, "learning parameters", "reconstruction", f"reconstruction_spectra_{counter_models}.png")
                os.makedirs(os.path.dirname(save_path_angles), exist_ok=True)
                plt.savefig(save_path_angles)
                plt.close()
                print(f"Comparison plot saved at {save_path_angles}")

                # Getting the results MRE of the gird search model
                model.eval()
                with torch.no_grad():
                    outputs_test = torch.squeeze(model(test_features_batches))  # Full dataset passed to model
                    outputs_predicted = np.array(outputs_test)
                    if args.model == "PMLP_OTCM":
                        for l in range(test_labels_batches.shape[1]):
                            parameter_predicted = outputs_predicted[l, :]
                            parameter_true = test_labels_batches[:, l]
                            mre_parameter = self.mre_parameter(parameter_true, parameter_predicted)
                            self.mre_lists_OTCM[l].append(mre_parameter)
                    elif args.model == "PMLP_TTCM":
                        for l in range(test_labels_batches.shape[1]):
                            parameter_predicted = outputs_predicted[l, :]
                            parameter_true = test_labels_batches[:, l]
                            mre_parameter = self.mre_parameter(parameter_true, parameter_predicted)
                            self.mre_lists_TTCM[l].append(mre_parameter)

        # Take the mean of the e fractions
        e_total = np.mean([self.e1_mre, self.e2_mre, self.e3_mre, self.e4_mre], axis= 0)
        # Save the results in the xlsx file
        states = [f"State{m+1}" for m in range(len(best_validation_loss_list))]
        grid_search_model_dataframe = pd.DataFrame({"States": states, "BS":batch_size_states, "LR":learning_rate_states, "Best validation": best_validation_loss_list,
                                                    "MvF": self.MvF_mre, "BvF": self.BvF_mre, "S": self.S_mre, "Rv": self.Rv_mre, "e": e_total})
        grid_search_model_dataframe.to_excel(save_path_grid_search)

    def regularization_mlp(self, c_batch_size = 100, c_lr_rate = 0.001):

        # List of models to iterate over
        w1 = [1, 100]
        w2 = [1, 100]
        w3 = [1, 100]
        best_validation_loss_list = []

        # Making directory
        today_str = datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(self.base_directory, today_str)
        os.makedirs(save_dir, exist_ok=True)
        # Make empty lists for appending
        w1_states = []
        w2_states = []
        w3_states = []
        counter_models = 0
        save_path_grid_search = os.path.join(save_dir, "regularization", "results_gs", f"model_gs.xlsx")
        os.makedirs(os.path.dirname(save_path_grid_search), exist_ok=True)

        # Initialize model
        if args.model == "PMLP_OTCM":
            model = PMLP_OTCM(
                hidden_size=128,
                num_hidden_layers=3,
                use_dropout=True
            )

        elif args.model == "PMLP_TTCM":
            model = PMLP_TTCM(
                hidden_size=128,
                num_hidden_layers=3,
                use_dropout=False
            )

        else:
            raise ValueError(f"Unknown model selected: {args.model}")

        # Generate datasets
        Dataloader().data_loader_synthetic(batch_size=c_batch_size)

        # Define dataset paths
        ROOT = Path(__file__).resolve().parent
        base = ROOT / "PINN" / args.model

        training_path = (
                base /
                "Dataset gridsearch" /
                "Training" /
                "Spectra" /
                "spectra.npy"
        )

        validation_path = (
                base /
                "Dataset gridsearch" /
                "Validation" /
                "Spectra" /
                "spectra.npy"
        )

        test_spectra_path = (
                base /
                "Dataset gridsearch" /
                "Testing" /
                "Spectra" /
                "spectra.npy"
        )

        test_parameter_path = (
                base /
                "Dataset gridsearch" /
                "Testing" /
                "Parameters" /
                "parameters.npy"
        )

        # Load datasets
        train_features_batches = np.load(
            training_path,
            allow_pickle=True
        )

        validation_features_batches = np.load(
            validation_path,
            allow_pickle=True
        )

        test_features_batches = np.load(
            test_spectra_path,
            allow_pickle=True
        ).squeeze()

        test_labels_batches = np.load(
            test_parameter_path,
            allow_pickle=True
        ).squeeze()
        # Making the datasets
        amount_train_batches = train_features_batches.shape[0]
        amount_validation_batches = validation_features_batches.shape[0]
        # Making tensors
        train_features_batches = torch.tensor(train_features_batches)
        validation_features_batches = torch.tensor(validation_features_batches)
        test_features_batches = torch.tensor(test_features_batches)

        for i in w1:
            for j in w2:
                for k in w3:
                    print("Creating model with w1:", i, "w2:", j, "w3:", k)
                    print(" ")
                    print(" ")
                    print(" ")
                    counter_epochs = 0
                    counter_models += 1

                    # Append in the lists
                    w1_states.append(i)
                    w2_states.append(j)
                    w3_states.append(k)

                    # Defining the model path
                    model_path = os.path.join(save_dir, "regularization", "checkpoints", f"gridsearch_model_{counter_models}.pkl")
                    save_path_loss_curve = os.path.join(save_dir, "regularization", "loss curve", f"state_{counter_models}.png")

                    # Make sure nested folders exist
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    os.makedirs(os.path.dirname(save_path_loss_curve), exist_ok=True)

                    best_val_loss = float('inf')
                    optimizer = Adam(lr=c_lr_rate, params=model.parameters())
                    training_loss_list = []
                    validation_loss_list = []
                    counter_total_epochs = 0

                    # Example of how to use full dataset for training
                    for epoch in range(args.epochs):
                        counter_total_epochs += 1
                        running_loss = 0.0
                        # Iterate over batches in the training DataLoader
                        for batch in range(amount_train_batches):
                            optimizer.zero_grad()
                            batch_data = train_features_batches[batch, :, :]
                            outputs = torch.squeeze(model(batch_data))  # Full dataset passed to model
                            loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = model.loss_dt(batch_data,outputs, lamda1=i, lamda2=j, lamda4=k)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss
                        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / amount_train_batches}")
                        # Saving the losses in list to plot
                        running_loss_training = running_loss / amount_train_batches
                        training_loss_list.append(running_loss_training.item())
                        model.eval()
                        batch_loss_validation = 0.0
                        with torch.no_grad():
                            # Iterate over batches in the training DataLoader
                            for batch in range(amount_validation_batches):
                                batch_data = validation_features_batches[batch, :, :]
                                outputs_val = torch.squeeze(model(batch_data))
                                val_loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = model.loss_dt(batch_data, outputs_val, lamda1=i, lamda2=j, lamda4=k)
                                batch_loss_validation += val_loss
                        validation_loss = batch_loss_validation / amount_validation_batches
                        print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {validation_loss}")
                        # Filling list validation for the plotting
                        validation_loss_list.append(validation_loss.item())
                        if validation_loss < best_val_loss:
                            best_val_loss = validation_loss
                            torch.save(model.state_dict(), model_path)  # Save the model
                            print(f'Saving the model with validation loss: ', validation_loss.item())
                            counter_epochs = 0
                        else:
                            counter_epochs += 1
                        print(f"Current counter epochs:{counter_epochs}")
                        if counter_epochs == 20:
                            break

                    best_validation_loss_list.append(best_val_loss.item())
                    # Plotting the training and validation loss
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(counter_total_epochs), training_loss_list, label='Training Loss')
                    plt.plot(range(counter_total_epochs), validation_loss_list, label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Loss over Epochs')
                    # Apply log scale to the y-axis
                    plt.yscale('log')
                    plt.legend()
                    plt.grid(True)
                    # Save the plot
                    plt.savefig(save_path_loss_curve)
                    plt.close()
                    print(f"Loss curve saved at {save_path_loss_curve}")
                    # Create figure with 2x2 subplots
                    x_axis = self.wavelengths
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
                    save_path_angles = os.path.join(save_dir, "regularization", "reconstruction", f"reconstruction_spectra_{counter_models}.png")
                    os.makedirs(os.path.dirname(save_path_angles), exist_ok=True)
                    plt.savefig(save_path_angles)
                    plt.close()
                    print(f"Comparison plot saved at {save_path_angles}")
                    # Getting the results MRE of the gird search model
                    model.eval()
                    with torch.no_grad():
                        outputs_test = torch.squeeze(model(test_features_batches))  # Full dataset passed to model
                        outputs_predicted = np.array(outputs_test)
                        if args.model == "PMLP_OTCM":
                            for l in range(test_labels_batches.shape[1]):
                                parameter_predicted = outputs_predicted[l, :]
                                parameter_true = test_labels_batches[:, l]
                                mre_parameter = self.mre_parameter(parameter_true, parameter_predicted)
                                self.mre_lists_OTCM[l].append(mre_parameter)
                        elif args.model == "PMLP_TTCM":
                            for l in range(test_labels_batches.shape[1]):
                                parameter_predicted = outputs_predicted[l, :]
                                parameter_true = test_labels_batches[:, l]
                                mre_parameter = self.mre_parameter(parameter_true, parameter_predicted)
                                self.mre_lists_TTCM[l].append(mre_parameter)

        # Take the mean of the e fractions
        e_total = np.mean([self.e1_mre, self.e2_mre, self.e3_mre, self.e4_mre], axis= 0)
        # Save the results in the xlsx file
        states = [f"State{m+1}" for m in range(len(best_validation_loss_list))]
        grid_search_model_dataframe = pd.DataFrame({"States": states, "w1":w1_states, "w2":w2_states, "w3":w3_states, "Best validation": best_validation_loss_list,
                                                    "MvF": self.MvF_mre, "BvF": self.BvF_mre, "S": self.S_mre, "Rv": self.Rv_mre, "e": e_total})
        grid_search_model_dataframe.to_excel(save_path_grid_search)

    # Transformer
    def set_transformer(self, c_batch_size=100, c_lr_rate=0.001):
        # List of models to iterate over
        set_d_spec = [32, 64]
        set_n_head = [2, 4]
        set_n_layers = [2, 3]
        best_validation_loss_list = []

        # Making directory
        today_str = datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(self.base_directory, today_str)
        os.makedirs(save_dir, exist_ok=True)
        # Make empty lists for appending
        set_d_spec_states = []
        set_n_head_states = []
        set_n_layers_states = []
        counter_models = 0
        save_path_grid_search = os.path.join(save_dir, "set", "results_gs", f"set_gs.xlsx")
        os.makedirs(os.path.dirname(save_path_grid_search), exist_ok=True)

        # Generate datasets
        Dataloader().data_loader_synthetic(batch_size=c_batch_size)

        # Define dataset paths
        ROOT = Path(__file__).resolve().parent
        base = ROOT / "PINN" / args.model

        training_path = (
                base /
                "Dataset gridsearch" /
                "Training" /
                "Spectra" /
                "spectra.npy"
        )

        validation_path = (
                base /
                "Dataset gridsearch" /
                "Validation" /
                "Spectra" /
                "spectra.npy"
        )

        test_spectra_path = (
                base /
                "Dataset gridsearch" /
                "Testing" /
                "Spectra" /
                "spectra.npy"
        )

        test_parameter_path = (
                base /
                "Dataset gridsearch" /
                "Testing" /
                "Parameters" /
                "parameters.npy"
        )

        # Load datasets
        train_features_batches = np.load(training_path, allow_pickle=True)
        validation_features_batches = np.load(validation_path, allow_pickle=True)

        test_features_batches = np.load(
            test_spectra_path,
            allow_pickle=True
        ).squeeze()

        test_labels_batches = np.load(
            test_parameter_path,
            allow_pickle=True
        ).squeeze()
        # Making the datasets
        amount_train_batches = train_features_batches.shape[0]
        amount_validation_batches = validation_features_batches.shape[0]
        # Making tensors
        train_features_batches = torch.tensor(train_features_batches)
        validation_features_batches = torch.tensor(validation_features_batches)
        test_features_batches = torch.tensor(test_features_batches)

        for i in set_d_spec:
            for j in set_n_head:
                for k in set_n_layers:
                    print("Creating model with d_spec:", i, "n_heads:", j, "n_layers:", k)
                    print(" ")
                    print(" ")
                    print(" ")
                    counter_epochs = 0
                    counter_models += 1
                    if args.model == "T_OTCM":
                        model = T_OTCM(d_spec_set=i, n_heads_set =j, n_layers_set =k)
                    elif args.model == "T_TTCM":
                        model = T_TTCM(d_spec_set=i, n_heads_set =j, n_layers_set =k)
                    else:
                        print("No T_OTCM or T_TTCM is selected")
                        sys.exit()

                    # Append in the lists
                    set_d_spec_states.append(i)
                    set_n_head_states.append(j)
                    set_n_layers_states.append(k)

                    # Defining the model path
                    model_path = os.path.join(save_dir, "set", "checkpoints", f"gridsearch_model_{counter_models}.pkl")
                    save_path_loss_curve = os.path.join(save_dir, "set", "loss curve", f"state_{counter_models}.png")

                    # Make sure nested folders exist
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    os.makedirs(os.path.dirname(save_path_loss_curve), exist_ok=True)

                    best_val_loss = float('inf')
                    optimizer = Adam(lr=c_lr_rate, params=model.parameters())
                    training_loss_list = []
                    validation_loss_list = []
                    counter_total_epochs = 0

                    # Example of how to use full dataset for training
                    for epoch in range(args.epochs):
                        counter_total_epochs += 1
                        running_loss = 0.0
                        # Iterate over batches in the training DataLoader
                        for batch in range(amount_train_batches):
                            optimizer.zero_grad()
                            batch_data = train_features_batches[batch, :, :, :]
                            outputs = torch.squeeze(model(batch_data))  # Full dataset passed to model
                            loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = model.loss_dt(batch_data, outputs)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss
                        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / amount_train_batches}")
                        # Saving the losses in list to plot
                        running_loss_training = running_loss / amount_train_batches
                        training_loss_list.append(running_loss_training.item())
                        model.eval()
                        batch_loss_validation = 0.0
                        with torch.no_grad():
                            # Iterate over batches in the training DataLoader
                            for batch in range(amount_validation_batches):
                                batch_data = validation_features_batches[batch, :, :, :]
                                outputs_val = torch.squeeze(model(batch_data))
                                val_loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = model.loss_dt(batch_data, outputs_val)
                                batch_loss_validation += val_loss
                        validation_loss = batch_loss_validation / amount_validation_batches
                        print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {validation_loss}")
                        # Filling list validation for the plotting
                        validation_loss_list.append(validation_loss.item())
                        if validation_loss < best_val_loss:
                            best_val_loss = validation_loss
                            torch.save(model.state_dict(), model_path)  # Save the model
                            print(f'Saving the model with validation loss: ', validation_loss.item())
                            counter_epochs = 0
                        else:
                            counter_epochs += 1
                        print(f"Current counter epochs:{counter_epochs}")
                        if counter_epochs == 20:
                            break

                    best_validation_loss_list.append(best_val_loss.item())
                    # Plotting the training and validation loss
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(counter_total_epochs), training_loss_list, label='Training Loss')
                    plt.plot(range(counter_total_epochs), validation_loss_list, label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Loss over Epochs')
                    # Apply log scale to the y-axis
                    plt.yscale('log')
                    plt.legend()
                    plt.grid(True)
                    # Save the plot
                    plt.savefig(save_path_loss_curve)
                    plt.close()
                    print(f"Loss curve saved at {save_path_loss_curve}")
                    # Create figure with 2x2 subplots
                    x_axis = self.wavelengths
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
                    save_path_angles = os.path.join(save_dir, "set", "reconstruction", f"reconstruction_spectra_{counter_models}.png")
                    os.makedirs(os.path.dirname(save_path_angles), exist_ok=True)
                    plt.savefig(save_path_angles)
                    plt.close()
                    print(f"Comparison plot saved at {save_path_angles}")

                    # Getting the results MRE of the gird search model
                    model.eval()
                    with torch.no_grad():
                        print("test features", test_features_batches.shape)
                        outputs_test = torch.squeeze(model(test_features_batches))  # Full dataset passed to model
                        outputs_predicted = np.array(outputs_test)
                        if args.model == "T_OTCM":
                            for l in range(test_labels_batches.shape[1]):
                                parameter_predicted = outputs_predicted[:, l]
                                parameter_true = test_labels_batches[:, l]
                                mre_parameter = self.mre_parameter(parameter_true, parameter_predicted)
                                self.mre_lists_OTCM[l].append(mre_parameter)
                        elif args.model =="T_TTCM":
                            for l in range(test_labels_batches.shape[1]):
                                parameter_predicted = outputs_predicted[:, l]
                                parameter_true = test_labels_batches[:, l]
                                mre_parameter = self.mre_parameter(parameter_true, parameter_predicted)
                                self.mre_lists_TTCM[l].append(mre_parameter)

        # Take the mean of the e fractions
        e_total = np.mean([self.e1_mre, self.e2_mre, self.e3_mre, self.e4_mre], axis= 0)
        # Save the results in the xlsx file
        states = [f"State{m+1}" for m in range(len(best_validation_loss_list))]
        grid_search_model_dataframe = pd.DataFrame({"States": states, "d_spec_set": set_d_spec_states, "n_head_set": set_n_head_states, "n_layer_set": set_n_layers_states, "Best validation": best_validation_loss_list,
                                                    "MvF": self.MvF_mre, "BvF": self.BvF_mre, "S": self.S_mre, "Rv": self.Rv_mre, "e": e_total})
        grid_search_model_dataframe.to_excel(save_path_grid_search)

    def aet_transformer(self, c_batch_size=100, c_lr_rate=0.001):

        # List of models to iterate over
        aet_d_spec = [32, 64]
        aet_n_head = [2, 4]
        aet_n_layers = [2, 3]
        best_validation_loss_list = []
        # Making directory
        today_str = datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(self.base_directory, today_str)
        os.makedirs(save_dir, exist_ok=True)
        # Make empty lists for appending
        aet_d_spec_states = []
        aet_n_head_states = []
        aet_n_layers_states = []
        counter_models = 0
        save_path_grid_search = os.path.join(save_dir, "aet", "results_gs", f"aet_gs.xlsx")
        os.makedirs(os.path.dirname(save_path_grid_search), exist_ok=True)

        # Generate datasets
        Dataloader().data_loader_synthetic(batch_size=c_batch_size)

        # Define dataset paths
        ROOT = Path(__file__).resolve().parent
        base = ROOT / "PINN" / args.model

        training_path = (
                base /
                "Dataset gridsearch" /
                "Training" /
                "Spectra" /
                "spectra.npy"
        )

        validation_path = (
                base /
                "Dataset gridsearch" /
                "Validation" /
                "Spectra" /
                "spectra.npy"
        )

        test_spectra_path = (
                base /
                "Dataset gridsearch" /
                "Testing" /
                "Spectra" /
                "spectra.npy"
        )

        test_parameter_path = (
                base /
                "Dataset gridsearch" /
                "Testing" /
                "Parameters" /
                "parameters.npy"
        )

        # Load datasets
        train_features_batches = np.load(training_path, allow_pickle=True)
        validation_features_batches = np.load(validation_path, allow_pickle=True)

        test_features_batches = np.load(
            test_spectra_path,
            allow_pickle=True
        ).squeeze()

        test_labels_batches = np.load(
            test_parameter_path,
            allow_pickle=True
        ).squeeze()

        # Making the datasets
        amount_train_batches = train_features_batches.shape[0]
        amount_validation_batches = validation_features_batches.shape[0]
        # Making tensors
        train_features_batches = torch.tensor(train_features_batches)
        validation_features_batches = torch.tensor(validation_features_batches)
        test_features_batches = torch.tensor(test_features_batches)

        for i in aet_d_spec:
            for j in aet_n_head:
                for k in aet_n_layers:
                    print("Creating model with d_spec:", i, "n_heads:", j, "n_layers:", k)
                    print(" ")
                    print(" ")
                    print(" ")
                    counter_epochs = 0
                    counter_models += 1
                    if args.model == "T_OTCM":
                        model = T_OTCM( d_model=i, n_heads=j, n_layers=k, d_spec_set=64, n_heads_set =2, n_layers_set =2)
                    elif args.model == "T_TTCM":
                        model = T_TTCM( d_model=i, n_heads=j, n_layers=k, d_spec_set=32, n_heads_set =2, n_layers_set =3)
                    else:
                        print("No T_OTCM or T_TTCM is selected")
                        sys.exit()

                    # Append in the lists
                    aet_d_spec_states.append(i)
                    aet_n_head_states.append(j)
                    aet_n_layers_states.append(k)

                    # Defining the model path
                    model_path = os.path.join(save_dir, "aet", "checkpoints", f"gridsearch_model_{counter_models}.pkl")
                    save_path_loss_curve = os.path.join(save_dir, "aet", "loss curve", f"state_{counter_models}.png")

                    # Make sure nested folders exist
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    os.makedirs(os.path.dirname(save_path_loss_curve), exist_ok=True)

                    best_val_loss = float('inf')
                    optimizer = Adam(lr=c_lr_rate, params=model.parameters())
                    training_loss_list = []
                    validation_loss_list = []
                    counter_total_epochs = 0

                    # Example of how to use full dataset for training
                    for epoch in range(args.epochs):
                        counter_total_epochs += 1
                        running_loss = 0.0
                        # Iterate over batches in the training DataLoader
                        for batch in range(amount_train_batches):
                            optimizer.zero_grad()
                            batch_data = train_features_batches[batch, :, :, :]
                            outputs = torch.squeeze(model(batch_data))  # Full dataset passed to model
                            loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = model.loss_dt(batch_data, outputs)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss
                        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / amount_train_batches}")
                        # Saving the losses in list to plot
                        running_loss_training = running_loss / amount_train_batches
                        training_loss_list.append(running_loss_training.item())
                        model.eval()
                        batch_loss_validation = 0.0
                        with torch.no_grad():
                            # Iterate over batches in the training DataLoader
                            for batch in range(amount_validation_batches):
                                batch_data = validation_features_batches[batch, :, :, :]
                                outputs_val = torch.squeeze(model(batch_data))
                                val_loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = model.loss_dt(batch_data, outputs_val)
                                batch_loss_validation += val_loss
                        validation_loss = batch_loss_validation / amount_validation_batches
                        print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {validation_loss}")
                        # Filling list validation for the plotting
                        validation_loss_list.append(validation_loss.item())
                        if validation_loss < best_val_loss:
                            best_val_loss = validation_loss
                            torch.save(model.state_dict(), model_path)  # Save the model
                            print(f'Saving the model with validation loss: ', validation_loss.item())
                            counter_epochs = 0
                        else:
                            counter_epochs += 1
                        print(f"Current counter epochs:{counter_epochs}")
                        if counter_epochs == 20:
                            break

                    best_validation_loss_list.append(best_val_loss.item())
                    # Plotting the training and validation loss
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(counter_total_epochs), training_loss_list, label='Training Loss')
                    plt.plot(range(counter_total_epochs), validation_loss_list, label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Loss over Epochs')
                    # Apply log scale to the y-axis
                    plt.yscale('log')
                    plt.legend()
                    plt.grid(True)
                    # Save the plot
                    plt.savefig(save_path_loss_curve)
                    plt.close()
                    print(f"Loss curve saved at {save_path_loss_curve}")
                    # Create figure with 2x2 subplots
                    x_axis = self.wavelengths
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
                    save_path_angles = os.path.join(save_dir, "aet", "reconstruction", f"reconstruction_spectra_{counter_models}.png")
                    os.makedirs(os.path.dirname(save_path_angles), exist_ok=True)
                    plt.savefig(save_path_angles)
                    plt.close()
                    print(f"Comparison plot saved at {save_path_angles}")
                    # Getting the results MRE of the gird search model
                    model.eval()
                    with torch.no_grad():
                        outputs_test = torch.squeeze(model(test_features_batches))  # Full dataset passed to model
                        outputs_predicted = np.array(outputs_test)
                        if args.model == "T_OTCM":
                            for l in range(test_labels_batches.shape[1]):
                                parameter_predicted = outputs_predicted[:, l]
                                parameter_true = test_labels_batches[:, l]
                                mre_parameter = self.mre_parameter(parameter_true, parameter_predicted)
                                self.mre_lists_OTCM[l].append(mre_parameter)
                        elif args.model =="T_TTCM":
                            for l in range(test_labels_batches.shape[1]):
                                parameter_predicted = outputs_predicted[:, l]
                                parameter_true = test_labels_batches[:, l]
                                mre_parameter = self.mre_parameter(parameter_true, parameter_predicted)
                                self.mre_lists_TTCM[l].append(mre_parameter)

        # Take the mean of the e fractions
        e_total = np.mean([self.e1_mre, self.e2_mre, self.e3_mre, self.e4_mre], axis= 0)
        # Save the results in the xlsx file
        states = [f"State{m+1}" for m in range(len(best_validation_loss_list))]
        grid_search_model_dataframe = pd.DataFrame({"States": states, "d_spec_aet": aet_d_spec_states, "n_head_aet": aet_n_head_states, "n_layer_aet": aet_n_layers_states, "Best validation": best_validation_loss_list,
                                                    "MvF": self.MvF_mre, "BvF": self.BvF_mre, "S": self.S_mre, "Rv": self.Rv_mre, "e": e_total})
        grid_search_model_dataframe.to_excel(save_path_grid_search)

    def learning_parameters_transformer(self):

        # List of models to iterate over
        batch_size = [50, 100, 200, 400]
        learning_rate = [0.001, 0.0001]
        best_validation_loss_list = []
        # Making directory
        today_str = datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(self.base_directory, today_str)
        os.makedirs(save_dir, exist_ok=True)
        # Make empty lists for appending
        batch_size_states = []
        learning_rate_states = []
        counter_models = 0
        save_path_grid_search = os.path.join(save_dir, "learning parameters", "results_gs", f"learning_parameters_gs.xlsx")
        os.makedirs(os.path.dirname(save_path_grid_search), exist_ok=True)
        # Initialize model
        if args.model == "T_OTCM":
            model = T_OTCM(d_model=64, n_heads=4, n_layers=2, d_spec_set=64, n_heads_set =2, n_layers_set =2)
        elif args.model == "T_TTCM":
            model = T_TTCM(d_model=64, n_heads=2, n_layers=2, d_spec_set=32, n_heads_set =2, n_layers_set =3)
        else:
            print("No PMLP_OTCM or PMLP_TTCM is selected")
            sys.exit()

        for i in batch_size:
            for j in learning_rate:
                counter_epochs = 0
                counter_models += 1
                # Generate datasets
                Dataloader().data_loader_synthetic(batch_size=i)

                # Define dataset paths
                ROOT = Path(__file__).resolve().parent
                base = ROOT / "PINN" / args.model

                training_path = (
                        base /
                        "Dataset gridsearch" /
                        "Training" /
                        "Spectra" /
                        "spectra.npy"
                )

                validation_path = (
                        base /
                        "Dataset gridsearch" /
                        "Validation" /
                        "Spectra" /
                        "spectra.npy"
                )

                test_spectra_path = (
                        base /
                        "Dataset gridsearch" /
                        "Testing" /
                        "Spectra" /
                        "spectra.npy"
                )

                test_parameter_path = (
                        base /
                        "Dataset gridsearch" /
                        "Testing" /
                        "Parameters" /
                        "parameters.npy"
                )

                # Load datasets
                train_features_batches = np.load(
                    training_path,
                    allow_pickle=True
                )

                validation_features_batches = np.load(
                    validation_path,
                    allow_pickle=True
                )

                test_features_batches = np.load(
                    test_spectra_path,
                    allow_pickle=True
                ).squeeze()

                test_labels_batches = np.load(
                    test_parameter_path,
                    allow_pickle=True
                ).squeeze()

                print("Creating model with BS:", i, "LR:", j)
                print(" ")
                print(" ")
                print(" ")
                # Append in the lists
                batch_size_states.append(i)
                learning_rate_states.append(j)

                # Defining the model path
                model_path = os.path.join(save_dir, "learning parameters", "checkpoints", f"gridsearch_model_{counter_models}.pkl")
                save_path_loss_curve = os.path.join(save_dir, "learning parameters", "loss curve", f"state_{counter_models}.png")

                # Make sure nested folders exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                os.makedirs(os.path.dirname(save_path_loss_curve), exist_ok=True)

                # Making the datasets
                amount_train_batches = train_features_batches.shape[0]
                amount_validation_batches = validation_features_batches.shape[0]
                # Making tensors
                train_features_batches = torch.tensor(train_features_batches)
                validation_features_batches = torch.tensor(validation_features_batches)
                test_features_batches = torch.tensor(test_features_batches)

                best_val_loss = float('inf')
                optimizer = Adam(lr=j, params=model.parameters())
                training_loss_list = []
                validation_loss_list = []
                counter_total_epochs = 0

                # Example of how to use full dataset for training
                for epoch in range(args.epochs):
                    counter_total_epochs += 1
                    running_loss = 0.0
                    # Iterate over batches in the training DataLoader
                    for batch in range(amount_train_batches):
                        optimizer.zero_grad()
                        batch_data = train_features_batches[batch, :, :, :]
                        outputs = torch.squeeze(model(batch_data))  # Full dataset passed to model
                        loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = model.loss_dt(batch_data, outputs)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss
                    print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / amount_train_batches}")
                    # Saving the losses in list to plot
                    running_loss_training = running_loss / amount_train_batches
                    training_loss_list.append(running_loss_training.item())
                    model.eval()
                    batch_loss_validation = 0.0
                    with torch.no_grad():
                        # Iterate over batches in the training DataLoader
                        for batch in range(amount_validation_batches):
                            batch_data = validation_features_batches[batch,:, :, :]
                            outputs_val = torch.squeeze(model(batch_data))
                            val_loss, Rdata1, Rdata2, Rdata3, Rdata4, Rpred1, Rpred2, Rpred3, Rpred4 = model.loss_dt(batch_data, outputs_val)
                            batch_loss_validation += val_loss
                    validation_loss = batch_loss_validation / amount_validation_batches
                    print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {validation_loss}")
                    # Filling list validation for the plotting
                    validation_loss_list.append(validation_loss.item())
                    if validation_loss < best_val_loss:
                        best_val_loss = validation_loss
                        torch.save(model.state_dict(), model_path)  # Save the model
                        print(f'Saving the model with validation loss: ', validation_loss.item())
                        counter_epochs = 0
                    else:
                        counter_epochs += 1
                    print(f"Current counter epochs:{counter_epochs}")
                    if counter_epochs == 20:
                        break

                best_validation_loss_list.append(best_val_loss.item())
                # Plotting the training and validation loss
                plt.figure(figsize=(10, 6))
                plt.plot(range(counter_total_epochs), training_loss_list, label='Training Loss')
                plt.plot(range(counter_total_epochs), validation_loss_list, label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss over Epochs')
                # Apply log scale to the y-axis
                plt.yscale('log')
                plt.legend()
                plt.grid(True)
                # Save the plot
                plt.savefig(save_path_loss_curve)
                plt.close()
                print(f"Loss curve saved at {save_path_loss_curve}")
                # Create figure with 2x2 subplots
                x_axis = self.wavelengths
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
                save_path_angles = os.path.join(save_dir, "learning parameters", "reconstruction", f"reconstruction_spectra_{counter_models}.png")
                os.makedirs(os.path.dirname(save_path_angles), exist_ok=True)
                plt.savefig(save_path_angles)
                plt.close()
                print(f"Comparison plot saved at {save_path_angles}")
                # Getting the results MRE of the gird search model
                model.eval()
                with torch.no_grad():
                    outputs_test = torch.squeeze(model(test_features_batches))  # Full dataset passed to model
                    outputs_predicted = np.array(outputs_test)
                    if args.model == "T_OTCM":
                        for l in range(test_labels_batches.shape[1]):
                            parameter_predicted = outputs_predicted[:, l]
                            parameter_true = test_labels_batches[:, l]
                            mre_parameter = self.mre_parameter(parameter_true, parameter_predicted)
                            self.mre_lists_OTCM[l].append(mre_parameter)
                    elif args.model == "T_TTCM":
                        for l in range(test_labels_batches.shape[1]):
                            parameter_predicted = outputs_predicted[:, l]
                            parameter_true = test_labels_batches[:, l]
                            mre_parameter = self.mre_parameter(parameter_true, parameter_predicted)
                            self.mre_lists_TTCM[l].append(mre_parameter)

        # Take the mean of the e fractions
        e_total = np.mean([self.e1_mre, self.e2_mre, self.e3_mre, self.e4_mre], axis= 0)
        # Save the results in the xlsx file
        states = [f"State{m+1}" for m in range(len(best_validation_loss_list))]
        grid_search_model_dataframe = pd.DataFrame({"States": states, "BS":batch_size_states, "LR":learning_rate_states, "Best validation": best_validation_loss_list,
                                                    "MvF": self.MvF_mre, "BvF": self.BvF_mre, "S": self.S_mre, "Rv": self.Rv_mre, "e": e_total})
        grid_search_model_dataframe.to_excel(save_path_grid_search)
