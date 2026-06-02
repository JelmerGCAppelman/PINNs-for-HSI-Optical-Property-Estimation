import argparse

def args_parser():

    parser = argparse.ArgumentParser()

    # Main settings
    parser.add_argument(
        "-mode",
        type=str,
        default="final_model",
        choices=[
            "grid_search",
            "final_model",
            "analysis"
        ]
    )

    # Selected model for grid search/ final model
    parser.add_argument(
        "-model",
        type=str,
        default="PMLP_TTCM",
        choices=[
            "PMLP_OTCM",
            "PMLP_TTCM",
            "T_OTCM",
            "T_TTCM"
        ]
    )

    # Grid search slices
    parser.add_argument(
        "-grid_search_type",
        type=str,
        default="model_architecture",
        choices=[
            # PMLP
            "model_architecture",
            "learning_parameters",
            "regularization",

            # Transformer
            "SET_architecture",
            "AET_architecture",
            "learning_parameters"
        ]
    )

    # Final Model Training
    # Parallel multilayer perceptron architecture
    # Model architecture
    parser.add_argument('-hidden_layer_size', type=int, default = 128)
    parser.add_argument('-n_hidden_layer', type=int, default = 3)
    parser.add_argument('-activation_function', type=str, default = 'relu')
    parser.add_argument('-dropout', type=bool, default = True)
    parser.add_argument('-dropout_size', type=float, default = 0.3)
    # Regularization weights
    parser.add_argument('-w1', type=int, default=100)
    parser.add_argument('-w2', type=int, default=100)
    parser.add_argument('-w3', type=int, default=1)
    parser.add_argument('-w4', type=int, default=1)

    # Transformer architecture
    # Spectral Encoder Transformer (SET)
    parser.add_argument('-SET_d_spec', type=int, default=64)
    parser.add_argument('-SET_n_head', type=int, default=2)
    parser.add_argument('-SET_n_layers', type=int, default=2)
    # Angular Encoder Transformer (AET)
    parser.add_argument('-AET_d_model', type=int, default=64)
    parser.add_argument('-AET_n_head', type=int, default=4)
    parser.add_argument('-AET_n_layers', type=int, default=2)
    parser.add_argument('-AET_hl_MLP', type=int, default=256)

    # Analysis: when the mode analysis is selected
    # Checkpoints (trained networks)
    parser.add_argument("-date_PMLP_OTCM", type=str, default="2026-05-16")
    parser.add_argument("-date_PMLP_TTCM", type=str, default="2026-05-16")
    parser.add_argument("-date_T_OTCM", type=str, default="2026-05-16")
    parser.add_argument("-date_T_TTCM", type=str, default="2026-05-16")

    # Analysis type
    parser.add_argument(
        "-analysis_type",
        type=str,
        default="all",
        choices=[
            "hexabin",
            "mrae",
            "reconstruction",
            "all"
        ]
    )

    parser.add_argument(
        "-analysis_parameter",
        type=str,
        default="b2",
        choices=[
            "MvF",
            "BvF",
            "S",
            "Rv",
            "b2",
            "e1",
            "e2",
            "e3",
            "e4"
        ]
    )

    # Initialize setup
    parser.add_argument('-n_wavelengths', type=int, default = 47)
    parser.add_argument('-min_wavelength', type=int, default = 450)
    parser.add_argument('-max_wavelength', type=int, default = 818)
    parser.add_argument('-n_angles', type=int, default = 4)

    # Grid search or final model
    parser.add_argument('-synthesize_data', type=bool, default=True)
    parser.add_argument('-samples_grid_search', type=int, default=10000)
    parser.add_argument('-samples_final_model', type=int, default=10000)
    parser.add_argument('-batch_size', type= int, default=100)

    # Optical parameters
    # Volume optical parameters
    # BvF
    parser.add_argument('-BvF_min', type=float, default = 0.001)
    parser.add_argument('-BvF_max', type=float, default = 0.05)
    # S
    parser.add_argument('-S_min', type=float, default = 0.30)
    parser.add_argument('-S_max', type=float, default = 1.0)
    # Rv
    parser.add_argument('-Rv_min', type=float, default = 0.001)
    parser.add_argument('-Rv_max', type=float, default = 0.005)
    # MvF
    parser.add_argument('-MvF_min', type=float, default = 0.001)
    parser.add_argument('-MvF_max', type=float, default = 0.025)
    # d1
    parser.add_argument('-d1', type=float, default= 0.01)

    # Scattering parameters
    # a1
    parser.add_argument('-a1_min', type=float, default = 68.7)
    parser.add_argument('-a1_max', type=float, default = 68.7)
    # b1
    parser.add_argument('-b1_min', type=float, default = 1.161)
    parser.add_argument('-b1_max', type=float, default = 1.161)
    # a2
    parser.add_argument('-a2_min', type=float, default = 47.8)
    parser.add_argument('-a2_max', type=float, default = 47.8)
    # b2
    parser.add_argument('-b2_min', type=float, default = 2.453)
    parser.add_argument('-b2_max', type=float, default = 2.453)

    # Surface optical parameter
    # e
    parser.add_argument('-e_min', type=float, default= 0)
    parser.add_argument('-e_max', type=float, default= 0.10)

    # Constants
    # Hemoglobin concentration
    parser.add_argument('-c_hb', type=float, default = 4.7)
    # Total internal reflection coefficient
    parser.add_argument('-r_eff', type=float, default = 0.439)

    # Training parameters FM
    parser.add_argument('-learning_rate', type = float, default = 0.0001)
    parser.add_argument('-epochs', type = int, default = 10)

    # model characteristics
    args = parser.parse_args()
    return args