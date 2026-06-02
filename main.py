from args_parse import args_parser
from grid_search import GridSearch
from final_model import FinalModel
from analysis import Analysis

def main():

    args = args_parser()

    # ==================================
    # Grid search
    # ==================================

    if args.mode == "grid_search":

        print(
            f"Starting grid search for {args.model}"
        )

        grid = GridSearch()

        # =========================
        # PMLP
        # =========================

        if args.model in ["PMLP_OTCM", "PMLP_TTCM"]:

            if args.grid_search_type == "model_architecture":

                grid.model_mlp()

            elif args.grid_search_type == "learning_parameters":

                grid.learning_parameters_mlp()

            elif args.grid_search_type == "regularization":

                grid.regularization_mlp()

            else:

                raise ValueError(
                    f"{args.grid_search_type} not available for PMLP"
                )

        # =========================
        # Transformer
        # =========================

        elif args.model in ["T_OTCM", "T_TTCM"]:

            if args.grid_search_type == "SET_architecture":

                grid.set_transformer()

            elif args.grid_search_type == "AET_architecture":

                grid.aet_transformer()

            elif args.grid_search_type == "learning_parameters":

                grid.learning_parameters_transformer()

            else:

                raise ValueError(
                    f"{args.grid_search_type} not available for Transformer"
                )

    # ==================================
    # Final model training
    # ==================================

    elif args.mode == "final_model":

        print(f"Training final model {args.model}")

        trainer = FinalModel()

        if args.model in ["PMLP_OTCM", "PMLP_TTCM"]:
            trainer.train_mlp()

        elif args.model in ["T_OTCM", "T_TTCM"]:
            trainer.train_transformer()

    # ==================================
    # Analysis
    # ==================================

    elif args.mode == "analysis":

        analysis = Analysis(
            model_dates={
                "PMLP_OTCM": args.date_PMLP_OTCM,
                "PMLP_TTCM": args.date_PMLP_TTCM,
                "T_OTCM": args.date_T_OTCM,
                "T_TTCM": args.date_T_TTCM,
            }
        )

        if args.analysis_type == "hexabin":

            analysis.hexabin_plot(
                current_model=args.model,
                parameter=args.analysis_parameter
            )

        elif args.analysis_type == "mrae":

            analysis.MRAE_plot_synthetic()

        elif args.analysis_type == "reconstruction":

            analysis.reconstruction_synthetic()

        elif args.analysis_type == "all":

            parameters = [
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

            models = [
                "PMLP_OTCM",
                "PMLP_TTCM",
                "T_OTCM",
                "T_TTCM"
            ]

            for model in models:
                for parameter in parameters:

                    analysis.hexabin_plot(
                        current_model=model,
                        parameter=parameter
                    )

            analysis.MRAE_plot_synthetic()
            analysis.reconstruction_synthetic()

    else:

        raise ValueError(
            f"Unknown mode: {args.mode}"
        )


if __name__ == "__main__":
    main()