from pathlib import Path
import yaml
from methods.process.segmentation_model_prediction import segmentation_model_prediction

PROCESS_METHODS = {
    "default": lambda: None,
    # Add below other methods
    "segmentation_model_prediction": segmentation_model_prediction,
}


def main() -> None:
    process_params = yaml.safe_load(open("params.yaml"))["process"]

    method = process_params["method"]
    method_kwargs = process_params[method]
    # Call the selected segmentation method function
    if method_kwargs:
        PROCESS_METHODS[method](**method_kwargs)
    else:
        PROCESS_METHODS[method]()


if __name__ == "__main__":
    main()
