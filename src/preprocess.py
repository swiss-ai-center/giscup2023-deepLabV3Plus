from pathlib import Path
import yaml
from methods.preprocess.random_split_validation import random_split_validation


def default() -> None:
    """Default method"""
    Path("data/preprocessed").mkdir(exist_ok=True, parents=True)


PREPROCESS_METHODS = {
    "default": default,
    # Add below other methods
    "random_split_validation": random_split_validation,
}


def main() -> None:
    preprocess_params = yaml.safe_load(open("params.yaml"))["preprocess"]

    method = preprocess_params["method"]
    method_kwargs = preprocess_params[method]
    # Call the selected prepare method function
    if method_kwargs:
        PREPROCESS_METHODS[method](**method_kwargs)
    else:
        PREPROCESS_METHODS[method]()


if __name__ == "__main__":
    main()
