from pathlib import Path
import yaml
from methods.segmentation.deep_lab_v3_plus import deep_lab_v3_plus


def default() -> None:
    """Default method"""
    Path("models").mkdir(exist_ok=True, parents=True)


SEGMENTATION_METHODS = {
    "default": default,
    # Add below other methods
    "deep_lab_v3_plus": deep_lab_v3_plus,
}


def main() -> None:
    segmentation_params = yaml.safe_load(open("params.yaml"))["segmentation"]

    method = segmentation_params["method"]
    method_kwargs = segmentation_params[method]
    # Call the selected segmentation method function
    if method_kwargs:
        SEGMENTATION_METHODS[method](**method_kwargs)
    else:
        SEGMENTATION_METHODS[method]()


if __name__ == "__main__":
    main()
