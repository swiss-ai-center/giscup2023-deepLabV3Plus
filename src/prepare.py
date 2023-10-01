import yaml
from methods.prepare.tiling_ann_simple import tiling_ann_simple

PREPARE_METHODS = {
    "default": lambda: None,
    # Add below other methods
    "tiling_ann_simple": tiling_ann_simple,
}


def main() -> None:
    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]

    method = prepare_params["method"]
    method_kwargs = prepare_params[method]
    # Call the selected prepare method function
    if method_kwargs:
        PREPARE_METHODS[method](**method_kwargs)
    else:
        PREPARE_METHODS[method]()


if __name__ == "__main__":
    main()
