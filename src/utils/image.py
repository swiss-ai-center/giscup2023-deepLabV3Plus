import numpy as np


def transform_tif_to_bgr_arr(tif_arr: np.ndarray) -> np.ndarray:
    """Transform tif array to bgr array"""
    arr = np.moveaxis(tif_arr, 0, -1)
    # rbg -> bgr
    return arr[:, :, [2, 1, 0]]
