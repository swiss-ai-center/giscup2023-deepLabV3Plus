import yaml
import numpy as np
import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing
from tqdm import tqdm
from tensorflow.keras import models
from tensorflow.keras.layers import Layer
from numpy.lib.stride_tricks import sliding_window_view
from utils.loss_function import focal_tversky, weighted_bce_and_dice
from pathlib import Path
from utils.regions import get_tif_from_region
from shapely.geometry import Polygon
from shapely.affinity import affine_transform


def prediction_count(pred_shape, n_models, n_overlap, tile_size):
    # Compute for each pixels how many predictions there will be
    n_pred = np.zeros(pred_shape)

    for _ in np.arange(n_models):
        for overlap_index in np.arange(n_overlap):
            shift = overlap_index * (tile_size // n_overlap)

            # Split image into tiles
            tiles = sliding_window_view(n_pred, (tile_size, tile_size, 1))[
                shift::tile_size, shift::tile_size, :
            ]

            for row_idx, row in enumerate(tiles):
                for col_idx, col in enumerate(row):
                    y, x = row_idx * tile_size + shift, col_idx * tile_size + shift
                    n_pred[y : (y + tile_size), x : (x + tile_size)] += 1

    return n_pred


def predict_batch(
    *,
    model,
    pred_img,
    pred_count,
    batch_tiles,
    batch_coordinates,
    tile_size,
    shift,
    resize_back_layer,
):
    preds = model(np.stack(batch_tiles))

    for pred, (pred_row, pred_col) in zip(preds, batch_coordinates):
        y, x = pred_row * tile_size + shift, pred_col * tile_size + shift
        pred_img[y : (y + tile_size), x : (x + tile_size)] = (
            resize_back_layer(pred)
            / pred_count[y : (y + tile_size), x : (x + tile_size)]
        )


def predict_image_region(kwargs):
    n_models = kwargs["n_models"]
    skip_first_n_models = kwargs["skip_first_n_models"]
    image_file = kwargs["image_file"]
    region_num = kwargs["region_num"]
    tile_size = kwargs["tile_size"]
    tile_output_size = kwargs["tile_output_size"]
    n_overlap = kwargs["n_overlap"]
    batch_size = kwargs["batch_size"]

    resize_layer = tf.keras.layers.Resizing(
        height=tile_output_size, width=tile_output_size
    )
    resize_back_layer = tf.keras.layers.Resizing(height=tile_size, width=tile_size)

    geometries = []
    images = []
    region_nums = []

    print(f"Processing {image_file} region {region_num}...")

    # Get region raster
    tif_image, transform = get_tif_from_region(
        tif_path=f"data/raw/{image_file}", region_num=region_num
    )

    # Extract image from raster
    img_arr = np.moveaxis(tif_image, 0, -1)

    # Pad with zeroes
    img_arr = cv2.copyMakeBorder(
        img_arr,
        0,
        img_arr.shape[0] % tile_size,
        0,
        img_arr.shape[1] % tile_size,
        cv2.BORDER_CONSTANT,
    )

    pred_shape = (img_arr.shape[0], img_arr.shape[1], 1)
    pred_count = prediction_count(pred_shape, n_models, n_overlap, tile_size)
    global_pred = np.zeros(pred_shape)

    for model_idx in np.arange(skip_first_n_models, n_models):
        # Load model
        model = models.load_model(
            f"data/segmentation/deep_lab_v3_plus_{model_idx + 1}",
            custom_objects={"weighted_bce_and_dice": weighted_bce_and_dice},
        )

        for overlap_index in np.arange(n_overlap):
            shift = overlap_index * (tile_size // n_overlap)

            # Split image into tiles
            tiles = sliding_window_view(img_arr, (tile_size, tile_size, 3))[
                shift::tile_size, shift::tile_size, :
            ]

            tiles = np.squeeze(tiles, axis=2)

            # Build region's prediction image
            pred_img = np.zeros_like(global_pred)

            batch_tiles = []
            batch_coordinates = []
            batch_tile_count = 0

            for row_idx, row in enumerate(tiles):
                for col_idx, tile in enumerate(row):
                    # Filter out tiles without data
                    if np.any(tile != 0):
                        tile = tf.keras.applications.resnet50.preprocess_input(
                            np.float32(tile)
                        )
                        tile = resize_layer(tile)

                        batch_tiles.append(tile)
                        batch_coordinates.append((row_idx, col_idx))
                        batch_tile_count += 1

                    if batch_tile_count == batch_size:
                        predict_batch(
                            model=model,
                            pred_img=pred_img,
                            pred_count=pred_count,
                            batch_tiles=batch_tiles,
                            batch_coordinates=batch_coordinates,
                            tile_size=tile_size,
                            shift=shift,
                            resize_back_layer=resize_back_layer,
                        )

                        batch_tiles = []
                        batch_coordinates = []
                        batch_tile_count = 0

            # If there's still tiles remaining
            if batch_tile_count > 0:
                predict_batch(
                    model=model,
                    pred_img=pred_img,
                    pred_count=pred_count,
                    batch_tiles=batch_tiles,
                    batch_coordinates=batch_coordinates,
                    tile_size=tile_size,
                    shift=shift,
                    resize_back_layer=resize_back_layer,
                )

            global_pred += pred_img

    # Threshold prediction
    global_pred = np.uint8(global_pred > 0.5)

    # Find contours
    contours, _ = cv2.findContours(global_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Add polygon to list
    for contour in contours:
        # Polygon needs at least 4 points
        if contour.shape[0] < 4:
            continue

        # Convert rasterio (9 coeff) transform to shapely one (6 coeff).
        shapely_coeff = np.ravel([element for element in transform.column_vectors])

        # Apply transform to points
        polygon = affine_transform(Polygon(np.squeeze(contour)), shapely_coeff)

        # Add polygon to data
        geometries.append(polygon)
        images.append(image_file)
        region_nums.append(region_num)

    return geometries, images, region_nums


def predict(
    *,
    images: list[str],
    n_models: int,
    skip_first_n_models: int,
    tile_size: int,
    tile_output_size: int,
    num_processes: int,
    out_dir_path: Path,
    n_overlap: int,
    batch_size: int,
) -> None:
    gdf_data = {"geometry": [], "image": [], "region_num": []}

    tasks = []
    for image_file in images:
        for region_num in range(1, 7):
            tasks.append(
                {
                    "n_models": n_models,
                    "skip_first_n_models": skip_first_n_models,
                    "image_file": image_file,
                    "region_num": region_num,
                    "tile_size": tile_size,
                    "tile_output_size": tile_output_size,
                    "n_overlap": n_overlap,
                    "batch_size": batch_size,
                }
            )

    for task in tqdm(tasks, total=len(tasks), desc="Processing"):
        geometries, images, region_nums = predict_image_region(task)

        gdf_data["geometry"].extend(geometries)
        gdf_data["image"].extend(images)
        gdf_data["region_num"].extend(region_nums)

    # Save polygons to gpkg file
    out_dir_path.mkdir(exist_ok=True)

    # Convert to geopandas dataframe
    lake_pred_polygons = gpd.GeoDataFrame(data=gdf_data, crs="EPSG:3857")
    lake_pred_polygons.to_file(out_dir_path / "lake_polygons_pred.gpkg", driver="GPKG")


def segmentation_model_prediction(**kwargs) -> None:
    params = yaml.safe_load(open("params.yaml"))
    out_dir = Path("out")

    predict(
        images=params["images"],
        n_models=kwargs["n_models"],
        skip_first_n_models=kwargs["skip_first_n_models"],
        tile_size=kwargs["tile_size"],
        tile_output_size=kwargs["tile_output_size"],
        num_processes=kwargs["num_processes"],
        out_dir_path=out_dir,
        n_overlap=kwargs["n_overlap"],
        batch_size=kwargs["batch_size"],
    )
