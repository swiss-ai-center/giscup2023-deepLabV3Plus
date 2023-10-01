import multiprocessing
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Polygon
from tqdm import tqdm

from utils.image import transform_tif_to_bgr_arr
from utils.io import tar_folder
from utils.regions import get_region, get_tif_from_region


def is_mostly_black(img, threshold=0.6):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    black_pixel_count = np.sum(img == 0)
    black_pixel_percentage = black_pixel_count / (img.shape[0] * img.shape[1])

    # Check if the image is mostly black
    return black_pixel_percentage >= threshold


def extract_tiles_from_image_region(kwargs: dict) -> dict:
    """
    Extract tiles from image region

    Args:
        kwargs (dict): Keyword arguments (see below)

    Keyword Args:
        image_path (Path): Path to image
        region_num (int): Region number
        tile_size (int): Tile size
        tile_overlap (float): Tile overlap
        tile_output_size (int): Tile output size
        out_dir_path (Path): Path to output directory
        extract_dem (bool): Whether to extract DEM

    Returns:
        list: Metadata dictionary
    """
    image_path = kwargs["image_path"]
    ann_image_path = kwargs["ann_image_path"]
    region_num = kwargs["region_num"]
    tile_size = kwargs["tile_size"]
    tile_overlap = kwargs["tile_overlap"]
    tile_output_size = kwargs["tile_output_size"]
    out_dir_path = kwargs["out_dir_path"]
    extract_dem = kwargs["extract_dem"]

    tiles_out_path = out_dir_path / "tiles"
    tiles_out_path.mkdir(parents=True, exist_ok=True)
    ann_tiles_out_path = out_dir_path / "ann_tiles"
    ann_tiles_out_path.mkdir(parents=True, exist_ok=True)
    if extract_dem:
        dem_tiles_out_path = out_dir_path / "dem_tiles"
        dem_tiles_out_path.mkdir(parents=True, exist_ok=True)

    image = image_path.name
    geo_bb_poly = get_region(region_num)

    tif_image, transform = get_tif_from_region(
        tif_path=image_path, region_num=region_num
    )
    tif_ann_image, _ = get_tif_from_region(
        tif_path=ann_image_path, region_num=region_num
    )
    if extract_dem:
        tif_dem_path = (
            "data/raw/38m_v4.1_dem_regions_1-4_clipped_grad.tif"
            if region_num < 5
            else "data/raw/38m_v4.1_dem_regions_5-6_clipped_grad.tif"
        )
        tif_dem, _ = get_tif_from_region(tif_path=tif_dem_path, region_num=region_num)
        # scale dem to [0, 255]
        tif_dem_arr = np.uint8(
            (tif_dem - tif_dem.min()) / (tif_dem.max() - tif_dem.min()) * 255
        )
        tif_dem_arr = np.moveaxis(tif_dem_arr, 0, -1)

    img_arr = transform_tif_to_bgr_arr(tif_image)
    img_arr = np.uint8(img_arr)
    ann_img_arr = np.uint8(tif_ann_image) * 255
    ann_img_arr = np.moveaxis(ann_img_arr, 0, -1)

    tot_tiles = 0
    metadata = {}

    for y in range(0, img_arr.shape[0], tile_size - round(tile_size * tile_overlap)):
        for x in range(
            0, img_arr.shape[1], tile_size - round(tile_size * tile_overlap)
        ):
            tile = img_arr[y : y + tile_size, x : x + tile_size]

            if is_mostly_black(tile):
                continue

            ann_tile = ann_img_arr[y : y + tile_size, x : x + tile_size]
            if extract_dem:
                dem_tile = tif_dem_arr[y : y + tile_size, x : x + tile_size]

            poly = Polygon(
                (
                    transform * (x, y),
                    transform * (x + tile_size, y),
                    transform * (x + tile_size, y + tile_size),
                    transform * (x, y + tile_size),
                )
            )
            is_tile_in_poly = geo_bb_poly.intersects(poly)
            if is_tile_in_poly and tile.shape[:2] == (tile_size, tile_size):
                filename = f"{image}_{region_num}_{tot_tiles}.png"
                # RGB Tile
                tile = cv2.resize(tile, (tile_output_size, tile_output_size))
                cv2.imwrite(str(tiles_out_path / filename), tile)
                # Mask Annotation Tile
                ann_tile = cv2.resize(ann_tile, (tile_output_size, tile_output_size))
                cv2.imwrite(str(ann_tiles_out_path / filename), ann_tile)
                # DEM Tile
                if extract_dem:
                    dem_filename = f"{region_num}_{tot_tiles}.png"
                    dem_tile = cv2.resize(
                        dem_tile, (tile_output_size, tile_output_size)
                    )
                    cv2.imwrite(str(dem_tiles_out_path / dem_filename), dem_tile)
                # Save metadata
                metadata[filename] = {
                    "image": image,
                    "region_num": region_num,
                    "has_lake": bool(np.any(ann_tile)),
                    **(
                        dict(dem_path=str(dem_tiles_out_path / dem_filename))
                        if extract_dem
                        else {}
                    ),
                }
                tot_tiles += 1
    return metadata


def prepare_dataset(
    *,
    images: list[str],
    tile_size: int,
    tile_overlap: float,
    tile_output_size: int,
    extract_dem: bool,
    training_data_regions: dict,
    out_dir_path: Path,
) -> None:
    """
    Prepare dataset by extracting tiles from image regions with multiprocessing

    Args:
        images (list[str]): List of images
        tile_size (int): Tile size
        tile_overlap (float): Tile overlap
        tile_output_size (int): Tile output size
        extract_dem (bool): Whether to extract DEM
        training_data_regions (dict): Training data regions
        out_dir_path (Path): Path to output directory
    """
    out_dir_path.mkdir(exist_ok=True, parents=True)

    tasks = []
    for image in images:
        for region_num in training_data_regions[image]:
            tasks.append(
                {
                    "image_path": Path("data/raw") / image,
                    "ann_image_path": Path("data/raw")
                    / image.replace(".tif", "_ann.tif"),
                    "region_num": region_num,
                    "tile_size": tile_size,
                    "tile_overlap": tile_overlap,
                    "tile_output_size": tile_output_size,
                    "out_dir_path": out_dir_path,
                    "extract_dem": extract_dem,
                }
            )

    all_metadata = {}
    with multiprocessing.Pool(processes=8) as pool:
        for metadata in tqdm(
            pool.imap_unordered(extract_tiles_from_image_region, tasks),
            total=len(tasks),
            desc="Extracting tiles from images",
        ):
            all_metadata.update(metadata)

    # Tar tiles
    tiles_out_path = out_dir_path / "tiles"
    tiles_tar_path = out_dir_path / "tiles.tar"
    tar_folder(
        src_folder=tiles_out_path, dest_path=tiles_tar_path, desc="Tarring tiles"
    )
    shutil.rmtree(tiles_out_path)

    ann_tiles_out_path = out_dir_path / "ann_tiles"
    ann_tiles_tar_path = out_dir_path / "ann_tiles.tar"
    tar_folder(
        src_folder=ann_tiles_out_path,
        dest_path=ann_tiles_tar_path,
        desc="Tarring annotation tiles",
    )
    shutil.rmtree(ann_tiles_out_path)

    if extract_dem:
        dem_tiles_out_path = out_dir_path / "dem_tiles"
        dem_tiles_tar_path = out_dir_path / "dem_tiles.tar"
        tar_folder(
            src_folder=dem_tiles_out_path,
            dest_path=dem_tiles_tar_path,
            desc="Tarring DEM tiles",
        )
        shutil.rmtree(dem_tiles_out_path)

    # Save metadata to .csv file
    print("[INFO] Saving metadata...")
    metadata_path = out_dir_path / "metadata.csv"
    pd.DataFrame.from_dict(all_metadata, orient="index").to_csv(
        metadata_path, index=True
    )

    print("Summary:")
    print(f"Extracted {len(metadata)} tiles")
    print(f"Of which {sum([v['has_lake'] for v in all_metadata.values()])} have a lake")


def tiling_ann_simple(**kwargs) -> None:
    params = yaml.safe_load(open("params.yaml"))

    prepare_dataset(
        images=params["images"],
        tile_size=kwargs["tile_size"],
        tile_overlap=kwargs["tile_overlap"],
        tile_output_size=kwargs["tile_output_size"],
        extract_dem=kwargs["extract_dem"],
        training_data_regions=params["training_data_regions"],
        out_dir_path=Path(params["prepare"]["out"]),
    )
