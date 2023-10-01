import multiprocessing
from pathlib import Path

import geopandas as gpd
import rasterio
import rasterio.features
import yaml


def create_lakes_annotations_rasters(
    images: [str], train_lake_polygons: gpd.GeoDataFrame, out_dir_path: Path
):
    """
    Create rasters annotations for a list of images and their corresponding lakes polygons

    Args:
        images ([str]): List of image file paths.
        train_lake_polygons (gpd.GeoDataFrame): GeoDataFrame containing lake polygons.
        out_dir_path (Path): Path to the directory where raster annotations will be saved.
    """

    print("Creating lakes annotation rasters")

    for image in images:
        # Select polygons corresponding to this image
        polygons = train_lake_polygons[(train_lake_polygons.image == image)].geometry

        create_lakes_annotations_raster(image, polygons, out_dir_path)


def create_lakes_annotations_raster(
    image: str, polygons: gpd.GeoSeries, out_dir_path: Path
):
    """
    Create a raster annotation for provided lakes polygons

    Args:
        image (str): File path to the image.
        polygons (gpd.GeoSeries): GeoSeries containing lake polygons.
        out_dir_path (Path): Path to the directory where the raster annotation will be saved.
    """

    print(f"Create lakes annotations raster {image}...")

    # Open raster as reference
    with rasterio.open(f"data/raw/{image}") as ref_raster:
        profile = ref_raster.profile.copy()
        profile.update(count=1)  # We update the number of bands to 1

    annotations_raster = rasterio.features.geometry_mask(
        polygons,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        all_touched=True,
        invert=True,
    )

    # Create raster
    with rasterio.open(
        f"{out_dir_path}/{image}".replace(".tif", "_ann.tif"), "w", **profile
    ) as raster:
        raster.write(annotations_raster, 1)


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    images = params["images"]
    out_dir_path = Path("data", "raw")
    train_lake_polygons = Path("data/raw/lake_polygons_training.gpkg")

    # Open lakes polygons annotation
    train_lake_polygons_df: gpd.GeoDataFrame = gpd.read_file(train_lake_polygons)

    # Create lakes annotations raster
    # in order to have a pixel-wise annotation of all lakes
    with multiprocessing.Pool(processes=len(images)) as pool:
        pool.starmap(
            create_lakes_annotations_rasters,
            [([im], train_lake_polygons_df, out_dir_path) for im in images],
        )


if __name__ == "__main__":
    main()
