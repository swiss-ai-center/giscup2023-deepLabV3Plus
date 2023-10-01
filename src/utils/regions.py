from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from shapely.geometry import Polygon


_regions_gdf = None


def _get_regions_gdf() -> gpd.GeoDataFrame:
    """
    Protected function to cache and get regions gdf

    Returns:
        gdf (gpd.GeoDataFrame): regions gdf
    """
    global _regions_gdf
    if _regions_gdf is None:
        _regions_gdf = gpd.read_file("data/raw/lakes_regions.gpkg")
    return _regions_gdf


def get_region(region_num: int) -> Polygon:
    """
    Get region from region number

    Args:
        region_num (int): region number (1-6)

    Returns:
        geo_bb_poly (Polygon): region polygon
    """
    regions_gdf = _get_regions_gdf()
    geo_bb_poly: Polygon = regions_gdf.loc[region_num - 1, "geometry"]
    return geo_bb_poly


def get_tif_from_region(
    *, tif_path: Path, region_num: int
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Get tif from region number

    Args:
        tif_path (str): path to tif file
        region_num (int): region number (1-6)

    Returns:
        image (np.ndarray): image array
        transform (rasterio.Affine): transform
    """
    region = get_region(region_num=region_num)

    with rasterio.open(tif_path) as image:
        image, transform = rasterio.mask.mask(image, [region], crop=True)

    return image, transform
