import geopandas as gpd
import rasterio
import rasterio.mask
import yaml
from shapely import Polygon
from skimage.metrics import structural_similarity as ssim

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


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    images = params["images"]
    channel = 0  # 0 = red, 1 = green, 2 = blue
    win_size = 11

    for region_num in range(1, 7):
        print(f"Processing region {region_num}...")
        region: Polygon = get_region(region_num)
        last_diff = None
        for i in range(1, len(images)):
            with rasterio.open(images[i - 1], "r") as src1, rasterio.open(
                images[i], "r"
            ) as src2:
                print("Opening images...")
                tile1, transform = rasterio.mask.mask(src1, [region], crop=True)
                tile2, _ = rasterio.mask.mask(src2, [region], crop=True)
                print("Calculating SSIM...")
                _, diff = ssim(
                    tile1.transpose(1, 2, 0)[:, :, channel],
                    tile2.transpose(1, 2, 0)[:, :, channel],
                    full=True,
                    win_size=win_size,
                )
                if last_diff is not None:
                    # sum the last diff and the current diff
                    diff = (diff + last_diff) / 2
                last_diff = diff
                diff = (diff * 255).astype("uint8")
                if i == 1:
                    profile = src1.profile
                    profile.update(
                        count=1,
                        transform=transform,
                        height=diff.shape[0],
                        width=diff.shape[1],
                    )

    print("Writing tif...")
    with rasterio.open(f"data/raw/38m_diff_{region_num}.tif", "w", **profile) as dst:
        dst.write(diff, 1)
