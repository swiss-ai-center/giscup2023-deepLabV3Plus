from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from utils.regions import get_region


def run_sanity_checks(
    test_polygons: gpd.GeoDataFrame, expected_crs: str, expected_images: list[str]
) -> None:
    """Checks that the dataframe is in the correct format for submission.

    Args:
        test_polygons (gpd.GeoDataFrame): The test polygons to check.
        expected_crs (str): The expected CRS.
        expected_images (list[str]): The expected image keys.
    """
    # Check that the geometry is correct
    actual_geometry_types = set(test_polygons.geometry.geom_type.unique().tolist())
    expected_geometry_types = set(["Polygon"])
    assert (
        actual_geometry_types == expected_geometry_types
    ), f"Geometry types are incorrect. Found {actual_geometry_types}, expected {expected_geometry_types}"
    # Check that the columns are correct
    actual_columns = set(test_polygons.columns.tolist())
    expected_columns = set(["image", "region_num", "geometry"])
    assert (
        actual_columns == expected_columns
    ), f"Columns are incorrect. Found {actual_columns}, expected {expected_columns}"
    # Check that the CRS is correct
    assert (
        test_polygons.crs == expected_crs
    ), f"CRS is incorrect. Found {test_polygons.crs}, expected {expected_crs}"
    # Check that the images are correct
    actual_images = set(test_polygons.image.unique().tolist())
    expected_images = set(expected_images)
    assert (
        actual_images == expected_images
    ), f"Images are incorrect. Found {actual_images}, expected {expected_images}"
    # Check that the region numbers are correct
    actual_region_nums = set(test_polygons.region_num.unique().tolist())
    expected_region_nums = set(range(1, 7))
    assert (
        actual_region_nums
    ) == expected_region_nums, f"Region numbers are incorrect. Found {actual_region_nums}, expected {expected_region_nums}"
    # Check that the geometries are valid
    assert (
        test_polygons.geometry.is_valid.all()
    ), "Some geometries are invalid. Please check and try again."
    # Check that the geometries have more than 4 points
    assert (
        test_polygons.geometry.apply(lambda x: len(x.exterior.coords)).min() > 4
    ), "Some geometries have less than 4 points. Please check and try again."
    # Check that the geometries do not overlap by image and region
    assert (
        test_polygons.groupby(["image", "region_num"])
        .apply(lambda x: not x.geometry.overlaps(x.geometry).any())
        .all()
    ), "Some geometries overlap. Please check and try again."


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    pred_path = Path("out/lake_polygons_pred_clean.gpkg")
    pred_polygons: gpd.GeoDataFrame = gpd.read_file(pred_path)
    test_polygons = gpd.GeoDataFrame(
        columns=pred_polygons.columns, crs=pred_polygons.crs
    )

    for image in params["images"]:
        for region_num in params["test_data_regions"][image]:
            region_poly = get_region(region_num=region_num)
            filtered_pred_polygons = pred_polygons[
                (pred_polygons.image == image)
                & (pred_polygons.region_num == region_num)
            ]
            intersects = filtered_pred_polygons.intersects(region_poly)
            idxs = intersects[intersects].index
            # Append to test_polygons
            test_polygons = pd.concat([test_polygons, filtered_pred_polygons.loc[idxs]])

    run_sanity_checks(
        test_polygons, expected_crs=params["crs"], expected_images=params["images"]
    )

    test_polygons.to_file("out/lake_polygons_test.gpkg", driver="GPKG")


if __name__ == "__main__":
    main()