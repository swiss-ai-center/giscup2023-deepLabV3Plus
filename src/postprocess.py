from pathlib import Path
import geopandas as gpd


def postprocess(
    *,
    pred_path: Path,
) -> None:
    pred: gpd.GeoDataFrame = gpd.read_file(pred_path)
    # merge overlapping geometries into one
    pred.geometry = pred.geometry.buffer(0)
    pred = pred.dissolve(by=["image", "region_num"]).reset_index()
    pred = pred.explode(index_parts=True)
    # remove small lakes under 100'000m2
    pred = pred[pred.area >= 100_000]
    pred = pred.reset_index(drop=True)

    # save to file
    pred.to_file(str(pred_path).replace(".gpkg", "_clean.gpkg"), driver="GPKG")


def main() -> None:
    postprocess(
        pred_path=Path("out/lake_polygons_pred.gpkg"),
    )


if __name__ == "__main__":
    main()
