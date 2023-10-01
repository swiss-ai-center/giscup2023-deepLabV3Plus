import json
import multiprocessing
from pathlib import Path
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import yaml
import shapely
import matplotlib.pyplot as plt


def compute_base_metrics(
    pred: gpd.GeoDataFrame,
    test: gpd.GeoDataFrame,
) -> dict:
    """Compute the metrics"""

    intersections = gpd.overlay(
        pred,
        test,
        how="intersection",
        keep_geom_type=False,
    )
    unions = gpd.overlay(
        pred,
        test,
        how="union",
        keep_geom_type=False,
    )
    # Jaccard Index
    accuracy = intersections.area.sum() / unions.area.sum()
    # Sorensen-Dice Coefficient
    f1 = 2 * intersections.area.sum() / (pred.area.sum() + test.area.sum())
    return {
        "accuracy": accuracy,
        "f1": f1,
    }


def calc_metrics(
    pred: gpd.GeoDataFrame,
    test: gpd.GeoDataFrame,
) -> dict:
    """Calculate the metrics based on true/false positives/negatives"""
    non_associated_training_lakes = test[~test.index.isin(pred.training_poly_id)]

    tp = (~pred.false_negative).values.sum()
    fp = pred.false_positive.values.sum()
    fn = pred.false_negative.values.sum() + len(non_associated_training_lakes)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = tp / (tp + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }


def compute_comp_metrics(
    pred: gpd.GeoDataFrame,
    test: gpd.GeoDataFrame,
) -> tuple[dict, gpd.GeoDataFrame]:
    """
    Compute the metrics based on the competition rules.

    Note that we modify the pred dataframe and return it with
    the false_negative, false_positive and training_poly_id columns.
    """

    pred = pred.copy()
    test = test.copy()
    pred.loc[:, "false_negative"] = False
    pred.loc[:, "false_positive"] = False
    pred.loc[:, "training_poly_id"] = None

    for index, poly in pred.iterrows():
        other_polys = pred[~pred.index.isin([index])]
        # 1. The polygon does not overlap any other polygon corresponding to the same image
        #    and region, in the same submitted GeoJson file. Duplicate or overlapping polygons
        #    will be classified as “false positives”. This avoids contestants outlining the
        #    same lake in the same image multiple times and having each considered a
        #    “true positive” to generate inflated F1 scores. Competitors should take care
        #    that each lake is associated with the correct image.
        self_overlaps = (
            len(
                other_polys.sjoin(
                    gpd.GeoDataFrame(geometry=[poly.geometry], crs=pred.crs),
                    how="inner",
                    predicate="intersects",
                )
            )
            > 0
        )
        # 2. The polygon partially or fully overlaps at least one polygon identified in the
        #    “test” dataset. (If yes, go to Step 3. If no, it is considered “False Positive.”).

        # overlap_training dataframe contains the intersection of the current polygon with
        # the training polygons as well as the associated training polygon index
        overlap_training = test.sjoin(
            gpd.GeoDataFrame(geometry=[poly.geometry], crs=pred.crs),
            how="inner",
            predicate="intersects",
        )
        nbr_overlap_training = len(overlap_training)

        if self_overlaps or nbr_overlap_training == 0:
            pred.loc[index, "false_positive"] = True
        else:
            #    If it overlaps more than one test polygon, it will be associated with the test
            #    polygon for which it has the greatest overlap (by area). Lakes will not be penalized
            #    for partially overlapping or “clipping” the corner of a nearby lake polygon in the
            #    test dataset.
            if nbr_overlap_training > 1:
                overlap_training["area"] = overlap_training.geometry.area
                overlap_training = overlap_training.sort_values(
                    by="area", ascending=False
                )
            candidate_training_poly = overlap_training.iloc[0]
            # 3. The submitted polygon area is no less than 50% of the area of the overlapping test polygon identified in Step 2.
            # 4. The submitted polygon area is no more than 160% of the area of the overlapping test polygon identified in Step 2.
            try:  # TODO: remove try/except and check for invalid geometries before with shapely.Polygon.is_valid
                overlap_area = candidate_training_poly.geometry.intersection(
                    poly.geometry
                ).area
            except shapely.errors.GEOSException:
                overlap_area = 0

            min_area_valid = overlap_area / candidate_training_poly.geometry.area >= 0.5
            max_area_valid = overlap_area / candidate_training_poly.geometry.area <= 1.6
            if min_area_valid and max_area_valid:
                # .name is the id of the training polygon
                pred.loc[index, "training_poly_id"] = candidate_training_poly.name
            else:
                pred.loc[index, "false_negative"] = True

    # 5. After assessing each submitted lake, each remaining test polygon that does not have an associated submitted lake, will be
    # considered a “False Negative” in computing the F1 score.
    metrics = calc_metrics(pred, test)

    return metrics, pred


def plot_lakes(
    *,
    pred: gpd.GeoDataFrame,
    test: gpd.GeoDataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the polygons with different colors for true/false positives/negatives"""
    pred = pred.copy()
    test = test.copy()

    # False negatives are blue
    pred.loc[:, "color"] = "blue"
    # True positives are green
    pred.loc[~pred.false_negative, "color"] = "green"
    # False positives are red
    pred.loc[pred.false_positive, "color"] = "red"
    # Unassociated training lakes are gray
    test.loc[:, "color"] = "gray"
    # Associated training lakes are black
    test.loc[
        test.index.isin(pred.training_poly_id),
        "color",
    ] = "black"

    fig, ax = plt.subplots(figsize=(16, 16))
    plt.tight_layout()
    # Plot the polygons
    test.plot(ax=ax, color=test.color)
    pred.plot(ax=ax, color=pred.color, alpha=0.75)
    return fig, ax


def make_summary(
    *,
    images: list[str],
    pred: gpd.GeoDataFrame,
    test: gpd.GeoDataFrame,
):
    """Create a summary of the metrics"""
    summary = {}

    # Compute metrics for the whole dataset
    summary.update(compute_base_metrics(pred, test))
    summary.update(
        {f"{name}_comp": val for name, val in calc_metrics(pred, test).items()}
    )

    # Compute metrics for each region
    for region in range(1, 7):
        pred_region = pred[pred.region_num == region]
        test_region = test[test.region_num == region]
        summary.update(compute_base_metrics(pred_region, test_region))
        summary.update(
            {
                f"{name}_comp_region_{region}": val
                for name, val in calc_metrics(pred_region, test_region).items()
            }
        )

    # Compute metrics for each image
    for image in images:
        pred_image = pred[pred.image == image]
        test_image = test[test.image == image]
        summary.update(compute_base_metrics(pred_image, test_image))
        summary.update(
            {
                f"{name}_comp_{image}": val
                for name, val in calc_metrics(pred_image, test_image).items()
            }
        )

    # Sort dict by key
    summary = dict(sorted(summary.items(), key=lambda item: item[0]))
    return summary


def evaluate_image_region(kwargs) -> tuple[dict, gpd.GeoDataFrame]:
    """
    Evaluate the predictions for a single image region

    Args:
        kwargs (dict): Keyword arguments (see below)

    Keyword Args:
        image (str): Image name
        region_num (int): Region number
        pred (gpd.GeoDataFrame): Predictions
        test (gpd.GeoDataFrame): Test data
        out_dir (Path): Output directory

    Returns:
        tuple[dict, gpd.GeoDataFrame]: Metrics and updated predictions
    """
    image: str = kwargs["image"]
    region_num: int = kwargs["region_num"]
    pred: gpd.GeoDataFrame = kwargs["pred"]
    test: gpd.GeoDataFrame = kwargs["test"]
    out_dir: Path = kwargs["out_dir"]

    # Get subset of lake polygons for region
    pred_region = pred[(pred.image == image) & (pred.region_num == region_num)]
    test_region = test[(test.image == image) & (test.region_num == region_num)]
    base_metrics = compute_base_metrics(
        pred=pred_region,
        test=test_region,
    )
    comp_metrics, updated_pred_region = compute_comp_metrics(
        pred=pred_region,
        test=test_region,
    )

    fig, _ = plot_lakes(
        pred=updated_pred_region,
        test=test_region,
    )
    (out_dir / "plots").mkdir(exist_ok=True, parents=True)
    fig.savefig(out_dir / "plots" / f"{image}_{region_num}.png", dpi=600)

    return {
        "image": image,
        "region": region_num,
        **base_metrics,
        "precision_comp": comp_metrics["precision"],
        "recall_comp": comp_metrics["recall"],
        "accuracy_comp": comp_metrics["accuracy"],
        "f1_comp": comp_metrics["f1"],
    }, updated_pred_region


def evaluate(*, pred_path: Path, test_path: Path, out_dir: Path) -> None:
    """
    Evaluate the performance of the lake polygon prediction algorithm. Saves metrics and
    plots to the output directory.

    Args:
        pred_path (Path): Path to the predicted lake polygons.
        test_path (Path): Path to the training lake polygons.
        out_dir (Path): Path to the output directory.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    params = yaml.safe_load(open("params.yaml"))

    pred: gpd.GeoDataFrame = gpd.read_file(pred_path)
    updated_pred = None
    test: gpd.GeoDataFrame = gpd.read_file(test_path)

    metrics: list[dict] = []
    tasks = []
    for image in params["images"]:
        for region_num in params["training_data_regions"][image]:
            tasks.append(
                {
                    "image": image,
                    "region_num": region_num,
                    "pred": pred,
                    "test": test,
                    "out_dir": out_dir,
                }
            )

    with multiprocessing.Pool(processes=1) as pool:
        for result in tqdm(
            pool.imap_unordered(evaluate_image_region, tasks),
            total=len(tasks),
            desc="Evaluating image regions",
        ):
            metric, updated_pred_region = result
            metrics.append(metric)
            if updated_pred is None:
                updated_pred = updated_pred_region
            else:
                updated_pred = pd.concat([updated_pred, updated_pred_region])

    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save summary
    summary = make_summary(images=params["images"], pred=updated_pred, test=test)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    evaluate(
        pred_path=Path("out/lake_polygons_pred_clean.gpkg"),
        test_path=Path("data/raw/lake_polygons_training.gpkg"),
        out_dir=Path("evaluation"),
    )


if __name__ == "__main__":
    main()
