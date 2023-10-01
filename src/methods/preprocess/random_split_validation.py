from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import os
import tarfile
import cv2
from sklearn.cluster import KMeans
import shutil
import matplotlib.pyplot as plt


def open_image_to_histogram(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist, _ = np.histogramdd(
        img.reshape(-1, 3), bins=(5, 5, 5), range=[(0, 256), (0, 256), (0, 256)]
    )

    return np.concatenate(hist, axis=None)


def image_clustering(
    paths: [Path],
):
    # Color histogram of each image
    histograms = np.vstack([open_image_to_histogram(path) for path in paths])
    kmeans = KMeans(n_clusters=4).fit(histograms)

    return np.stack(kmeans.labels_)


def balance_dataset(
    *,
    metadata_path: Path,
    images: list[str],
    training_data_regions: dict,
    lake_image_ratio: float,
    output_tiles_dir: Path,
    out_dir_path: Path,
) -> list[str]:
    """
    Balance dataset between images with and without lakes

    Args:
        metadata_path (Path): Path to metadata
        images (list[str]): List of images
        training_data_regions (dict): Training data regions
        lake_image_ratio (float): Ratio of images with lakes
        output_tiles_dir (Path): output dir for tiles
        out_dir_path (Path); output dir for this dvc stage
    """
    metadata = pd.read_csv(metadata_path, dtype={"has_lake": bool})

    # Filter any non-training data
    metadata = (
        pd.concat(
            [
                metadata[(metadata.image == image) & (metadata.region_num == region)]
                for image in images
                for region in training_data_regions[image]
            ],
            axis=0,
        )
        .rename(columns={"Unnamed: 0": "tile"})
        .set_index("tile")
    )

    # Shuffle
    metadata = metadata.sample(frac=1)

    # Balance dataset
    indices = metadata.index.to_numpy()
    has_lake = metadata["has_lake"].to_numpy()

    lake_images_indices = indices[has_lake]
    non_lake_images_indices = indices[~has_lake]

    max_non_lake_images = round(len(lake_images_indices) / lake_image_ratio)

    # Go through non lake images in a random order
    # and take as many as requested while picking from
    # each clusters found by dbscan
    non_lake_images_paths = np.stack(
        [
            os.path.join(output_tiles_dir, Path(path).name)
            for path in non_lake_images_indices
        ]
    )
    non_lake_images_clusters = image_clustering(non_lake_images_paths)
    non_lake_images_clusters = np.stack(non_lake_images_clusters)

    cluster_labels = np.unique(non_lake_images_clusters)
    print(f"Clusters: {list(cluster_labels)}")

    cluster_counts = np.bincount(non_lake_images_clusters)
    print(f"Clusters count: {list(cluster_counts)}")

    plt.bar(cluster_labels, cluster_counts)
    plt.xlabel("Clusters")
    plt.ylabel("Count")
    plt.savefig(os.path.join(out_dir_path, "cluster_counts_plot.png"))

    # Save some images of each clusters to show
    # what clusters look like
    for label in cluster_labels:
        selected_paths = np.random.choice(
            non_lake_images_paths[non_lake_images_clusters == label], 5
        )

        for idx, path in enumerate(selected_paths):
            shutil.copyfile(
                path, os.path.join(out_dir_path, f"cluster_{label}-{idx}.png")
            )

    # Choose non lake images with with probability
    # inversly proportional to its cluster's frequency
    inverse_probabilities = 1 / cluster_counts
    normalized_probabilities = inverse_probabilities / np.sum(inverse_probabilities)
    cluster_samples_probabilities = normalized_probabilities / cluster_counts

    selected_non_lakes_indices = np.random.choice(
        np.arange(len(non_lake_images_indices)),
        max_non_lake_images,
        p=[cluster_samples_probabilities[label] for label in non_lake_images_clusters],
    )

    selected_non_lakes_indices = non_lake_images_indices[selected_non_lakes_indices]

    return lake_images_indices.tolist() + selected_non_lakes_indices.tolist()


def random_split_validation(**kwargs) -> None:
    params = yaml.safe_load(open("params.yaml", "r"))
    np.random.seed(kwargs["seed"])

    original_metadata = pd.read_csv(kwargs["metadata_path"]).rename(
        columns={"Unnamed: 0": "tile"}
    )

    out_dir_path = Path(params["preprocess"]["out"])
    out_dir_path.mkdir(exist_ok=True, parents=True)

    # Untar images
    print("[INFO] Untarring tiles...")

    tiles_dir_path = Path(kwargs["tiles_dir"])
    output_tiles_dir = out_dir_path / os.path.basename(tiles_dir_path.stem)

    annotations_dir_path = Path(kwargs["annotations_dir"])
    output_annotations_dir = out_dir_path / os.path.basename(annotations_dir_path.stem)

    # Extract all annotations and tile from the archive to their outputs directorys
    with tarfile.open(tiles_dir_path, "r") as tiles:
        tiles.extractall(output_tiles_dir)

    print("[INFO] Finished untarring tiles...")

    # Balance dataset
    kept_images_indices = balance_dataset(
        metadata_path=kwargs["metadata_path"],
        images=params["images"],
        training_data_regions=params["training_data_regions"],
        lake_image_ratio=kwargs["lake_image_ratio"],
        output_tiles_dir=output_tiles_dir,
        out_dir_path=out_dir_path,
    )

    # Filter metadata based on kept images after balancing
    EMPTY = "empty"
    metadata = original_metadata[original_metadata.tile.isin(kept_images_indices)]
    metadata = metadata.fillna(value=EMPTY)

    # Print summary
    print("[INFO] Summary:")
    print(f"  - Nbr. of original images: {len(original_metadata)}")
    print(f"  - Nbr. of filtered images: {len(metadata)}")

    # Extract annotations
    print("[Info] Untarring annotations...")

    with tarfile.open(annotations_dir_path, "r") as annotations:
        for annot_path in metadata["tile"]:
            annotations.extract(Path(annot_path).name, output_annotations_dir)

    print("[Info] Finish untarring annotations...")

    print("[Info] Remove filtred non lake tiles from disk...")

    # Remove filtered images from disk
    for tile in original_metadata[
        ~original_metadata.tile.isin(kept_images_indices)
    ].tile:
        os.remove(os.path.join(output_tiles_dir, f"{Path(tile).name}"))

    # Create dataset dataframe
    is_val = np.full((len(metadata["tile"])), False)
    is_val[
        np.random.choice(
            np.arange(len(metadata["tile"])),
            round(len(metadata["tile"]) * kwargs["val_ratio"]),
        )
    ] = True

    pd.DataFrame(
        {
            "img": [output_tiles_dir / Path(path).name for path in metadata["tile"]],
            "annotation": [
                output_annotations_dir / Path(path).name for path in metadata["tile"]
            ],
            "region_num": metadata["region_num"],
            "has_lake": metadata["has_lake"],
            "is_val": is_val,
        }
    ).to_csv(out_dir_path / "dataset.csv", index=False)
