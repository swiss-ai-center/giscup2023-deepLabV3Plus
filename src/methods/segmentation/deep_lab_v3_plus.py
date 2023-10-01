import geopandas as gpd
import pandas as pd
import yaml
import numpy as np
import cv2
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from utils.deep_lab_v3_plus import DeeplabV3Plus
from utils.loss_function import focal_tversky, weighted_bce_and_dice
from utils.lr_finder import LRFinder
from pathlib import Path
from functools import partial
from typing import Callable, Dict, Any, Literal
from tensorflow.keras.layers import Layer
from tqdm import tqdm
import albumentations as A
import tensorflow_addons as tfa


def plot_training_history(history, label, has_val=True):
    plt.plot(history.history["binary_io_u"], label=f"{label} - Training IoU")

    if has_val:
        plt.plot(history.history["val_binary_io_u"], label=f"{label} - Validation IoU")

    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Training History")
    plt.legend()


def augment(augmentations: Dict[str, Any]):
    # See : https://albumentations.ai/docs/examples/tensorflow-example/
    def fn(image, target):
        data = {"image": image, "mask_1": target}
        aug_data = augmentations(**data)
        aug_img = aug_data["image"]
        aug_target = aug_data["mask_1"]

        return aug_img, aug_target

    return fn


def set_shapes(input_shape: tuple[int, int, int], target_shape: tuple[int, int, int]):
    def fn(input_image, target_image):
        input_image.set_shape(input_shape)
        target_image.set_shape(target_shape)

        return input_image, target_image

    return fn


def augment_images(
    dataset: tf.data.Dataset,
    augmentations: Dict[str, Any],
    input_shape: tuple,
    output_shape: tuple,
):
    dataset = dataset.map(
        lambda img, annotation: tf.numpy_function(
            func=augment(augmentations),
            inp=[img, annotation],
            Tout=(tf.float32, tf.float32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Fix incorrect shape that occurs after augmentations
    dataset = dataset.map(set_shapes(input_shape, output_shape))

    return dataset


def process_image_and_annotation(
    tile_size: int,
):
    resize_layer = tf.keras.layers.Resizing(height=tile_size, width=tile_size)

    def process_fun(
        image_path: tf.string,
        annotation_path: tf.string,
    ):
        # Load the raw data from the file as a string
        img = tf.io.read_file(image_path)
        img = tf.io.decode_png(img, channels=3)
        img = tf.cast(img, tf.float32)
        img = resize_layer(img)
        img = tf.keras.applications.resnet50.preprocess_input(img)

        annotation = tf.io.read_file(annotation_path)
        annotation = tf.io.decode_png(annotation, channels=1)
        annotation = tf.cast(annotation, tf.float32)
        annotation = resize_layer(annotation)
        annotation = annotation / 255.0

        return img, annotation

    return process_fun


def get_time_label_from_image(image_path: str):
    if not hasattr(get_time_label_from_image, "image_time"):
        get_time_label_from_image.image_time = dict(
            {
                "Greenland26X_22W_Sentinel2_2019-06-03_05.tif": 1,
                "Greenland26X_22W_Sentinel2_2019-06-19_20.tif": 2,
                "Greenland26X_22W_Sentinel2_2019-07-31_25.tif": 3,
                "Greenland26X_22W_Sentinel2_2019-08-25_29.tif": 4,
            }
        )

    return get_time_label_from_image.image_time[
        Path(image_path).stem[: len("Greenland26X_22W_Sentinel2_2019-06-19_20.tif")]
    ]


def data_generator(dataset_df: pd.DataFrame):
    for _, row in dataset_df.iterrows():
        yield row["img"], row["annotation"]


def create_dataset(
    dataset_df: pd.DataFrame,
    input_shape: list[int],
    target_shape: list[int],
    shuffle_buffer_size: int = 512,
    augmentations: Dict[str, Any] = None,
):
    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        partial(data_generator, dataset_df),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )

    # Define process function
    process_function = process_image_and_annotation(tile_size=input_shape[0])

    # Preprocess and batch the datase
    dataset = dataset.map(process_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()

    if augmentations is not None:
        # Important: do augmentations after caching with cache().
        # Because otherwise we would have same augmentations each epochs
        dataset = augment_images(dataset, augmentations, input_shape, target_shape)

    dataset = dataset.shuffle(shuffle_buffer_size)

    # Make it compatible for multi input model
    dataset = dataset.map(lambda image, annotations: (image, annotations))

    return dataset


def get_loss_function(
    name: Literal["focal_tversky", "binary_crossentropy", "weighted_bce_and_dice"]
):
    if name == "focal_tversky":
        loss_function = focal_tversky()
    elif name == "weighted_bce_and_dice":
        loss_function = weighted_bce_and_dice
    elif name == "binary_crossentropy":
        loss_function = tf.keras.losses.BinaryCrossentropy()
    else:
        raise ValueError(f"Unknown loss function {loss_function}")

    return loss_function


def learning_rate_analysis(
    *,
    train_dataset,
    n_train_samples,
    batch_size,
    loss_function,
    input_shape,
    freeze_backbone_model,
):
    # Multi-GPU strategy
    mirrored_strategy = tf.distribute.MirroredStrategy()

    # Define a multi-gpu strategy
    # Model needs to be created and compiled here
    with mirrored_strategy.scope():
        model = DeeplabV3Plus(input_shape, freeze_backbone_model)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(clipnorm=1, momentum=0.99),
            loss=loss_function,
            metrics=[
                tf.keras.metrics.BinaryIoU(name="binary_io_u"),
            ],
        )

    n_gpus = mirrored_strategy.num_replicas_in_sync

    # File sharding policy doesn't seem to work. Set it to Data instead.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    train_dataset = train_dataset.with_options(options)

    # Prepare datasets before training
    replicate_batch_size = batch_size * n_gpus
    train_dataset = (
        train_dataset.batch(replicate_batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .repeat()
    )

    train_steps_per_epoch = n_train_samples // replicate_batch_size

    lr_finder = LRFinder(
        min_lr=1e-7, max_lr=10, img_output_path="data/segmentation/lr_analysis.png"
    )

    # Train model
    _ = model.fit(
        train_dataset,
        epochs=5,
        callbacks=[lr_finder],
        steps_per_epoch=train_steps_per_epoch,
    )


def train_deep_lab_v3_plus_models_with_all_data(
    *,
    out_dir_path: Path,
    dataset_csv: Path,
    input_shape: list[int],
    target_shape: list[int],
    augmentations: Dict[str, Any],
    loss_function: Callable,
    batch_size: int,
    learning_rate: float,
    n_epochs_per_cycle: int,
    n_cycles: int,
    freeze_backbone_model: bool,
):
    out_dir_path.mkdir(exist_ok=True)

    # Open dataset
    dataset_df = pd.read_csv(dataset_csv, dtype={"is_val": bool})

    # Shuffle dataset
    dataset_df = dataset_df.sample(frac=1)

    # Take all data as train, do not validate
    train_dataset = create_dataset(
        dataset_df,
        input_shape=input_shape,
        target_shape=target_shape,
        augmentations=augmentations,
    )

    # Multi-GPU strategy
    mirrored_strategy = tf.distribute.MirroredStrategy()
    n_gpus = mirrored_strategy.num_replicas_in_sync

    # File sharding policy doesn't seem to work. Set it to Data instead.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    train_dataset = train_dataset.with_options(options)

    # Prepare datasets before training
    replicate_batch_size = batch_size * n_gpus
    train_dataset = (
        train_dataset.batch(replicate_batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .repeat()
    )

    train_steps_per_epoch = len(dataset_df) // replicate_batch_size

    cdr = tf.keras.optimizers.schedules.CosineDecayRestarts(
        learning_rate,
        n_epochs_per_cycle * train_steps_per_epoch,
        t_mul=1.0,
        m_mul=1.0,
        alpha=0.0,
    )

    # Define a multi-gpu strategy
    # Model needs to be created and compiled here
    with mirrored_strategy.scope():
        model = DeeplabV3Plus(input_shape, freeze_backbone_model)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(cdr, momentum=0.99, clipnorm=1),
            loss=loss_function,
            metrics=[
                tf.keras.metrics.BinaryIoU(name="binary_io_u"),
            ],
        )

    for idx_cycle in tqdm(np.arange(n_cycles)):
        # Train model
        history = model.fit(
            train_dataset,
            epochs=n_epochs_per_cycle,
            steps_per_epoch=train_steps_per_epoch,
        )

        # Save histories plots
        plt.figure(figsize=(10, 6))
        plot_training_history(history, "", has_val=False)
        plt.savefig(
            out_dir_path / f"deep_lab_v3_plus_plot_{idx_cycle + 1}.png",
            bbox_inches="tight",
        )

        model.save(out_dir_path / f"deep_lab_v3_plus_{idx_cycle + 1}")


def train_deep_lab_v3_plus_models(
    *,
    out_dir_path: Path,
    dataset_csv: Path,
    input_shape: list[int],
    target_shape: list[int],
    augmentations: Dict[str, Any],
    loss_function: Callable,
    batch_size: int,
    learning_rate: float,
    n_epochs_per_cycle: int,
    n_cycles: int,
    do_learning_rate_analysis: bool,
    freeze_backbone_model: bool,
) -> None:
    out_dir_path.mkdir(exist_ok=True)

    # Open dataset
    dataset_df = pd.read_csv(dataset_csv, dtype={"is_val": bool})

    # Shuffle dataset
    dataset_df = dataset_df.sample(frac=1)

    # Split train and validation
    train_df = dataset_df[~dataset_df.is_val]
    val_df = dataset_df[dataset_df.is_val]

    if do_learning_rate_analysis:
        train_dataset = create_dataset(
            train_df,
            input_shape=input_shape,
            target_shape=target_shape,
            augmentations=augmentations,
        )

        learning_rate_analysis(
            train_dataset=train_dataset,
            n_train_samples=len(train_df),
            batch_size=batch_size,
            loss_function=loss_function,
            input_shape=input_shape,
            freeze_backbone_model=freeze_backbone_model,
        )
    else:
        train_dataset = create_dataset(
            train_df,
            input_shape=input_shape,
            target_shape=target_shape,
            augmentations=augmentations,
        )

        # Note: make sure to not augment validation data!
        val_dataset = create_dataset(
            val_df,
            input_shape=input_shape,
            target_shape=target_shape,
        )

        # Multi-GPU strategy
        mirrored_strategy = tf.distribute.MirroredStrategy()
        n_gpus = mirrored_strategy.num_replicas_in_sync

        # File sharding policy doesn't seem to work. Set it to Data instead.
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )

        train_dataset = train_dataset.with_options(options)
        val_dataset = val_dataset.with_options(options)

        # Prepare datasets before training
        replicate_batch_size = batch_size * n_gpus
        train_dataset = (
            train_dataset.batch(replicate_batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .repeat()
        )
        val_dataset = (
            val_dataset.batch(replicate_batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .repeat()
        )

        train_steps_per_epoch = len(train_df) // replicate_batch_size
        val_steps_per_epoch = len(val_df) // replicate_batch_size

        cdr = tf.keras.optimizers.schedules.CosineDecayRestarts(
            learning_rate,
            n_epochs_per_cycle * train_steps_per_epoch,
            t_mul=1.0,
            m_mul=1.0,
            alpha=0.0,
        )

        # Define a multi-gpu strategy
        # Model needs to be created and compiled here
        with mirrored_strategy.scope():
            model = DeeplabV3Plus(input_shape, freeze_backbone_model)
            model.compile(
                optimizer=tf.keras.optimizers.SGD(cdr, momentum=0.99, clipnorm=1),
                loss=loss_function,
                metrics=[
                    tf.keras.metrics.BinaryIoU(name="binary_io_u"),
                ],
            )

        for idx_cycle in tqdm(np.arange(n_cycles)):
            # Train model
            history = model.fit(
                train_dataset,
                epochs=n_epochs_per_cycle,
                validation_data=val_dataset,
                steps_per_epoch=train_steps_per_epoch,
                validation_steps=val_steps_per_epoch,
            )

            # Save histories plots
            plt.figure(figsize=(10, 6))
            plot_training_history(history, "")
            plt.savefig(
                out_dir_path / f"deep_lab_v3_plus_boxplot_{idx_cycle + 1}.png",
                bbox_inches="tight",
            )

            model.save(out_dir_path / f"deep_lab_v3_plus_{idx_cycle + 1}")


def deep_lab_v3_plus(**kwargs) -> None:
    params = yaml.safe_load(open("params.yaml"))

    augmentations = A.Compose(
        [
            A.Flip(p=0.1),
            A.RandomRotate90(p=0.1),
            A.RandomBrightnessContrast(
                brightness_limit=0.05, contrast_limit=0.0, p=0.1
            ),
            A.OneOf(
                [
                    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.CoarseDropout(),
                ],
                p=0.1,
            ),
            A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, value=0, p=0.1),
        ]
    )

    loss_function = get_loss_function(kwargs["loss_function"])

    segmentation_params = yaml.safe_load(open("params.yaml", "r"))["segmentation"]
    out_dir_path = Path(segmentation_params["out"])

    if kwargs["with_all_data"]:
        train_deep_lab_v3_plus_models_with_all_data(
            out_dir_path=out_dir_path,
            dataset_csv=Path("data/preprocessed/dataset/dataset.csv"),
            input_shape=kwargs["input_shape"],
            target_shape=kwargs["target_shape"],
            augmentations=augmentations,
            loss_function=loss_function,
            batch_size=kwargs["batch_size"],
            learning_rate=kwargs["learning_rate"],
            n_epochs_per_cycle=kwargs["n_epochs_per_cycle"],
            n_cycles=kwargs["n_cycles"],
            freeze_backbone_model=kwargs["freeze_backbone_model"],
        )
    else:
        train_deep_lab_v3_plus_models(
            out_dir_path=out_dir_path,
            dataset_csv=Path("data/preprocessed/dataset/dataset.csv"),
            input_shape=kwargs["input_shape"],
            target_shape=kwargs["target_shape"],
            augmentations=augmentations,
            loss_function=loss_function,
            batch_size=kwargs["batch_size"],
            learning_rate=kwargs["learning_rate"],
            n_epochs_per_cycle=kwargs["n_epochs_per_cycle"],
            n_cycles=kwargs["n_cycles"],
            do_learning_rate_analysis=kwargs["do_learning_rate_analysis"],
            freeze_backbone_model=kwargs["freeze_backbone_model"],
        )
