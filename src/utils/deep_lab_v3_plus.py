# Code taken and adapted from:
# https://keras.io/examples/vision/deeplabv3_plus/

# 1) Spatial dropout inspired by:
# https://github.com/smspillaz/seg-reg
# 2) relu replaced by gelu
# 3) num_filters 48 -> 32, (default: 256 -> 128), last convolutions -> 64
# 4) BN -> GN32

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def convolution_block(
    block_input,
    num_filters=128,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
        kernel_regularizer=regularizers.l2(0.01),
    )(block_input)

    x = layers.GroupNormalization(groups=32)(x)

    return tf.keras.activations.gelu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)

    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)

    return output


def DeeplabV3Plus(input_shape, freeze_backbone_model=True, dropout=0.1):
    model_input = keras.Input(shape=input_shape)

    image_size = input_shape[0]  # We assume it's a square

    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )

    if freeze_backbone_model:
        for layer in resnet50.layers:
            layer.trainable = False

            if layer.name == "conv4_block1_1_relu":
                break

    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)
    x = layers.SpatialDropout2D(dropout)(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)

    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=32, kernel_size=1)
    input_b = layers.SpatialDropout2D(dropout)(input_b)

    x = layers.Concatenate(axis=-1)([input_a, input_b])

    x = convolution_block(x, num_filters=64)
    x = layers.SpatialDropout2D(dropout)(x)
    x = convolution_block(x, num_filters=64)

    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)

    model_output = layers.Conv2D(
        1, kernel_size=(1, 1), padding="same", activation="sigmoid"
    )(x)

    return keras.Model(inputs=model_input, outputs=model_output)
