from tensorflow.keras import backend as K
import tensorflow as tf


# Code taken and slightly adapted from:
# https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch
def focal_tversky(gamma=1.0, alpha=0.75, beta=0.25):
    def focal_tversky_loss(y_true, y_pred, smooth=1e-6):
        if y_pred.shape[-1] <= 1:
            y_pred = tf.keras.activations.sigmoid(y_pred)
        elif y_pred.shape[-1] >= 2:
            y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
            y_true = K.squeeze(y_true, 3)
            y_true = tf.cast(y_true, "int32")
            y_true = tf.one_hot(y_true, num_class, axis=-1)

        y_true = K.cast(y_true, "float32")
        y_pred = K.cast(y_pred, "float32")

        # flatten label and prediction tensors
        inputs = K.flatten(y_pred)
        targets = K.flatten(y_true)

        # True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1 - targets) * inputs))
        FN = K.sum((targets * (1 - inputs)))

        tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return K.pow((1 - tversky), gamma)

    return focal_tversky_loss


# Code taken and slightly adapted from:
# https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch
def weighted_bce_and_dice(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
        y_true, pool_size=(11, 11), strides=(1, 1), padding="same", pool_mode="avg"
    )
    border = K.cast(K.greater(averaged_mask, 0.005), "float32") * K.cast(
        K.less(averaged_mask, 0.995), "float32"
    )
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= w0 / w1

    return weighted_dice_loss(y_true, y_pred, weight) + weighted_bce_loss(
        y_true, y_pred, weight
    )


def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    logit_y_pred = K.log(y_pred / (1.0 - y_pred))
    # logit_y_pred = y_pred

    loss = (1.0 - y_true) * logit_y_pred + (1.0 + (weight - 1.0) * y_true) * (
        K.log(1.0 + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.0)
    )

    return K.sum(loss) / K.sum(weight)


def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.0
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = m1 * m2
    score = (2.0 * K.sum(w * intersection) + smooth) / (
        K.sum(w * (m1**2)) + K.sum(w * (m2**2)) + smooth
    )  # Uptill here is Dice Loss with squared

    return 1.0 - K.sum(score)  # Soft Dice Loss
