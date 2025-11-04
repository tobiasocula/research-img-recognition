import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
def resize_with_tf(tensors):
    return tf.image.resize(
        tensors[0],
        [tf.shape(tensors[1])[1], tf.shape(tensors[1])[2]],
        method=tf.image.ResizeMethod.BILINEAR,
    )

@register_keras_serializable()
def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """Dice coefficient (overlap) between predicted and target masks."""
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denominator = tf.reduce_sum(y_true_f + y_pred_f, axis=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(dice)

@register_keras_serializable()
def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Dice loss encourages overlap between predicted and true masks."""
    return 1.0 - dice_coef(y_true, y_pred)

@register_keras_serializable()
def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Combined Binary Cross-Entropy + Dice loss for stable training."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)

@register_keras_serializable()
def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """Intersection-over-Union metric (Jaccard index)."""
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    total = tf.reduce_sum(y_true_f + y_pred_f, axis=1)
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)