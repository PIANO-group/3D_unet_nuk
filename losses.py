import tensorflow as tf
from tensorflow.keras.losses import Loss


def dice_loss(y_true,y_pred, loss_type='sorensen', smooth=1.):

    y_true_f = tf.cast(tf.reshape(y_true,[-1]),tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred,[-1]),tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (1-(2. * intersection + smooth) / (union + smooth))


def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3),
                   epsilon=0.00001):
    """
    Compute mean soft dice loss over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator in formula for dice loss.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_loss (float): computed value of dice loss.
    """

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    print(y_true.dtype)
    print(y_pred.dtype)

    y_true = tf.cast(y_true, tf.float32)
    print(y_true.dtype)
    dice_numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = tf.reduce_sum(tf.square(y_true), axis=axis) + tf.reduce_sum(tf.square(y_pred), axis=axis) + epsilon
    dice_loss = 1 - tf.reduce_mean(dice_numerator / dice_denominator)

    ### END CODE HERE ###

    return dice_loss


def iou_loss (y_true, y_pred, axis = (1,2,3), smooth=100):
    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=axis)
    sum_ = tf.reduce_sum(tf.abs(y_true) + tf.abs(y_pred), axis=axis)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (1 - jac) * smooth


def meanIoU_loss(y_true, y_pred, smooth=0):
    y_true_f = tf.cast(tf.reshape(y_true,[-1]),tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred,[-1]),tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f) - tf.reduce_sum(y_true_f * y_pred_f)


    return (1-(intersection + smooth) / (union + smooth))
