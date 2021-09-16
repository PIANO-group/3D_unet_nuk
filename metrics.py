import tensorflow as tf
import numpy as np


def dice(y_true, y_pred, axis=(1, 2, 3),
         epsilon=0.00001):
    y_true = tf.cast(y_true, tf.float32)
    # to calculate dice with binary values
    y_pred = tf.round(y_pred)

    dice_numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=axis) + epsilon
    # dice_denominator = tf.reduce_sum(tf.square(y_true), axis=axis) + tf.reduce_sum(tf.square(y_pred), axis=axis) + epsilon
    # without squaring:
    dice_denominator = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis) + epsilon
    dice_coefficient = dice_numerator / dice_denominator

    return dice_coefficient


def dice_single(y_true, y_pred, epsilon=0.00001):
    dice_numerator = 2 * np.sum(y_true * y_pred) + epsilon
    dice_denominator = np.sum(y_true) + np.sum(y_pred) + epsilon
    dice_coefficient = dice_numerator / dice_denominator

    return dice_coefficient