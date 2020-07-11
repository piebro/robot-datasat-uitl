import numpy as np
import scipy
import tensorflow as tf
from find_robot_position_with_keypoints import get_pixel_with_label

def mean_center_distance_loss(gt_label_mask, pred_label_mask):
    labels_in_predict = np.unique(pred_label_mask)
    losses = []
    for label in labels_in_predict:
        if label == 0:
            continue
        gt_pixel = get_pixel_with_label(gt_label_mask, label, False)
        pred_pixel = get_pixel_with_label(pred_label_mask, label, False)
        
        gt_mean = np.mean(gt_pixel, axis=0)
        pred_mean = np.mean(pred_pixel, axis=0)
        loss = np.linalg.norm(gt_mean-pred_mean)
        losses.append(loss)
    
    if len(losses)>0:
        final_loss = np.average(losses)
    else:
        final_loss = 100000
    return final_loss
        
def average_min_distance_loss(gt_label_mask, pred_label_mask):
    labels_in_predict = np.unique(pred_label_mask)
    losses = []
    for label in labels_in_predict:
        if label == 0:
            continue
        gt_pixel = get_pixel_with_label(gt_label_mask, label, False)
        pred_pixel = get_pixel_with_label(pred_label_mask, label, False)
        if len(gt_pixel)>0 and len(pred_pixel)>0:
            dist = scipy.spatial.distance.cdist(pred_pixel, gt_pixel, metric="euclidean")
            min_distances = np.min(dist, axis=1)
            avg_dist = np.average(min_distances)
            loss = avg_dist
            losses.append(loss)
        
    if len(losses)>0:
        final_loss = np.average(losses)
    else:
        final_loss = 100000
    return final_loss

def convert_to_logits(y_pred):
    # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    return tf.math.log(y_pred / (1 - y_pred))

def weighted_cross_entropy_loss(beta, y_true, y_pred):
    y_pred = convert_to_logits(np.float32(y_pred))
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=np.float32(y_true), pos_weight=beta)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss).numpy()

def balanced_cross_entropy_loss(beta, y_true, y_pred):
    y_pred = convert_to_logits(np.float32(y_pred))
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=np.float32(y_true), pos_weight=pos_weight)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss * (1 - beta)).numpy()

def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    
    return (tf.math.log1p(tf.math.exp(-tf.math.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    y_pred = tf.clip_by_value(np.float32(y_pred), tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))

    loss = focal_loss_with_logits(logits=logits, targets=np.float32(y_true), alpha=alpha, gamma=gamma, y_pred=y_pred)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss).numpy()


def dice_loss(y_true, y_pred):
    y_true = np.float32(y_true)
    y_pred = np.float32(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return (1 - numerator / denominator).numpy()

def tversky_loss(beta, y_true, y_pred):
    y_true = np.float32(y_true)
    y_pred = np.float32(y_pred)
    numerator = tf.reduce_sum(y_true * y_pred, axis=None)
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

    return (1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=None) + 1)).numpy()

weighted_cross_entropy_loss_1 = lambda y_true, y_pred: weighted_cross_entropy_loss(1, y_true, y_pred)
weighted_cross_entropy_loss_2 = lambda y_true, y_pred: weighted_cross_entropy_loss(2, y_true, y_pred)

tversky_loss_0_5 = lambda y_true, y_pred: tversky_loss(0.5, y_true, y_pred)
tversky_loss_0_8 = lambda y_true, y_pred: tversky_loss(0.8, y_true, y_pred)
tversky_loss_0_95 = lambda y_true, y_pred: tversky_loss(0.95, y_true, y_pred)
tversky_loss_1_2 = lambda y_true, y_pred: tversky_loss(1.2, y_true, y_pred)
tversky_loss_2_0 = lambda y_true, y_pred: tversky_loss(2.0, y_true, y_pred)

# losses_str_to_func = {
#     "average_min_distance_loss": average_min_distance_loss,
#     "mean_center_distance_loss": mean_center_distance_loss,
#     "weighted_cross_entropy_loss_1": lambda y_true, y_pred: weighted_cross_entropy_loss(1, y_true, y_pred),
#     "balanced_cross_entropy_loss_2": lambda y_true, y_pred: balanced_cross_entropy_loss(2, y_true, y_pred),
#     "focal_loss": focal_loss,
#     "dice_loss": dice_loss,
#     "tversky_loss_0_5": lambda y_true, y_pred: tversky_loss(0.5, y_true, y_pred),
#     "tversky_loss_0_8": lambda y_true, y_pred: tversky_loss(0.8, y_true, y_pred),
#     "tversky_loss_0_95": lambda y_true, y_pred: tversky_loss(0.95, y_true, y_pred),
#     "tversky_loss_1_2": lambda y_true, y_pred: tversky_loss(1.2, y_true, y_pred),
#     "tversky_loss_2_0": lambda y_true, y_pred: tversky_loss(2, y_true, y_pred),
# }