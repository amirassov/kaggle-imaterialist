import numpy as np
from functools import partial


def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def calc_iou(y_true, y_prediction):
    y_true, y_prediction = map(partial(np.expand_dims, axis=0), (y_true, y_prediction))

    true_objects = len(np.unique(y_true))
    pred_objects = len(np.unique(y_prediction))

    # Compute intersection between all objects
    intersection = np.histogram2d(y_true.flatten(), y_prediction.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(y_true, bins=true_objects)[0]
    area_pred = np.histogram(y_prediction, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union
    return iou


def calc_score_per_class(y_true, y_prediction):
    iou = calc_iou(y_true, y_prediction)

    # Loop over IoU thresholds
    precisions = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        precisions.append(p)
    return np.mean(precisions)
