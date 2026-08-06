
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, precision_recall_curve, auc, f1_score, matthews_corrcoef
from sklearn.metrics import average_precision_score
from skimage.morphology import skeletonize
import numpy as np
import sys

import numpy as np
from typing import List, Union
IGNORE_INDEX = 255

def _cl_score(v: np.ndarray, s: np.ndarray) -> float:
    """
    Compute the centerline-based precision or sensitivity score used in CLDice.

    This function calculates the overlap of a binary volume `v` with a binary
    skeleton `s`. It is defined as the sum of `v * s` divided by the sum of `s`.

    Args:
        v (np.ndarray): Binary mask (volume or 2D/3D array)
        s (np.ndarray): Skeletonized binary mask of the reference volume

    Returns:
        float: Overlap score (between 0 and 1), NaN if skeleton is empty
    """
    s_sum = np.sum(s)
    if s_sum == 0:
        return np.nan
    return np.sum(v * s) / s_sum


def multiclass_cldice(
        actual: np.ndarray,
        predicted: np.ndarray,
        n_classes: int,
        ignore_index: int = IGNORE_INDEX
) -> List[Union[float, None]]:

    cldsc_scores = []

    # valid pixels
    valid = actual != ignore_index

    for c in range(1, n_classes):  # exclude background
        v_l = (actual == c) & valid
        v_p = (predicted == c) & valid

        if not np.any(v_l):
            cldsc_scores.append(np.nan)
            continue

        skel_l = skeletonize(v_l)
        skel_p = skeletonize(v_p)

        tprec = _cl_score(v_p, skel_l)
        tsens = _cl_score(v_l, skel_p)

        if np.isnan(tprec) or np.isnan(tsens):
            cldsc_scores.append(np.nan)
        else:
            cldsc_scores.append(100.0 * (2 * tprec * tsens / (tprec + tsens)))

    return cldsc_scores

def masked_multiclass_cldice(actual: np.ndarray,
                             predicted: np.ndarray,
                             mask: np.ndarray,
                             n_classes: int,
                             ignore_index: int = IGNORE_INDEX) -> list:
    """
    Compute multiclass clDice restricted to a mask by temporarily setting
    pixels outside the mask to the ignore_index, then calling multiclass_cldice.
    """
    # Make copies to avoid modifying original arrays
    actual_masked = actual.copy()
    predicted_masked = predicted.copy()

    # Set pixels outside mask to ignore_index
    outside = ~mask
    actual_masked[outside] = ignore_index
    predicted_masked[outside] = ignore_index

    #Call original multiclass_cldice
    return multiclass_cldice(actual_masked, predicted_masked, n_classes, ignore_index=ignore_index)


def bin_dice(actual: np.ndarray, 
             predicted: np.ndarray,
             ignore_index: int = IGNORE_INDEX) -> float:
    """
    Compute the Dice Similarity Coefficient (DSC) between two binary masks.
    
    DSC = 2 * |A ∩ B| / (|A| + |B|), where A and B are the sets of pixels in 
    the actual and predicted masks, respectively.

    Args:
        actual (np.ndarray): Ground truth binary mask
        predicted (np.ndarray): Predicted binary mask

    Returns:
        float: Dice Similarity Coefficient (between 0 and 1)
    """
    actual = np.asarray(actual).astype(bool)
    predicted = np.asarray(predicted).astype(bool)

    if actual.shape != predicted.shape:
        raise ValueError(f'Shape mismatch: actual {actual.shape} vs predicted {predicted.shape}')

    valid = actual != ignore_index
    actual = actual[valid].astype(bool)
    predicted = predicted[valid].astype(bool)

    im_sum = actual.sum() + predicted.sum()
    if im_sum == 0:  # both masks empty, perfect match
        return 1.0

    intersection = np.logical_and(actual, predicted)
    return 2.0 * intersection.sum() / im_sum


def multiclass_dice(
    actual: np.ndarray, 
    predicted: np.ndarray, 
    n_classes: int, 
    exclude_background: bool = True,
    ignore_index: int = IGNORE_INDEX) -> List[Union[float, None]]:
    """
    Compute the Dice Similarity Coefficient (DSC) for each class in a multi-class segmentation.
    
    Args:
        actual (np.ndarray): Ground truth segmentation mask with integer class labels
        predicted (np.ndarray): Predicted segmentation mask with integer class labels
        n_classes (int): Total number of classes including background
        exclude_background (bool): Whether to exclude the background class (assumed to be 0)

    Returns:
        List[float or None]: DSC score for each class (percentage), NaN if class absent
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    classes = list(range(n_classes))
    if exclude_background:
        classes = classes[1:]

    # consider classes present in either ground truth or prediction
    present_classes = np.unique(np.concatenate([np.unique(actual), np.unique(predicted)]))

    dsc_scores = []
    for c in classes:
        if c in present_classes:
            dsc_scores.append(100.0 * bin_dice(actual == c, predicted == c, ignore_index=ignore_index))
        else:
            dsc_scores.append(np.nan)

    return dsc_scores

def masked_multiclass_dice(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray, **kwargs):

    """
    Helper function to compute multiclass dice only within a given mask.
    Args:
        actual (np.ndarray): Ground truth segmentation mask with integer class labels
        predicted (np.ndarray): Predicted segmentation mask with integer class labels
        mask (np.ndarray): Boolean mask indicating valid regions for computation
        **kwargs: Additional arguments for multiclass_dice function
    Returns:
        List[float or None]: DSC score for each class (percentage), NaN if class absent
    """

    actual_m = actual.copy()
    predicted_m = predicted.copy()

    actual_m[~mask] = IGNORE_INDEX
    predicted_m[~mask] = IGNORE_INDEX

    return multiclass_dice(actual_m, predicted_m, **kwargs)


def bin_recall(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = actual.astype(bool)
    predicted = predicted.astype(bool)

    tp = np.logical_and(actual, predicted).sum()
    fn = np.logical_and(actual, ~predicted).sum()

    if tp + fn == 0:
        return np.nan  # no GT pixels in this ROI

    return tp / (tp + fn)

def bin_tnr(actual: np.ndarray, 
            predicted: np.ndarray, 
            ignore_index: int = 255,
            mask: np.ndarray = None,
            target_label: int = 0) -> float:
    """
    Compute True Negative Rate (specificity) for a given target label in a masked region.

    TN / (TN + FP) where TN = target_label correctly predicted as target_label,
    FP = target_label incorrectly predicted as other.

    Args:
        actual: ground truth label mask
        predicted: predicted label mask
        ignore_index: label to ignore
        mask: optional boolean mask to restrict computation
        target_label: the label considered as "negative" (default 0, background)

    Returns:
        float: TNR in [0,1]
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # Mask out ignore_index
    valid = actual != ignore_index
    actual = actual[valid]
    predicted = predicted[valid]

    # Apply zone mask if provided
    if mask is not None:
        mask = mask[valid]
        actual = actual[mask]
        predicted = predicted[mask]

    # True negative = target_label correctly predicted
    tn = np.logical_and(actual == target_label, predicted == target_label).sum()
    fp = np.logical_and(actual == target_label, predicted != target_label).sum()

    if tn + fp == 0:
        return np.nan  # no negative pixels in zone
    return tn / (tn + fp)



