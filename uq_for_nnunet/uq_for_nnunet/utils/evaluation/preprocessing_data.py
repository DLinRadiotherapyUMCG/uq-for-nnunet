import numpy as np

def crop_around_segmentation(gt_arr=None, pred_arr=None, uncty_arr=None, buffer=10):
    """
    Crop arrays around the union of GT and predicted segmentations with a spatial buffer.

    Args:
        gt_arr : np.ndarray or None
            Ground truth 3D array (binary or labeled).
        pred_arr : np.ndarray or None
            Predicted 3D array (binary or labeled).
        uncty_arr : np.ndarray, optional
            3D uncertainty map array to crop alongside.
        buffer : int, default=10
            Number of voxels to include as padding around the segmentation.

    Returns:
        cropped_gt : np.ndarray or None
            Cropped ground truth segmentation.
        cropped_pred : np.ndarray or None
            Cropped predicted segmentation.
        cropped_uncty : np.ndarray or None
            Cropped uncertainty map.
        crop_bounds : tuple of slices
            The slicing indices used for cropping (for later use if needed).
    """
    
    if gt_arr is None and pred_arr is None:
        raise ValueError("At least one of gt_arr or pred_arr must be provided.")

    # Determine the reference mask for cropping i.e. where ground truth or prediction is present
    if gt_arr is not None and pred_arr is not None:
        reference_mask = (gt_arr > 0) | (pred_arr > 0)
        shape = gt_arr.shape
    else:
        arr = gt_arr if gt_arr is not None else pred_arr
        reference_mask = arr > 0
        shape = arr.shape

    # Ensure uncertainty array has the same shape
    assert uncty_arr is None or uncty_arr.shape == shape, \
        f"GT and/or prediction, and uncertainty must all have same shape, got " \
        f"{'None' if gt_arr is None else gt_arr.shape}, " \
        f"{'None' if pred_arr is None else pred_arr.shape}, " \
        f"{'None' if uncty_arr is None else uncty_arr.shape}"

    # Get bounding box with buffer
    coords = np.array(np.nonzero(reference_mask))
    min_coords = np.maximum(coords.min(axis=1) - buffer, 0)
    max_coords = np.minimum(coords.max(axis=1) + buffer + 1, np.array(shape))

    crop_slices = tuple(slice(int(minc), int(maxc)) for minc, maxc in zip(min_coords, max_coords))

    # Apply cropping if arrays are present
    cropped_gt = gt_arr[crop_slices] if gt_arr is not None else None
    cropped_pred = pred_arr[crop_slices] if pred_arr is not None else None
    cropped_uncty = uncty_arr[crop_slices] if uncty_arr is not None else None

    return cropped_gt, cropped_pred, cropped_uncty, crop_slices
