import numpy as np

def entropy_prob(probs, classes=None):
    """
    Compute voxel-wise predictive entropy from multiple predictions.

    Args:
        probs (np.ndarray): Array of shape [n_samples, n_classes, x, y, z] 
                            containing softmax predictions for each MC sample.
        classes (int, optional): Number of classes for normalised entropy computation. 
                                 Defaults to None (non-normalised).

    Returns:
        np.ndarray: Entropy map of shape [n_classes, x, y, z] or [x, y, z] depending on usage.

    ----------------------------------------------------------------------------
    # entropy as described in mukhoti2018 thesis
    # assumed probs is an array of shape [n,m,x,y,z]
    # with n number of monte carlo samples
    # with m number of predicted classes
    # if 'classes' is specified, a normalised entropy is computed.
    ----------------------------------------------------------------------------
    """
    eps = 1e-12  # small deviation as log(0) does not exist
    p = np.mean(probs, axis=0)
    
    if classes:
        plogp = (p * np.log10(p + eps)) / np.log10(classes)  # normalised entropy
    else:
        plogp = p * np.log10(p + eps)  # non-normalised entropy
    
    entropy = -np.sum(plogp, axis=0)
    return entropy


def mutual_information_prob(probs, classes):
    """
    Compute voxel-wise mutual information from multiple predictions.

    Args:
        probs (np.ndarray): Array of shape [n_samples, n_classes, x, y, z] 
                            containing softmax predictions.
        classes (int): Number of predicted classes for normalised MI computation.
                            Defaults to None (non-normalised).

    Returns:
        np.ndarray: Mutual information map of shape [x, y, z].

    ----------------------------------------------------------------------------
    # mutual information as described in mukhoti2018 thesis
    # assumed probs is an array of shape [n,m,x,y,z]
    # if 'classes' is specified, a normalised mutual information is computed
    ----------------------------------------------------------------------------
    """
    entropy = entropy_prob(probs, classes)
    eps = 1e-12 # small deviation as log(0) does not exist
    p = probs + eps
    
    if classes:
        plogp = (p * np.log10(p + eps)) / np.log10(classes)
    else:
        plogp = p * np.log10(p + eps)
    
    exp_entropies = np.mean(np.sum(plogp, axis=1), axis=0)
    mutual_information = np.add(entropy, exp_entropies)
    return mutual_information


def entropy_classwise_prob(probs, classes):
    """
    Compute class-wise entropy maps comparing each ROI vs all other ROIs.

    Args:
        probs (np.ndarray): Array of shape [n_samples, n_classes, x, y, z] 
                            containing softmax predictions.
        classes (int): Number of predicted classes (ROIs).

    Returns:
        np.ndarray: Class-wise entropy map of shape [classes, x, y, z].

    ----------------------------------------------------------------------------
    # class-wise entropy as described in camarasa2021
    # assumed probs is an array of shape [n,m,x,y,z]
    ----------------------------------------------------------------------------
    """
    eps = 1e-12 # small deviation as log(0) does not exist
    classwise_entropy = np.zeros(np.shape(probs)[-4:])  # [classes, x, y, z]

    for roi in range(classes):
        # ROI to be evaluated
        probs_evaluating_roi = np.mean(probs[:, roi, :, :, :], axis=0)

        # Other ROIs
        probs_other_rois_individual = np.delete(probs, roi, axis=1)
        probs_sum_other_rois = np.sum(probs_other_rois_individual, axis=1)
        probs_other_rois = np.mean(probs_sum_other_rois, axis=0)

        # 2-channel volume: ROI vs non-ROI
        p = np.stack((probs_evaluating_roi, probs_other_rois), axis=0)

        plogp = (p * np.log10(p + eps)) / np.log10(2)  # normalised entropy

        classwise_entropy[roi, :, :, :] = -np.sum(plogp, axis=0)

    return classwise_entropy
    
def mutual_information_classwise_prob(probs, classes):
    """
    Compute class-wise voxel-wise mutual information from multiple Monte Carlo predictions.

    Args:
        probs (np.ndarray): Array of shape [n_samples, n_classes, x, y, z] 
                            containing softmax predictions.
        classes (int): Number of predicted classes for normalised MI computation.

    Returns:
        np.ndarray: Mutual information map of shape [classes, x, y, z].

    ----------------------------------------------------------------------------
    # mutual information as described in mukhoti2018 thesis
    # assumed probs is an array of shape [n,m,x,y,z]
    ----------------------------------------------------------------------------
    """
    
    eps = 1e-12 # small deviation as log(0) does not exist
    
    classwise_mutual_information = np.zeros(np.shape(probs)[-4:]) # [classes, x, y, z]
    
    for roi in range(classes):
        # ROI to be evaluated
        probs_evaluating_roi = probs[:, roi, : , :, :] # [n_samples, x , y, z]
        
        # Other ROIs
        probs_other_rois_individual = np.delete(probs, roi, axis=1)
        probs_sum_other_rois = np.sum(probs_other_rois_individual, axis=1) # [n_samples, x, y, z]
        
        # Stack to create a 2-channel volme: ROI vs non-ROI
        p = np.stack((probs_evaluating_roi, probs_sum_other_rois), axis=1) # [n_samples, x, y, z]
        
        mean_p = np.mean(p, axis=0) #[x,y,z]
        
        # Compute entropy of the mean probabilties for this class vs others
        plogp = (mean_p * np.log10(mean_p + eps)) / np.log10(2)  # normalised entropy
        
        H = -np.sum(plogp, axis=0) # [x, y, z]
        
        # Compute expected entropy for this class vs others
        plogp_samples = (p * np.log10(p + eps)) / np.log10(2)
        
        exp_entropies = -np.mean(np.sum(plogp_samples, axis=1), axis=0) # [x,y,z]
        
        classwise_mutual_information[roi, :, :, :] = H + exp_entropies
        
    return classwise_mutual_information


def variance_prob(probs):
    """
    Compute voxel-wise variance across Monte Carlo predictions.

    Args:
        probs (np.ndarray): Array of shape [n_samples, n_classes, x, y, z] 
                            containing softmax predictions.

    Returns:
        np.ndarray: Variance map of shape [x, y, z].

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    """
    variance = np.var(np.array(probs), axis=0)    #[n_classes,x,y,z]
    
    return variance
    
    
def variance_classwise_prob(probs):
    """
    Compute voxel-wise variance across Monte Carlo predictions.

    Args:
        probs (np.ndarray): Array of shape [n_samples, n_classes, x, y, z] 
                            containing softmax predictions.

    Returns:
        np.ndarray: Variance map of shape [n_classes, x, y, z].

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    """

    return np.var(np.array(probs), axis=0)
