import numpy as np

def compute_ECE(acc, prob, bins=20):
    """Compute reliability diagram and ECE.
    
    Args:
        acc (np.ndarray): Array of any shape [often [x,y,z]] 
                        Containing accuracy values (0 or 1).
        prob (np.ndarray): Array of any shape [often [x,y,z]] 
                        Containing predicted probabilities.
        bins (int): Number of bins for reliability diagram. Defaults to 20.
    
    Returns:
        reliability_acc (np.ndarray): Array of shape [bins] containing average accuracy per bin.
        reliability_prob (np.ndarray): Array of shape [bins] containing average predicted probability per bin.
        ECE (float): Expected Calibration Error.
    """

    y_acc = acc.flatten()
    y_prob = prob.flatten()
    bin_values = np.linspace(np.min(y_prob), np.max(y_prob), num=bins)

    reliability_acc = []
    reliability_prob = []
    ECE = 0

    for idx in range(len(bin_values) - 1):
        locs = np.where((y_prob > bin_values[idx]) & (y_prob <= bin_values[idx + 1]))[0]
        if len(locs) == 0:
            continue
        avg_acc = np.mean(y_acc[locs])
        avg_prob = np.mean(y_prob[locs])

        reliability_acc.append(avg_acc)
        reliability_prob.append(avg_prob)
        ECE += len(locs) / len(y_acc) * abs(avg_prob - avg_acc)

    return np.array(reliability_acc), np.array(reliability_prob), ECE
