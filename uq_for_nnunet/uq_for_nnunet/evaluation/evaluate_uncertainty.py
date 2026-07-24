import argparse
import os
import re
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_file_mappings(UQ_folder, pred_folder=None, gt_folder=None):
    """Collect files in folders and map them to patient IDs.
    Args:
        UQ_folder (str): Folder containing uncertainty maps.
        pred_folder (str, optional): Folder containing predicted segmentations. 
        gt_folder (str, optional): Folder containing ground truth segmentations.
    """
    # Uncertainty maps
    uq_files = {}
    for f in os.listdir(UQ_folder):
        if f.endswith(".nii") or f.endswith(".nii.gz"):
            # To do: fix consistent naming of uncertainty maps
            #m = re.match(r"uncertainty_map_[^_]+_(.+)\.nii(\.gz)?$", f)
            #m = re.match(r"uncertainty_map_entropy_(\d{3}).nii(\.gz)?$", f) 
            m = re.match(r"uncertainty_values_multiclass_entropy_mutual_information_classwise_variance_(\d{3})_entropy\.nii(\.gz)?$", f)
            if m:
                print(m)
                pid = m.group(1)
                uq_files[pid] = os.path.join(UQ_folder, f)

    # Predicted segmentations
    pred_files = {}
    for f in os.listdir(pred_folder):
        if f.endswith(".nii") or f.endswith(".nii.gz"):
            m = re.match(r"combined_segmentation_patient_(.+)\.nii(\.gz)?$", f)
            if m:
                pid = m.group(1)
                pred_files[pid] = os.path.join(pred_folder, f)

    # Ground truth (search for patient ID in filename)
    gt_files = {}
    if gt_folder:
        gt_list = [f for f in os.listdir(gt_folder) if f.endswith(".nii") or f.endswith(".nii.gz")]
        for pid in pred_files:  # iterate over known patient IDs
            for f in gt_list:
                if pid in f:
                    gt_files[pid] = os.path.join(gt_folder, f)
                    break

    return uq_files, pred_files, gt_files

def compute_uncertainty_statistics(uncty_arr, thresholds=None):
    """
    Compute summary statistics for an uncertainty array.

    Args:
        uncty_arr (np.ndarray): 3D array of uncertainty values.
        thresholds (list or np.ndarray, optional): List of thresholds to compute additional statistics.
        
    """
    uncty_values = uncty_arr.flatten()
    values_above_median = uncty_values[uncty_values > np.median(uncty_values)]
    counts, bins = np.histogram(uncty_arr.flatten())
    stats = {
        "mean": float(np.mean(uncty_values)),
        "median": float(np.median(uncty_values)),
        "std": float(np.std(uncty_values)),
        "mean_above_median": float(np.mean(values_above_median)),
        "std_above_median": float(np.std(values_above_median)),
        "min": float(np.min(uncty_values)),
        "max": float(np.max(uncty_values)),
        "p25": float(np.percentile(uncty_values, 25)),
        "p75": float(np.percentile(uncty_values, 75)),
        "p95": float(np.percentile(uncty_values, 95)),
        "hist_bins": bins,
        "hist_counts": counts
    }

    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.05)

    for t in thresholds:
        # Proportion below threshold
        proportion = float(np.mean(uncty_values < t))
        stats[f"proportion_uncertain_voxels_{t:.2f}"] = proportion

        # Mean above threshold
        above_vals = uncty_values[uncty_values > t]
        mean_above_threshold = float(np.mean(above_vals)) if above_vals.size > 0 else np.nan
        stats[f"mean_above_threshold_{t:.2f}"] = mean_above_threshold
        
        # number of voxels above threshold
        stats[f"number_of_voxels_above_threshold_{t:.2f}"] = len(above_vals)
        
        # relative number of voxels above threshold
        stats[f"relative_number_of_voxels_above_threshold_{t:.2f}"] = len(above_vals)/len(uncty_values)
    return stats

def load_arrays(uq_file, pred_file=None, gt_file=None, transformer=None):
    """
    Load NIfTI files and compute flattened accuracy and uncertainty arrays.
    If ground truth or predictions are not provided, returns None for accuracy.

    Args:
        uq_file (str): Path to the uncertainty map NIfTI file.
        pred_file (str, optional): Path to the predicted segmentation NIfTI file.
        gt_file (str, optional): Path to the ground truth segmentation NIfTI file.
        transformer (object, optional): An object with a method `transform_uncertainty_map` to transform the uncertainty map.
    
    Returns:
        uncty_flat (np.ndarray): Flattened uncertainty values.
        acc_mask (np.ndarray or None): Flattened accuracy mask (1 for correct, 0 for incorrect), or None if pred or gt is not provided.
    """

    uq = nib.load(uq_file).get_fdata()
    if transformer:
        uq = transformer.transform_uncertainty_map(uq)
        
    # Load predictions and ground truth if provided to compute accuracy map
    pred = nib.load(pred_file).get_fdata() if pred_file else None
    gt = nib.load(gt_file).get_fdata() if gt_file else None

    if pred is not None and gt is not None:
        acc_mask = (pred == gt).astype(np.float32).flatten()
    else:
        acc_mask = None

    return uq.flatten(), acc_mask
    
def load_file(uncertainty_map):
    return nib.load(uncertainty_map).get_fdata()

def main():
    parser = argparse.ArgumentParser(description="Compute uncertainty statistics for folder of files")
    parser.add_argument("--UQ_folder", type=str, required=True, help="Folder containing uncertainty maps")
    parser.add_argument("--output_excel", type=str, required=True, help="Excel file to save results")
    parser.add_argument("--patients", type=str, nargs="+", default=None,
                        help="Optional list of patient IDs to process. If not provided, all patients are processed")
    args = parser.parse_args()
  
    uq_files, pred_files, gt_files = get_file_mappings(args.UQ_folder)

    print(uq_files)
    patients = sorted(set(uq_files.keys()))  # Use keys instead of the whole dictionary
    print(f"Patients: {patients}")
        
    if args.patients:
        patients = [pid for pid in patients if pid in args.patients]

    if not patients:
        print("No patients found matching the criteria.")
        return

    results = []
    pbar = tqdm(patients, desc="Processing patients")
    
    for pid in pbar:
        pbar.set_description(f"Processing {pid}")
        
        uncertainty_array = load_file(uq_files[pid])
        stats = compute_uncertainty_statistics(uncertainty_array)
        stats["Patient_ID"] = pid
        stats["UQ_file"] = os.path.basename(uq_files[pid])
        results.append(stats)

    df = pd.DataFrame(results)

    sheet_name = 'Results uncertainty values'
    if os.path.exists(args.output_excel):
        with pd.ExcelWriter(args.output_excel, mode='a', if_sheet_exists='new', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        df.to_excel(args.output_excel, sheet_name=sheet_name, index=False)

    print(f"Results saved/updated at {args.output_excel}")

if __name__ == "__main__":
    main()
