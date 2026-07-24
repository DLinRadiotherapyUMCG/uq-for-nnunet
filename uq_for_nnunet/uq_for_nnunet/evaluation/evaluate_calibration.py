import argparse
import os
import re
import nibabel as nib
import numpy as np
import pandas as pd
import sklearn 

def get_file_mappings(UQ_folder, pred_folder, gt_folder=None):
    """Collect files in folders and map them to patient IDs."""
    # Uncertainty maps
    uq_files = {}
    for f in os.listdir(UQ_folder):
        if f.endswith(".nii") or f.endswith(".nii.gz"):
            m = re.match(r"uncertainty_map_[^_]+_(.+)\.nii(\.gz)?$", f)
            m = re.match(r"uncertainty_values_multiclass_entropy_mutual_information_classwise_variance_[^_]+_(.+)\entorpy.nii(\.gz)?$", f)
            if m:
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

def bounding_box_3d_with_padding(volume, padding=5):
    """
    volume: 3D numpy array of 0s and 1s
    padding: The number of voxels to pad around the bounding box
    returns: (min_z, max_z, min_y, max_y, min_x, max_x) with padding
    """
    # Find indices where volume exists
    coords = np.argwhere(volume)

    if coords.size == 0:
        return None  # No '1's in the volume

    # Get min and max along each dimension
    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0)

    # Apply padding (subtract padding for min, add padding for max)
    min_z = max(min_z - padding, 0)
    min_y = max(min_y - padding, 0)
    min_x = max(min_x - padding, 0)
    
    max_z = min(max_z + padding, volume.shape[0] - 1)
    max_y = min(max_y + padding, volume.shape[1] - 1)
    max_x = min(max_x + padding, volume.shape[2] - 1)

    return (min_z, max_z, min_y, max_y, min_x, max_x)
    
def extract_accuracy_uncertainty_arrays(gt_file, pred_file, uq_file, metric='accuracy'):
    """
    Load NIfTI files and compute flattened accuracy and uncertainty arrays.
    If no ground truth is provided, assumes perfect accuracy mask of ones.
    """
    pred = nib.load(pred_file).get_fdata()
    uq = nib.load(uq_file).get_fdata()
    gt = nib.load(gt_file).get_fdata() if gt_file else None

    acc_mask = (pred == gt).astype(np.float32) if gt is not None else np.ones_like(pred, dtype=np.float32)
    
    # Compute bounding box with padding for the accuracy mask
    bbox = bounding_box_3d_with_padding(acc_mask, padding=5)
    
    # Use the bounding box to crop the arrays
    min_z, max_z, min_y, max_y, min_x, max_x = bbox

    # Crop the accuracy and uncertainty arrays using the bounding box
    cropped_acc = acc_mask[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
    cropped_uq = uq[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]

    return cropped_acc, cropped_uq

def compute_ECE(acc, prob, bins=20):
    """
    Compute reliability diagram and Expected Calibration Error (ECE).
    prob should represent model confidence (1 - uncertainty).
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
    print(ECE)

    return np.array(reliability_acc), np.array(reliability_prob), ECE
    

def main():
    parser = argparse.ArgumentParser(description="Reliability/calibration evaluation for folder of files")
    parser.add_argument("--UQ_folder", type=str, required=True, help="Folder containing uncertainty maps")
    parser.add_argument("--Pred_folder", type=str, required=True, help="Folder containing predicted segmentations")
    parser.add_argument("--GT_folder", type=str, default=None, help="Folder containing ground truth segmentations")
    parser.add_argument("--output_excel", type=str, required=True, help="Excel file to save results")
    parser.add_argument("--patients", type=str, nargs="+", default=None,
                        help="Optional list of patient IDs to process. If not provided, all patients are processed")
    parser.add_argument("--bins", type=int, default=20, help="Number of bins for reliability calculation")
    args = parser.parse_args()
    
    # determine which patients to process
    uq_files, pred_files, gt_files = get_file_mappings(args.UQ_folder, args.Pred_folder, args.GT_folder)

    print(gt_files)
    patients = sorted(set(pred_files) & set(uq_files) & (set(gt_files) if gt_files else set(pred_files)))
    if args.patients:
        patients = [pid for pid in patients if pid in args.patients]

    if not patients:
        print("No patients found matching the criteria.")
        return

    results = []
    
    # obtain calibration evaluation results per patient
    for pid in patients:
        acc_arr, uncty_arr = extract_accuracy_uncertainty_arrays(gt_files.get(pid), pred_files[pid], uq_files[pid], metric='accuracy')
        reliability_acc, reliability_prob, ECE_value = compute_ECE(acc_arr, 1 - uncty_arr, bins=args.bins)

        results.append({
            "Patient_ID": pid,
            "UQ_file": os.path.basename(uq_files[pid]),
            "Reliability_Accuracy": reliability_acc.tolist(),
            "Reliability_Probability": reliability_prob.tolist(),
            "ECE": ECE_value
        })

    df = pd.DataFrame(results)
    
    #save dataframe as excel, create new file if file does not exist yet, otherwise append data
    if os.path.exists(args.output_excel):
        with pd.ExcelWriter(args.output_excel, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results calibration analysis', index=False, header=writer.sheets.get('Results calibration analysis') is None)
    else:
        df.to_excel(args.output_excel, sheet_name='Results calibration analysis', index=False)
        
    print(f"Results saved/updated at {args.output_excel}")

if __name__ == "__main__":
    main()
