import os

def save_results_to_excel(geo_df=None, dose_df=None, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    if geo_df is not None and not geo_df.empty:
        geo_file = os.path.join(output_dir, f"geo_results_nnunet_ood.xlsx")
        geo_df.to_excel(geo_file, index=False)
        print(f"Saved geo results to {geo_file}")

    if dose_df is not None and not dose_df.empty:
        dose_file = os.path.join(output_dir, f"dose_results_nnunet_ood.xlsx")
        dose_df.to_excel(dose_file, index=False)
        print(f"Saved dose results to {dose_file}")