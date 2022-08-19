from os.path import join

import gasperp_functions.new_functions as new_functions

dataset = "hn1_pred_180"

base_dirpath = (
    f"/media/medical/projects/head_and_neck/onkoi_2019/dissm/{dataset}/Parotid_L"
)
take_N_components = 8

# pca_json_path = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/Parotid_L/interp_mesh_simplified_reg_scaled_correspondence_mix-rpm_out/pca.json"
pca_json_path = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi_curated_nnunet_190/Parotid_L/interp_mesh_simplified_reg_scaled_correspondence_mix-rpm_scale50_out/pca.json"

in_dirpath = join(
    base_dirpath, "interp_mesh_simplified_reg_scaled_correspondence_mix-rpm_scale50"
)

nnunet_results_csv_path = "/media/medical/projects/head_and_neck/nnUnet/Task180_onkoi-2019-CT-resampled/results/results.csv"

new_functions.compute_mahalanobis_dir(
    in_dirpath,
    dataset,
    pca_json_path,
    nnunet_results_csv_path,
    take_N_components,
)
