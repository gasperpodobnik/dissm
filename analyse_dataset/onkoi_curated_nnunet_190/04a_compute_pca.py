from os.path import join

import gasperp_functions.new_functions as new_functions
from utilities import utilities
import trimesh

dataset = "onkoi_curated_nnunet_190"
base_dirpath = (
    f"/media/medical/projects/head_and_neck/onkoi_2019/dissm/{dataset}/Parotid_L"
)

in_dirpath = join(
    base_dirpath, "interp_mesh_simplified_reg_scaled_correspondence_mix-rpm_scale50"
)
pca_json_dir = join(
    base_dirpath,
    "interp_mesh_simplified_reg_scaled_correspondence_mix-rpm_scale50_out",
)
pca_json_path = join(
    pca_json_dir,
    "pca.json",
)

in_fps = sorted(
    utilities.listdir_fpaths(
        dir_path=in_dirpath,
        only_files=True,
        suffix="with_correspondence_gpa.ply",
        avoid_suffix="case_01_OAR_Parotid_L_target_with_correspondence_gpa.ply",
    )
)
pca = new_functions.compute_and_save_pca(pcd_fps=in_fps, json_path=pca_json_path)
_ = trimesh.Trimesh((pca.mean_).reshape(3, -1).T).export(
    join(pca_json_dir, "mean_pca.ply")
)
