from os.path import join

import gasperp_functions.new_functions as new_functions
from utilities import utilities
import trimesh

pca_json_dir = join(f"/media/medical/projects/head_and_neck/onkoi_2019/dissm/HN1_PDDCA")
pca_json_path = join(
    pca_json_dir,
    "pca.json",
)


# HN1
dataset = "hn1_raw"
base_dirpath = (
    f"/media/medical/projects/head_and_neck/onkoi_2019/dissm/{dataset}/Parotid_L"
)

in_dirpath = join(
    base_dirpath, "interp_mesh_simplified_reg_scaled_correspondence_mix-rpm_scale50"
)

in_fps1 = sorted(
    utilities.listdir_fpaths(
        dir_path=in_dirpath,
        only_files=True,
        suffix="with_correspondence_gpa.ply",
        avoid_suffix="case_01_OAR_Parotid_L_target_with_correspondence_gpa.ply",
    )
)


# PDDCA

dataset = "pddca_raw"
base_dirpath = (
    f"/media/medical/projects/head_and_neck/onkoi_2019/dissm/{dataset}/Parotid_L"
)

in_dirpath = join(
    base_dirpath, "interp_mesh_simplified_reg_scaled_correspondence_mix-rpm_scale50"
)

in_fps2 = sorted(
    utilities.listdir_fpaths(
        dir_path=in_dirpath,
        only_files=True,
        suffix="with_correspondence_gpa.ply",
        avoid_suffix="case_01_OAR_Parotid_L_target_with_correspondence_gpa.ply",
    )
)


pca = new_functions.compute_and_save_pca(
    pcd_fps=in_fps1 + in_fps2, json_path=pca_json_path
)
_ = trimesh.Trimesh((pca.mean_).reshape(3, -1).T).export(
    join(pca_json_dir, "mean_pca.ply")
)
print(pca_json_path)
