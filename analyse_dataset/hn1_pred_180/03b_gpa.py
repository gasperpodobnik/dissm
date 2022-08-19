from os.path import join

import numpy as np
import trimesh
from gasperp_functions import new_functions
from gasperp_functions.run_rpm_tps import set_color_and_save
from procrustes import generic
from utilities import utilities

dataset = "hn1_pred_180"
base_dirpath = (
    f"/media/medical/projects/head_and_neck/onkoi_2019/dissm/{dataset}/Parotid_L"
)
in_dirpath = join(
    base_dirpath, "interp_mesh_simplified_reg_scaled_correspondence_mix-rpm_scale50"
)

gpa_ref_mesh_path = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi_curated_nnunet_190/Parotid_L/interp_mesh_simplified_reg_scaled_correspondence_mix-rpm_scale50/GPA_new_ref.ply"


in_fps = sorted(
    utilities.listdir_fpaths(
        dir_path=in_dirpath,
        only_files=True,
        suffix="with_correspondence.ply",
    )
)

gpa_ref = trimesh.load(gpa_ref_mesh_path).vertices

for fp in in_fps:
    a = new_functions.scale_pcd(
        trimesh.load(fp), centroid=np.array([50.0, 50.0, 50.0]), max_dist=50.0
    ).vertices
    res = generic(a, gpa_ref, translate=True, scale=True)
    set_color_and_save(
        res["new_a"],
        fp=fp.replace(".ply", "_gpa.ply"),
        color=[131, 149, 69],
    )
