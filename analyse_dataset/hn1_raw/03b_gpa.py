from os.path import join

import numpy as np
import trimesh
from gasperp_functions import new_functions
from gasperp_functions.run_rpm_tps import set_color_and_save
from procrustes import generic
from procrustes import generalized
from utilities import utilities


####

dataset = "onkoi_curated_nnunet_190"
base_dirpath = (
    f"/media/medical/projects/head_and_neck/onkoi_2019/dissm/{dataset}/Parotid_L"
)
in_dirpath = join(
    base_dirpath, "interp_mesh_simplified_reg_scaled_correspondence_mix-rpm_scale50"
)
ref_mesh_fp_subsampled_scaled = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/ref_subsampled1400_scaled50.ply"


in_fps = sorted(
    utilities.listdir_fpaths(
        dir_path=in_dirpath,
        only_files=True,
        suffix="with_correspondence.ply",
    )
)

array_list = [
    new_functions.scale_pcd(
        trimesh.load(fp), centroid=np.array([50.0, 50.0, 50.0]), max_dist=50.0
    ).vertices
    for fp in in_fps
]

ref = new_functions.scale_pcd(
    trimesh.load(ref_mesh_fp_subsampled_scaled),
    centroid=np.array([50.0, 50.0, 50.0]),
    max_dist=50.0,
).vertices


####

dataset = "hn1_raw"
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

    # array_aligned_list, new_distance_gpa, new_ref = generalized(
    #     array_list=[a] + array_list,
    #     ref=ref,
    #     tol=1e-12,
    #     n_iter=500,
    # )
    # set_color_and_save(
    #     array_aligned_list[0],
    #     fp=fp.replace(".ply", "_gpa_new.ply"),
    #     color=[131, 149, 69],
    # )
