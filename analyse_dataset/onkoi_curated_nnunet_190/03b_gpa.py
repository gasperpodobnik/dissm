from os.path import join

import numpy as np
import trimesh
from gasperp_functions import new_functions
from procrustes import generalized
from utilities import utilities
from gasperp_functions.run_rpm_tps import set_color_and_save

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


array_aligned_list, new_distance_gpa, new_ref = generalized(
    array_list=array_list,
    ref=ref,
    tol=1e-12,
    n_iter=500,
)

new_ref_path = join(in_dirpath, "GPA_new_ref.ply")
_ = trimesh.Trimesh(new_ref).export(new_ref_path)
print("\n", new_ref_path, "\n")


[
    set_color_and_save(
        new_verts,
        fp=orig_fp.replace(".ply", "_gpa.ply"),
        color=[131, 149, 69],
    )
    for orig_fp, new_verts in zip(in_fps, array_aligned_list)
]


# import numpy as np


# def procrustes_distance(reference_shape, shape):

#     ref_x = reference_shape[0, :]
#     ref_y = reference_shape[1, :]
#     ref_z = reference_shape[2, :]

#     x = shape[0, :]
#     y = shape[1, :]
#     z = shape[2, :]

#     dist = np.sum(np.sqrt((ref_x - x) ** 2 + (ref_y - y) ** 2 + (ref_z - z) ** 2))

#     return dist


# procrustes_distance(reference_shape=ref, shape=array_list[0])
