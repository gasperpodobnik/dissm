import os
from os.path import join

# pcd to mesh, experimental
import numpy as np
import trimesh

# from implicitshapes.mesh_utils import scale_mesh, rigid_align_meshes
# from implicitshapes import convert
from sklearn.decomposition import PCA
from utilities import utilities

import gasperp_functions.new_functions as new_functions


# from point_library.rpm_tps import m_functions

COMPONENT_NUMs = [0, 1, 2, 3, 4]
SIGMA_factor = 2
N_STEPS = 8  # must be even number (2, 4, 6...)
SAVE_ply_and_obj = False
pca_path = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi_curated_nnunet_190/Parotid_L/interp_mesh_simplified_reg_scaled_correspondence_mix-rpm_scale50_out/pca.json"
out_dirpath = pca_path.replace("/pca.json", "_pca")
os.makedirs(out_dirpath, exist_ok=True)


pca = new_functions.load_pca_json(pca_path)

for COMPONENT_NUM in COMPONENT_NUMs:
    new_functions.create_gif_from_pca(
        pca_components_=pca.components_,
        pca_explained_variance_=pca.explained_variance_,
        pca_mean_=pca.mean_,
        COMPONENT_NUM=COMPONENT_NUM,
        SIGMA_factor=SIGMA_factor,
        N_STEPS=N_STEPS,
        SAVE_ply_and_obj=SAVE_ply_and_obj,
        out_dirpath=out_dirpath,
    )
