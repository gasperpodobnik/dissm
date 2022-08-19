from os.path import join

import gasperp_functions.new_functions as new_functions
import numpy as np
import trimesh
from utilities import utilities

base_dirpath = (
    r"/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/Parotid_L"
)
take_N_components = 3
pca_json_path = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/Parotid_L/interp_mesh_simplified_reg_scaled_correspondence_mix-rpm_out/pca.json"
in_dirpath = join(
    base_dirpath, "interp_mesh_simplified_reg_scaled_correspondence_mix-rpm"
)


in_fps = sorted(
    utilities.listdir_fpaths(
        dir_path=in_dirpath,
        only_files=True,
        suffix="with_correspondence.ply",
    )
)

m_dist = new_functions.compute_mahalanobis(
    pca_json_path=pca_json_path, take_N_components=take_N_components
)

results = []
for fp in in_fps:
    pcd = new_functions.scale_pcd(
        trimesh.load(fp), centroid=np.array([50.0, 50.0, 50.0]), max_dist=50.0
    )
    dist = m_dist.compute(pcd.vertices.flatten("F"))
    results.append(
        {
            "path": fp,
            "mahalanobis_distance": dist,
            "take_N_components": take_N_components,
        }
    )

utilities.write_dict_as_json(results, join(in_dirpath, "mahalanobis_distance.json"))
