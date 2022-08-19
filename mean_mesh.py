import json
import os
import sys
from os.path import join
import numpy as np

sys.path.append(r"/media/medical/gasperp/projects")
from utilities import utilities  # , point_utilities


import trimesh
from implicitshapes.mesh_utils import scale_mesh, rigid_align_meshes

# from functions.utility_functions import (
#     copy_image_metadata,
#     visualization_functions,
#     utility_functions,
#     preprocessing_functions,
# )

import gasperp_functions.new_functions as new_functions

## uncomment to create a mean mesh
meshes_fp = utilities.listdir_fpaths(
    "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/Parotid_L_interp_mesh_simplified_reg_scaled",
    suffix=".obj",
)
meshes = [trimesh.load(fp) for fp in meshes_fp]
new_functions.get_mean_mesh(
    meshes,
    mean_sitk_path="/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/mean_mesh/Parotid_L_mean.nii.gz",
)


def load_img_and_get_smoothed_verts(pth):
    verts, faces = point_utilities.load_and_march_mesh(pth)

    mesh_smoothed = point_utilities.create_mesh_form_vertices_and_faces(
        vertices=verts, faces=faces, smooth=True
    )
    verts_smoothed, _ = trimesh.sample.sample_surface(mesh_smoothed, count=count)
    return verts_smoothed


base_dirpath = r"/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train"
in_dirpath = join(base_dirpath, "Parotid_L_interp_mesh_simplified_reg_scaled")
out_dirpath = join(base_dirpath, "Parotid_L_interp_mesh_scaled_subsampled_1500")
os.makedirs(out_dirpath, exist_ok=True)

count = 1400


mesh_path = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/mean_mesh/Parotid_L_mean.nii.gz.obj"
anchor_mesh_path_super_simpl = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/Parotid_L_interp_mesh_simplified_super/case_01_OAR_Parotid_L.obj"
anchor_mesh_path_simpl = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/Parotid_L_interp_mesh_simplified/case_01_OAR_Parotid_L.obj"


# super simplified registration
cur_path = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/mean_mesh/Parotid_L_mean.nii.gz.obj_simplified_super.obj"
out_file = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/mean_mesh/Parotid_L_mean.nii.gz.obj_simplified_super_reg.obj"
anchor_mesh_path = anchor_mesh_path_super_simpl
anchor_mesh = trimesh.load(anchor_mesh_path)
moving_mesh = trimesh.load(cur_path)
# perform rigid registration using CPD based on the two sets of vertices
new_mesh, cur_dict = rigid_align_meshes(anchor_mesh, moving_mesh)
cur_dict["anchor_name"] = os.path.basename(anchor_mesh_path)
new_mesh.export(out_file)
# save the sacle, translation, and rotation and anchor mesh name to a json file
with open(out_file + ".json", "w") as f:
    json.dump(cur_dict, f)

# simplified registration
cur_path = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/mean_mesh/Parotid_L_mean.nii.gz.obj_simplified.obj"
out_file = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/mean_mesh/Parotid_L_mean.nii.gz.obj_simplified_reg.obj"
# load up the transform parameters from the json file
json_file = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/mean_mesh/Parotid_L_mean.nii.gz.obj_simplified_super_reg.obj.json"
with open(json_file, "r") as f:
    params = json.load(f)
scale = params["s"]
R = np.array(params["R"])
t = np.array(params["t"])
# apply the transform to the vertices
moving_mesh = trimesh.load(cur_path)
moving_vertices = moving_mesh.vertices
moving_vertices = scale * np.dot(moving_vertices, R) + t
# save the aligned mesh
new_mesh = trimesh.Trimesh(moving_vertices, moving_mesh.faces)
new_mesh.export(out_file)
scale_mesh(new_mesh).export(out_file + "_scaled.obj")
