from os.path import join
from implicitshapes import interpolate_mask, convert

root_dir = "/media/medical/projects/head_and_neck/onkoi_2019/dissm"
# PHASE = "nnunet-pred-test"
PHASE = "test"


phase_dir = join(root_dir, PHASE)
mask_folder = join(phase_dir, "Parotid_L")
interp_mask_folder = join(phase_dir, "Parotid_L_interp")
mesh_folder = join(phase_dir, "Parotid_L_interp_mesh")
scaled_mesh_folder = join(phase_dir, "Parotid_L_interp_mesh_scaled")
simplified_mesh_folder = join(phase_dir, "Parotid_L_interp_mesh_simplified")
super_simplified_mesh_folder = join(phase_dir, "Parotid_L_interp_mesh_simplified_super")
super_simplified_registered_folder = join(
    phase_dir, "Parotid_L_interp_mesh_simplified_super_reg"
)
simplified_registered_folder = join(phase_dir, "Parotid_L_interp_mesh_simplified_reg")
sdf_folder = join(phase_dir, "Parotid_L_sdf")

# python /media/medical/gasperp/projects/dissm/implicitshapes/embed_shape.py --im_root /media/medical/projects/head_and_neck/onkoi_2019/dissm/train/Parotid_L_sdf --yaml_file /media/medical/gasperp/projects/dissm/implicitshapes/configs/embed_config.yml --save_path /media/medical/gasperp/projects/dissm/network --file_list /media/medical/projects/head_and_neck/onkoi_2019/dissm/train/Parotid_L_sdf/json_list.json

# python /media/medical/gasperp/projects/dissm/implicitshapes/adjust_to_shape.py \
# --im_root /media/medical/projects/head_and_neck/onkoi_2019/dissm/test/Parotid_L_sdf \
# --model_path /media/medical/gasperp/projects/dissm/network/ckpts/last_checkpoint.ckpt \
# --yaml_file /media/medical/gasperp/projects/dissm/implicitshapes/configs/embed_config.yml \
# --save_path /media/medical/gasperp/projects/dissm/network_test_and_adjust \
# --file_list /media/medical/projects/head_and_neck/onkoi_2019/dissm/test/Parotid_L_sdf/json_list0.json

from implicitshapes import infer_shape
from pathlib import Path

OUT_DIR = join(root_dir, "train", "out")
MODEL_CKPT = "/media/medical/gasperp/projects/dissm/network/ckpts/last_checkpoint.ckpt"
YML_CONFIG = (
    "/media/medical/gasperp/projects/dissm/implicitshapes/configs/embed_config.yml"
)
SDF_RESOLUTION = 300

# basic inference
infer_shape.infer_mean_mesh(
    MODEL_CKPT, YML_CONFIG, join(OUT_DIR, "infer_mean_mesh.obj"), SDF_RESOLUTION
)
infer_shape.conduct_pca(MODEL_CKPT, YML_CONFIG, join(OUT_DIR, "pca"), SDF_RESOLUTION)


# ------------------------------------------
# ------------------------------------------
# pca gif
pca = infer_shape._get_pca(MODEL_CKPT)
vec1 = pca.mean_ - pca.components_[0, :]
vec2 = pca.mean_ + pca.components_[0, :]

infer_shape.interpolate_mesh_given_vectors(
    model_path=MODEL_CKPT,
    config_file=YML_CONFIG,
    first_latent_vec=vec1,
    second_latent_vec=vec2,
    save_loc_root=join(OUT_DIR, "pca_gif", "pca_component0_steps10"),
    sdf_size=SDF_RESOLUTION,
    steps=10,
)

LATENT_IDX = 3
infer_shape.infer_mesh(
    MODEL_CKPT,
    YML_CONFIG,
    LATENT_IDX,
    join(OUT_DIR, "infer_mesh_num3.obj"),
    SDF_RESOLUTION,
)


# ------------------------------------------
# ------------------------------------------
# train a model to learn new latent vectors so that they best approximate a given (test) mesh
from implicitshapes import adjust_to_shape
import json
import torch
import numpy as np

list_file = join(sdf_folder, "json_list.json")
with open(list_file, "r") as f:
    json_list = json.load(f)

OUT_DIR = join(root_dir, PHASE, "out")
MODEL_CKPT = "/media/medical/gasperp/projects/dissm/network_test_and_adjust/ckpts/last_checkpoint.ckpt"
YML_CONFIG = (
    "/media/medical/gasperp/projects/dissm/implicitshapes/configs/embed_config.yml"
)
SDF_RESOLUTION = 300
ADJ_EPOCHS = 300

latent_vecs_dict = {}
for file in json_list:
    latent_vecs = adjust_to_shape.run_adjust_to_shape(
        {
            "im_root": f"/media/medical/projects/head_and_neck/onkoi_2019/dissm/{PHASE}/Parotid_L_sdf",
            "model_path": "/media/medical/gasperp/projects/dissm/network/ckpts/last_checkpoint.ckpt",
            "file_list": [file],
            "epochs": ADJ_EPOCHS,
            "yaml_file": YML_CONFIG,
            "save_path": "/media/medical/gasperp/projects/dissm/network_test_and_adjust",
        }
    )
    # loaded = torch.load(MODEL_CKPT)
    # latent_vecs = loaded['component.latent_vector']['weight']
    latent_vecs_dict[file["path"]] = np.squeeze(
        latent_vecs.detach().cpu().numpy()
    ).tolist()
    infer_shape.infer_mean_mesh(
        "/media/medical/gasperp/projects/dissm/network_test_and_adjust/ckpts/last_checkpoint.ckpt",
        YML_CONFIG,
        join(
            OUT_DIR, f'inferred_mean_mesh_{Path(file["path"]).name}_e{ADJ_EPOCHS}.obj'
        ),
        SDF_RESOLUTION,
    )

with open(join(OUT_DIR, "latent_vec_dict.json"), "w") as fp:
    json.dump(latent_vecs_dict, fp)
# ------------------------------------------

# infer_shape.infer_mean_mesh(MODEL_CKPT, YML_CONFIG, join(out_dir, 'infer_mean_mesh43_epoch1000.obj'), SDF_RESOLUTION)
# infer_shape.conduct_pca(MODEL_CKPT, YML_CONFIG, join(out_dir, 'pca'), SDF_RESOLUTION)

# LATENT_IDX = 3
# infer_shape.infer_mesh(MODEL_CKPT, YML_CONFIG, LATENT_IDX, join(out_dir, 'infer_mesh_num3.obj'), SDF_RESOLUTION)
