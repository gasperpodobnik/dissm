import os
from os.path import join
from pathlib import Path

import trimesh

import gasperp_functions.new_functions as new_functions

# mesh_path = '/media/medical/projects/head_and_neck/onkoi_2019/dissm/test/Parotid_L_interp_mesh_simplified/case_43_OAR_Parotid_L.obj'
# mesh = trimesh.load(mesh_path)


pca_gif_dir = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/train/out/pca_gif"
png_gif_dir = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/train/out/png_gif"

pil_imgs = []
for msg_fn in sorted(os.listdir(pca_gif_dir))[-2:-1]:
    mesh = trimesh.load(join(pca_gif_dir, msg_fn))
    pil_img = new_functions.plot_mesh(
        mesh,
        show=False,
        save_filepath=None,  # join(png_gif_dir, msg_fn.replace(".obj", ".png")),
        eye=dict(x=-0.5, y=-0.5, z=1.5),
        lightposition_kwargs=dict(x=-100, y=-100, z=0),
    )
    pil_imgs.append(pil_img)

fp_out = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/train/out/component0_steps10.gif"
new_functions.save_to_gif(pil_imgs, fp_out)
