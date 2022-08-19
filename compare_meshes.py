import json
import os
from pathlib import Path
import torch
import trimesh
import numpy as np
import SimpleITK as sitk
from gasperp_functions.new_functions import mesh_2_volume


def igl_hausd(mesh1, mesh2):
    import igl

    return igl.hausdorff(mesh1.vertices, mesh1.faces, mesh2.vertices, mesh2.faces)


get_filepaths = lambda dir, suffix="": sorted(
    [os.path.join(dir, fn) for fn in os.listdir(dir) if fn.endswith(suffix)]
)

load_nifti = lambda fp: sitk.ReadImage(fp)

pred_adjusted_objs_dir = (
    "/media/medical/projects/head_and_neck/onkoi_2019/dissm/nnunet-pred-test/out"
)
pred_img_dir = (
    "/media/medical/projects/head_and_neck/onkoi_2019/dissm/nnunet-pred-test/Parotid_L"
)
test_scaled_dir = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/test/Parotid_L_interp_mesh_scaled"
test_img_dir = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/test/Parotid_L"

latent_vec_json_path = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/nnunet-pred-test/out/latent_vec_dict.json"

MODEL_CKPT = "/media/medical/gasperp/projects/dissm/network/ckpts/last_checkpoint.ckpt"

loaded = torch.load(MODEL_CKPT)
latent_vecs = loaded["component.latent_vector"]["weight"].numpy()
latent_vec_mean = latent_vecs.mean(axis=0)
latent_vec_std = latent_vecs.std(axis=0)


with open(latent_vec_json_path, "r") as f:
    latent_vec_json = json.load(f)

pred_adjusted_objs = get_filepaths(pred_adjusted_objs_dir, ".obj")
pred_imgs = get_filepaths(pred_img_dir, ".nii.gz")
test_scaled_objs = get_filepaths(test_scaled_dir, ".obj")
test_imgs = get_filepaths(test_img_dir, ".nii.gz")

print(len(pred_adjusted_objs), len(test_scaled_objs), len(pred_imgs), len(test_imgs))

import sys

sys.path.append(r"/media/medical/gasperp/projects/surface-distance")
from surface_distance import metrics

results = []
for test_fp, pred_fp, test_img_fp, pred_img_fp in zip(
    test_scaled_objs, pred_adjusted_objs, test_imgs, pred_imgs
):
    mesh_test = trimesh.load(test_fp)
    mesh_pred = trimesh.load(pred_fp)
    test_sitk = load_nifti(test_img_fp)
    pred_sitk = load_nifti(pred_img_fp)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(test_sitk.GetOrigin())
    resampler.SetOutputSpacing(test_sitk.GetSpacing())
    resampler.SetSize(test_sitk.GetSize())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    pred_np_img = sitk.GetArrayFromImage(resampler.Execute(pred_sitk))
    test_np_img = sitk.GetArrayFromImage(test_sitk)

    test_np = mesh_2_volume(mesh_test)
    pred_np = mesh_2_volume(mesh_pred)

    dice_mesh = metrics.compute_dice_coefficient(
        mask_gt=test_np.astype(bool), mask_pred=pred_np.astype(bool)
    )
    surface_distances_mesh = metrics.compute_surface_distances(
        mask_gt=test_np.astype(bool),
        mask_pred=pred_np.astype(bool),
        spacing_mm=[1 / 300] * 3,
    )
    hd95_mesh = metrics.compute_robust_hausdorff(surface_distances_mesh, percent=95)

    dice_img = metrics.compute_dice_coefficient(
        mask_gt=test_np_img.astype(bool), mask_pred=pred_np_img.astype(bool)
    )
    surface_distances_img = metrics.compute_surface_distances(
        mask_gt=test_np_img.astype(bool),
        mask_pred=pred_np_img.astype(bool),
        spacing_mm=test_sitk.GetSpacing()[::-1],
    )
    # igl_hausd(mesh1=mesh_test, mesh2=mesh_pred)
    hd95_img = metrics.compute_robust_hausdorff(surface_distances_img, percent=95)

    pred_latent_vec = latent_vec_json[
        Path(pred_fp).name.replace("inferred_mean_mesh_", "").replace("_e300.obj", "")
    ]

    res = dict(
        test_path=test_fp,
        pred_path=pred_fp,
        dice__mesh=dice_mesh,
        hd95_mesh=hd95_mesh,
        dice_img=dice_img,
        hd95_img=hd95_img,
        pred_norm=np.linalg.norm(pred_latent_vec),
        pred_minus_mean_norm=np.linalg.norm(pred_latent_vec - latent_vec_mean),
        pred_minus_mean_div_by_std_norm=np.linalg.norm(
            (pred_latent_vec - latent_vec_mean) / latent_vec_std
        ),
    )
    results.append(res)
    print(res)

with open(
    "/media/medical/projects/head_and_neck/onkoi_2019/dissm/nnunet-pred-test/out/results.json",
    "w",
) as fp:
    json.dump(results, fp)


with open(
    "/media/medical/projects/head_and_neck/onkoi_2019/dissm/nnunet-pred-test/out/results.json",
    "r",
) as f:
    results = json.load(f)

import pandas as pd
from pandas.plotting import scatter_matrix

df = pd.DataFrame(results)


scatter_matrix(df.drop(columns=['test_path', 'pred_path']), alpha=0.2, figsize=(6, 6), diagonal="kde")
