import logging
import os
import time
from os.path import join

import numpy as np
import trimesh
from point_library.rpm_tps import m_functions
from tqdm.std import tqdm
from utilities import utilities


def create_reference_mesh(
    ref_mesh_fp="/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/out/infer_mean_mesh.obj",
    count=1400,
    SCALE=50,
):
    ref_mesh_smoothed = trimesh.load(ref_mesh_fp)
    ref_verts_subsampled, ref_faces_smoothed = trimesh.sample.sample_surface_even(
        ref_mesh_smoothed, count=count
    )
    assert ref_verts_subsampled.shape[0] == (count), "incorrect number of vertices"
    ref_verts_subsampled *= SCALE
    ref_verts_subsampled += SCALE
    return ref_verts_subsampled


def run_registration(
    input_mesh_path, out_dirpath, ref_verts_scaled, cmix_params, SCALE=50, count=1400
):
    start = time.time()
    logging.info(f"Starting with file:\t{input_mesh_path}")
    fname = utilities.get_filename_without_format(input_mesh_path)

    curr_mesh = trimesh.load(input_mesh_path)
    # verts_smoothed = trimesh.sample.sample_surface(mesh, count=count)
    (
        curr_verts_subsampled,
        curr_faces_idx_subsampled,
    ) = trimesh.sample.sample_surface_even(curr_mesh, count=count)
    assert curr_verts_subsampled.shape[0] == count, "incorrect number of vertices"

    curr_verts_subsampled *= SCALE
    curr_verts_subsampled += SCALE

    V_source = ref_verts_scaled
    X_target = curr_verts_subsampled
    set_color_and_save(
        X_target,
        fp=join(out_dirpath, fname + "_target.ply"),
        color=[255, 255, 0],
    )

    c, d, probability_matrix = m_functions.cMIX(
        V_source=V_source, X_target=X_target, **cmix_params
    )

    n, dim = V_source.shape
    V_source_a = np.concatenate((np.ones((n, 1)), V_source), axis=1)
    X_target_a = np.concatenate((np.ones((n, 1)), X_target), axis=1)
    d2 = np.linalg.inv(d)
    V_source_transformed_a = (V_source_a @ d)[:, 1 : dim + 1]
    set_color_and_save(
        V_source_transformed_a,
        fp=join(out_dirpath, fname + "_affine_reg.ply"),
        color=[255, 128, 0],
    )
    V_source_transformed_a2 = (X_target_a @ d2)[:, 1 : dim + 1]
    set_color_and_save(
        V_source_transformed_a2,
        fp=join(out_dirpath, fname + "_target_inv_affine_reg.ply"),
        color=[255, 128, 128],
    )

    PHI = m_functions.ctps_gen(V_source, V_source, nargout=1)
    V_source_transformed22 = V_source_a + PHI @ c
    V_source_transformed22 = V_source_transformed22[:, 1 : dim + 1]
    set_color_and_save(
        V_source_transformed22,
        fp=join(out_dirpath, fname + "_just_nonlinear_reg.ply"),
        color=[255, 128, 128],
    )

    ref_verts_reg = m_functions.cMIX_warp_pts(
        V_source=V_source,
        z=V_source,
        c_tps=c,
        d_tps=d,
    )
    Y_target_with_correspondence = m_functions.update_correspondence(
        X_target, probability_matrix
    )
    set_color_and_save(
        ref_verts_reg, fp=join(out_dirpath, fname + "_reg.ply"), color=[255, 0, 0]
    )
    set_color_and_save(
        Y_target_with_correspondence,
        fp=join(out_dirpath, fname + "_target_with_correspondence.ply"),
        color=[0, 0, 255],
    )
    end = time.time()
    logging.info(f"Finished with current file")
    logging.info(f"Time consumed:\t{np.round(end - start)} s\n")


def set_color_and_save(vertices, fp, color):
    pcd = trimesh.PointCloud(vertices)
    pcd.visual.vertex_colors = [color] * len(vertices)
    _ = pcd.export(fp)


def register_dir(
    in_dirpath, cmix_params, ref_mesh_fp_subsampled_scaled, SCALE=50, count=1400
):
    # create out dir path
    out_dirpath = in_dirpath + f"_correspondence_{cmix_params['m_method']}_scale{SCALE}"
    os.makedirs(out_dirpath, exist_ok=True)

    # create logger
    utilities.setup_logging(
        join(out_dirpath, "RPM-TPS.log"),
        delete_existing=True,
        level=logging.INFO,
    )

    # create new reference mesh (subsample and smooth ref mesh)
    # ref_verts_subsampled_scaled = create_reference_mesh(
    #     ref_mesh_fp=ref_mesh_fp_NOT_subsampled, count=count, SCALE=SCALE
    # )

    # load already subsampled ref vertices
    ref_verts_subsampled_scaled = trimesh.load(ref_mesh_fp_subsampled_scaled).vertices
    set_color_and_save(
        ref_verts_subsampled_scaled, fp=join(out_dirpath, "ref.ply"), color=[0, 255, 0]
    )

    input_meshes = sorted(
        utilities.listdir_fpaths(
            dir_path=in_dirpath,
            only_files=True,
            suffix=".obj",
        )
    )

    logging.info(f"Running RPM-TPS")
    logging.info(f"Input dir:\t{in_dirpath}")
    logging.info(f"Output dir:\t{out_dirpath}")

    for enum, input_mesh_path in tqdm(
        enumerate(input_meshes),
        total=len(input_meshes),
        desc="running RPM-TPS for the entire dir of meshes",
    ):
        run_registration(
            input_mesh_path=input_mesh_path,
            out_dirpath=out_dirpath,
            ref_verts_scaled=ref_verts_subsampled_scaled,
            cmix_params=cmix_params,
            SCALE=SCALE,
            count=count,
        )
    logging.info(f"Finished with all files")
