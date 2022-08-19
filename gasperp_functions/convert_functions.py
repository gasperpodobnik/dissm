from os.path import join
from implicitshapes import interpolate_mask, convert


def convert_dir(
    input_dir,
    anchor_mesh_path_super_simpl="/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/Parotid_L/interp_mesh_simplified_super/case_01_OAR_Parotid_L.obj",
    anchor_mesh_path_simpl="/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/Parotid_L/interp_mesh_simplified/case_01_OAR_Parotid_L.obj",
    compute_sdf=False,
):
    mask_folder = join(input_dir, "nifti")
    interp_mask_folder = join(input_dir, "interp")
    mesh_folder = join(input_dir, "interp_mesh")
    scaled_mesh_folder = join(input_dir, "interp_mesh_scaled")
    simplified_mesh_folder = join(input_dir, "interp_mesh_simplified")
    super_simplified_mesh_folder = join(input_dir, "interp_mesh_simplified_super")
    super_simplified_registered_folder = join(
        input_dir, "interp_mesh_simplified_super_reg"
    )
    simplified_registered_folder = join(input_dir, "interp_mesh_simplified_reg")
    scaled_simplified_registered_folder = join(
        input_dir, "interp_mesh_simplified_reg_scaled"
    )
    sdf_folder = join(input_dir, "sdf")

    interpolate_mask.interpolate_all_masks(mask_folder, interp_mask_folder)
    convert.convert_all_masks(interp_mask_folder, mesh_folder)

    QBIN_PATH = "/tmp/Fast-Quadric-Mesh-Simplification/bin.Linux/simplify"
    convert.simplify_all_meshes(mesh_folder, simplified_mesh_folder, QBIN_PATH, 0.1)
    convert.simplify_all_meshes(
        mesh_folder, super_simplified_mesh_folder, QBIN_PATH, 0.01
    )

    convert.register_meshes(
        super_simplified_mesh_folder,
        super_simplified_registered_folder,
        anchor_mesh_path=anchor_mesh_path_super_simpl,
    )
    convert.align_large_meshes(
        simplified_mesh_folder,
        super_simplified_registered_folder,
        simplified_registered_folder,
        anchor_mesh_path=anchor_mesh_path_simpl,
    )

    if compute_sdf:
        convert.sample_sdf(
            simplified_registered_folder,
            sdf_folder,
            number_of_points=1000000,
            uniform_proportion=0.2,
            jitter=0.1,
        )
        convert.create_sample_json(sdf_folder)

    convert.scale_all_meshes(mesh_folder, scaled_mesh_folder)
    convert.scale_all_meshes(
        simplified_registered_folder, scaled_simplified_registered_folder
    )
