from os.path import join

import gasperp_functions.run_rpm_tps as run_rpm_tps

base_dirpath = (
    r"/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/Parotid_L"
)
in_dirpath = join(base_dirpath, "interp_mesh_simplified_reg_scaled")
ref_mesh_fp_subsampled_scaled = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/ref_subsampled1400_scaled50.ply"
SCALE = 50
count = 1400
# ref_mesh_fp_NOT_subsampled = "/media/medical/projects/head_and_neck/onkoi_2019/dissm/onkoi-train/out/infer_mean_mesh.obj"

# m_method = "mix-rpm": this is better, because final registered points are better distributed over entire structure
cmix_params = dict(
    frac=1,
    T_init=50,
    T_finalfac=10,
    m_method="mix-rpm",
    lamda1_init=1,
    lamda2_init=0.01,
    perT_maxit=5,
    verbose=True,
)

run_rpm_tps.register_dir(
    in_dirpath=in_dirpath,
    cmix_params=cmix_params,
    SCALE=SCALE,
    count=count,
    ref_mesh_fp_subsampled_scaled=ref_mesh_fp_subsampled_scaled,
)
