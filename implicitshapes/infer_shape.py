import numpy as np
import os
from skimage import measure
import trimesh
import nibabel as ni
import torch
from sklearn.decomposition import PCA
from implicitshapes.networks.deep_sdf_decoder import create_decoder_model
import SimpleITK as sitk
from implicitshapes.dlt_utils import read_yaml, load_model_from_ckpt


"""
Infer an SDF given a latent code in directly in normalized [-1, 1] space
Uniformly samples an SDF based off of the sdf_size parameter
"""


def infer_from_latent(
    model_path, config_file, latent_vec, sdf_size=300, batch_size=50000
):

    config = read_yaml(config_file)

    model = create_decoder_model(config)
    load_model_from_ckpt(model, model_path)

    model = model.cuda()
    model.eval()

    size_range = np.linspace(-1, 1, sdf_size)
    points = np.meshgrid(size_range, size_range, size_range)

    points = np.stack(points)

    points = np.swapaxes(points, 1, 2)

    points = points.reshape(3, -1).transpose()

    samples = torch.from_numpy(points).float().cuda()

    if not torch.is_tensor(latent_vec):
        latent_vec = torch.from_numpy(latent_vec)

    latent_vec = torch.unsqueeze(latent_vec, 0)

    latent_vec = latent_vec.repeat(batch_size, 1).cuda()

    total_size = samples.shape[0]

    sdf = np.zeros(total_size)

    cur_idx = 0

    while cur_idx < total_size:
        end_idx = min(total_size, cur_idx + batch_size)
        print(end_idx)

        with torch.no_grad():

            batch_samples = samples[cur_idx:end_idx, :]
            batch_samples = torch.cat(
                (latent_vec[: batch_samples.shape[0]], batch_samples), 1
            )
            batch_dict = {"samples": batch_samples}

            batch_dict = model(batch_dict)
            output = batch_dict["output"].cpu().detach().numpy()

            sdf[cur_idx:end_idx] = np.squeeze(output)

        cur_idx = end_idx
    if not isinstance(sdf_size, list) and not isinstance(sdf_size, tuple):
        sdf = sdf.reshape((sdf_size, sdf_size, sdf_size))
    else:
        sdf = sdf.reshape(sdf_size)
    return sdf


def infer_from_latent_affine_mtx(
    model_path, config_file, latent_vec, sdf_size, affine_mtx, Apx2sdf, batch_size=50000
):

    config = read_yaml(config_file)

    model = create_decoder_model(config)
    load_model_from_ckpt(model, model_path)

    model = model.cuda()
    model.eval()

    # create a centering transform to allow rotations to be properly applied
    sdf_size = np.asarray(sdf_size)
    center_affine = np.eye(4)
    center_affine[:3, 3] = -(sdf_size - 1) / 2

    size_range_i = np.arange(0, sdf_size[0])
    size_range_j = np.arange(0, sdf_size[1])
    size_range_k = np.arange(0, sdf_size[2])

    points = np.meshgrid(size_range_i, size_range_j, size_range_k)

    points = np.stack(points)

    points = np.swapaxes(points, 1, 2)

    points = points.reshape(3, -1).transpose()
    dummy = np.ones((points.shape[0], 1))

    points = np.concatenate((points, dummy), 1)

    A = center_affine
    A = np.linalg.solve(affine_mtx, A)
    A = np.linalg.solve(center_affine, A)

    A = Apx2sdf @ A
    A = np.transpose(A)

    points = points @ A
    points = points[:, :3]

    samples = torch.from_numpy(points).float().cuda()

    if not torch.is_tensor(latent_vec):
        latent_vec = torch.from_numpy(latent_vec)

    latent_vec = torch.unsqueeze(latent_vec, 0)

    latent_vec = latent_vec.repeat(batch_size, 1).cuda()

    total_size = samples.shape[0]

    sdf = np.zeros(total_size)

    cur_idx = 0

    while cur_idx < total_size:
        end_idx = min(total_size, cur_idx + batch_size)
        print(end_idx)

        with torch.no_grad():

            batch_samples = samples[cur_idx:end_idx, :]
            batch_samples = torch.cat(
                (latent_vec[: batch_samples.shape[0]], batch_samples), 1
            )
            batch_dict = {"samples": batch_samples}

            batch_dict = model(batch_dict)
            output = batch_dict["output"].cpu().detach().numpy()

            sdf[cur_idx:end_idx] = np.squeeze(output)

        cur_idx = end_idx
    sdf = sdf.reshape(sdf_size)
    return sdf


def infer_sdf(
    model_path, config_file, latent_idx, save_loc, sdf_size=300, batch_size=50000
):

    loaded = torch.load(model_path)
    latent_vecs = loaded["component.latent_vector"]["weight"]

    latent_vec = latent_vecs[latent_idx, :]

    sdf = infer_from_latent(model_path, config_file, latent_vec, sdf_size, batch_size)

    if save_loc != None:

        sdf_ni = ni.Nifti1Image(sdf, np.eye(4))
        ni.save(sdf_ni, save_loc)

    return sdf


def scale_mesh(mesh, factor=1):
    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def infer_mean_mesh(model_path, config_file, save_loc, sdf_size=300, batch_size=500000):

    loaded = torch.load(model_path)
    latent_vecs = loaded["component.latent_vector"]["weight"]

    latent_vec = torch.mean(latent_vecs, 0)

    sdf = infer_from_latent(
        model_path, config_file, latent_vec, sdf_size=sdf_size, batch_size=batch_size
    )

    vertices, triangles, _, _ = measure.marching_cubes(sdf, 0, allow_degenerate=False)

    create_dir_for_file(save_loc)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh = scale_mesh(mesh)
    mesh.export(save_loc)


def create_dir_for_file(file_save_loc):
    out_folder = file_save_loc.rsplit("/", 1)[0]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)


def infer_mean_aligned_sdf(
    model_path, config_file, save_loc, ref_im, scale, affine_mtx=None, batch_size=500000
):
    """
    Compute an SDF of the mean latent vector and project it to the ref_im pixel space
    Must also input the scale factor that scale pixel values to the SDF canonical space
    """
    if affine_mtx is None:
        affine_mtx = np.eye(4)
    loaded = torch.load(model_path)
    latent_vecs = loaded["component.latent_vector"]["weight"]
    # import pdb;pdb.set_trace()
    latent_vec = torch.mean(latent_vecs, 0)
    ref_im = sitk.ReadImage(ref_im)
    sdf_size = ref_im.GetSize()
    ref_spacing = ref_im.GetSpacing()

    Apx2sdf = np.eye(4)
    Apx2sdf[0, 3] = -(sdf_size[0] - 1) / 2
    Apx2sdf[1, 3] = -(sdf_size[1] - 1) / 2
    Apx2sdf[2, 3] = -(sdf_size[2] - 1) / 2

    Apx2sdf[0, :] *= -ref_spacing[0] / scale
    Apx2sdf[1, :] *= -ref_spacing[1] / scale
    Apx2sdf[2, :] *= ref_spacing[2] / scale

    print(Apx2sdf)

    sdf = infer_from_latent_affine_mtx(
        model_path,
        config_file,
        latent_vec,
        sdf_size=sdf_size,
        batch_size=batch_size,
        affine_mtx=affine_mtx,
        Apx2sdf=Apx2sdf,
    )

    sdf = np.swapaxes(sdf, 0, 2)
    new_im = sitk.GetImageFromArray(sdf)
    new_im.SetSpacing(ref_im.GetSpacing())
    new_im.SetDirection(ref_im.GetDirection())
    new_im.SetOrigin(ref_im.GetOrigin())
    sitk.WriteImage(new_im, save_loc)


def infer_aligned_sdf(
    model_path,
    config_file,
    latent_idx,
    save_loc,
    ref_im,
    trans,
    scale,
    batch_size=500000,
):

    loaded = torch.load(model_path)
    latent_vecs = loaded["component.latent_vector"]["weight"]

    ref_im = sitk.ReadImage(ref_im)
    sdf_size = ref_im.GetSize()

    latent_vec = latent_vecs[latent_idx, :]

    sdf = infer_from_latent(
        model_path,
        config_file,
        latent_vec,
        sdf_size=sdf_size,
        batch_size=batch_size,
        trans=trans,
        scale=scale,
        normalized=False,
    )

    sdf = np.swapaxes(sdf, 0, 2)
    new_im = sitk.GetImageFromArray(sdf)
    new_im.SetSpacing(ref_im.GetSpacing())
    new_im.SetDirection(ref_im.GetDirection())
    new_im.SetOrigin(ref_im.GetOrigin())
    sitk.WriteImage(new_im, save_loc)


def _get_pca(model_path):
    loaded = torch.load(model_path)
    latent_vecs = loaded["component.latent_vector"]["weight"]
    latent_vecs = latent_vecs.numpy()
    pca = PCA(0.95)
    pca.fit(latent_vecs)

    return pca


def conduct_pca(model_path, config_file, out_folder, sdf_size=300, batch_size=500000):
    loaded = torch.load(model_path)
    latent_vecs = loaded["component.latent_vector"]["weight"]
    latent_vecs = latent_vecs.numpy()
    pca = PCA(0.95)
    pca.fit(latent_vecs)

    components = pca.components_
    for i in range(3):
        for scale in [-1, 1]:

            cur_vec = pca.mean_ + scale * components[i, :]

            sdf = infer_from_latent(
                model_path,
                config_file,
                cur_vec,
                sdf_size=sdf_size,
                batch_size=batch_size,
            )

            vertices, triangles, _, _ = measure.marching_cubes(
                sdf, 0, allow_degenerate=False, method="lewiner"
            )

            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            mesh = scale_mesh(mesh)  # added for comparison to mean shape, gasperp
            save_loc = os.path.join(out_folder, str(i) + "_" + str(scale) + ".obj")
            mesh.export(save_loc)


def interpolate_mesh(
    model_path,
    config_file,
    latent_one,
    latent_two,
    save_loc_root,
    sdf_size=300,
    batch_size=50000,
    steps=10,
):
    """given the indices of the two latent vectors from training set, this function can create interpolated meshes starting from first shape and moving to second shape

    Args:
        model_path (_type_): _description_
        config_file (_type_): _description_
        latent_one (_type_): _description_
        latent_two (_type_): _description_
        save_loc_root (_type_): _description_
        sdf_size (int, optional): _description_. Defaults to 300.
        batch_size (int, optional): _description_. Defaults to 50000.
        steps (int, optional): _description_. Defaults to 10.
    """

    loaded = torch.load(model_path)
    latent_vecs = loaded["component.latent_vector"]["weight"]

    latent_vec_one = latent_vecs[latent_one, :]
    latent_vec_two = latent_vecs[latent_two, :]

    diff = latent_vec_two - latent_vec_one

    for i in range(steps + 1):
        step_size = diff * i / steps
        cur_latent_vec = latent_vec_one + step_size

        sdf = infer_from_latent(
            model_path,
            config_file,
            cur_latent_vec,
            sdf_size=sdf_size,
            batch_size=batch_size,
        )

        vertices, triangles, _, _ = measure.marching_cubes(
            sdf, 0, allow_degenerate=False, method="lewiner"
        )

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.export(save_loc_root + "_" + str(i) + ".obj")


def interpolate_mesh_given_vectors(
    model_path,
    config_file,
    first_latent_vec,
    second_latent_vec,
    save_loc_root,
    sdf_size=300,
    batch_size=50000,
    steps=10,
):
    """given the two **vectors**, this function can create interpolated meshes starting from first shape and moving to second shape

    Args:
        model_path (_type_): _description_
        config_file (_type_): _description_
        latent_one (_type_): _description_
        latent_two (_type_): _description_
        save_loc_root (_type_): _description_
        sdf_size (int, optional): _description_. Defaults to 300.
        batch_size (int, optional): _description_. Defaults to 50000.
        steps (int, optional): _description_. Defaults to 10.
    """

    latent_vec_one = first_latent_vec
    latent_vec_two = second_latent_vec

    diff = latent_vec_two - latent_vec_one

    for i in range(steps + 1):
        step_size = diff * i / steps
        cur_latent_vec = latent_vec_one + step_size

        sdf = infer_from_latent(
            model_path,
            config_file,
            cur_latent_vec,
            sdf_size=sdf_size,
            batch_size=batch_size,
        )

        vertices, triangles, _, _ = measure.marching_cubes(
            sdf, 0, allow_degenerate=False, method="lewiner"
        )

        create_dir_for_file(save_loc_root)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh = scale_mesh(mesh)
        mesh.export(save_loc_root + "_" + str(i).zfill(3) + ".obj")


def infer_mesh(
    model_path, config_file, latent_idx, save_loc, sdf_size=300, batch_size=500000
):

    loaded = torch.load(model_path)
    latent_vecs = loaded["component.latent_vector"]["weight"]

    latent_vec = latent_vecs[latent_idx, :]

    sdf = infer_from_latent(
        model_path, config_file, latent_vec, sdf_size=sdf_size, batch_size=batch_size
    )

    vertices, triangles, _, _ = measure.marching_cubes(sdf, 0, allow_degenerate=False)

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh = scale_mesh(mesh)

    mesh.export(save_loc)


# def generate_mask_training_samples(in_folder, out_size, anchor_mesh):
