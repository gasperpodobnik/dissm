import io
import os
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import trimesh
from implicitshapes.mesh_utils import scale_mesh
from PIL import Image
from sklearn.decomposition import PCA
from utilities import utilities
from tqdm import tqdm


def pcd_to_mesh(vertices_np, k_smooth_normals=50, depth=9):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices_np)
    # pcd.estimate_normals()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k_smooth_normals)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )

    mesh_trimesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)
    )
    return mesh_trimesh


# pcd to mesh with ball pivoting - not as good as poisson
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(ipcd)

# distances = pcd.compute_nearest_neighbor_distance()
# avg_dist = np.mean(distances)
# radius = 3 * avg_dist
# radii = [radius, radius * 2]
# pcd.estimate_normals()
# pcd.orient_normals_consistent_tangent_plane(50)
# bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd, o3d.utility.DoubleVector(radii)
# )


def scale_all_pcds(pcd_list, centroid=None, max_dist=None):
    if centroid is None:
        centroid = np.array([pcd.bounding_box.centroid for pcd in pcd_list]).mean(
            axis=0
        )
    vertices_list = [pcd.vertices - centroid for pcd in pcd_list]
    if max_dist is None:
        max_dist = max(
            [np.linalg.norm(vertices, axis=1).max() for vertices in vertices_list]
        )
    return [trimesh.PointCloud(vertices / max_dist) for vertices in vertices_list]


def scale_pcd(pcd, centroid=None, max_dist=None):
    if centroid is not None:
        vertices = pcd.vertices - centroid
    else:
        vertices = pcd.vertices - pcd.bounding_box.centroid
    if max_dist is None:
        max_dist = np.linalg.norm(vertices, axis=1).max()
    return trimesh.PointCloud(vertices / max_dist)


def mesh_2_volume(mesh, spacing=1 / 300):
    """assumes mesh is normalized to square [-1,1]

    Args:
        mesh (_type_): _description_
        spacing (_type_, optional): _description_. Defaults to 1/300.

    Returns:
        _type_: _description_
    """

    spacing = tuple([spacing] * 3)
    voxelized = mesh.voxelized(pitch=spacing[0])
    voxelized.fill()

    sitk_img = sitk.GetImageFromArray(np.array(voxelized.matrix, dtype=float))
    sitk_img.SetOrigin(voxelized.origin[::-1])
    sitk_img.SetSpacing(spacing)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin((-1, -1, -1))
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize((np.array([2, 2, 2]) / spacing[0] + 1).astype(int).tolist())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return sitk.GetArrayFromImage(resampler.Execute(sitk_img))


def get_mean_mesh(meshes_list, mean_sitk_path):
    from implicitshapes import convert

    spacing = 1 / 50
    np_array = mesh_2_volume(meshes_list[0], spacing)
    for mesh in meshes_list[1:]:
        np_array += mesh_2_volume(mesh, spacing)
    np_array /= len(meshes_list)
    sitk_image = sitk.GetImageFromArray(
        (np.swapaxes(np_array, 0, 2) > 0.5).astype(float)
    )
    sitk.WriteImage(sitk_image, mean_sitk_path)
    mesh_path = mean_sitk_path + ".obj"
    convert.convert_binary2mesh(mean_sitk_path, mesh_path)
    mesh_path = mesh_path + "_scaled.obj"
    scale_mesh(trimesh.load(mesh_path)).export(mesh_path)
    QBIN_PATH = "/tmp/Fast-Quadric-Mesh-Simplification/bin.Linux/simplify"
    os.system(
        QBIN_PATH
        + " "
        + mesh_path
        + " "
        + mesh_path
        + "_simplified.obj"
        + " "
        + str(0.1)
    )
    os.system(
        QBIN_PATH
        + " "
        + mesh_path
        + " "
        + mesh_path
        + "_simplified_super.obj"
        + " "
        + str(0.01)
    )


from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh

# mesh_path = '/media/medical/projects/head_and_neck/onkoi_2019/dissm/test/Parotid_L_interp_mesh_simplified/case_43_OAR_Parotid_L.obj'
# mesh = trimesh.load(mesh_path)


def plot_mesh(
    mesh,
    show=True,
    save_filepath=None,
    lighting_kwargs=dict(
        ambient=0.6, diffuse=0.7, roughness=0.3, specular=0.9, fresnel=0.5
    ),
    lightposition_kwargs=dict(x=-100, y=-100, z=5),
    eye=dict(x=-1.25, y=-1.25, z=1.25),
    title_text="",
):
    data = go.Mesh3d(
        # 8 vertices of a cube
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        # colorbar_title='z',
        colorscale=[[0.0, "rgb(170,170,170)"], [1.0, "rgb(72,144,255)"]],
        # Intensity of each vertex, which will be interpolated and color-coded
        # intensity = intensitiys,
        intensitymode="cell",
        color="grey",
        # i, j and k give the vertices of triangles
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        name="y",
        showscale=True,
        lighting=lighting_kwargs,
        lightposition=lightposition_kwargs,
    )

    fig = go.Figure(data=[data])

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=eye,
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgb(255, 255, 255)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                visible=False,
            ),
            yaxis=dict(
                backgroundcolor="rgb(255, 255,255)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                visible=False,
            ),
            zaxis=dict(
                backgroundcolor="rgb(255, 255,255)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                visible=False,
            ),
        ),
        margin=dict(r=10, l=10, b=10, t=10),
        scene_camera=camera,
        title_text=title_text,
    )
    if show:
        fig.show()

    if save_filepath is not None:
        os.makedirs(Path(save_filepath).parent, exist_ok=True)
        fig.write_image(save_filepath)

    # convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return img


def save_to_gif(imgs_list, fp_out, duration=200):
    imgs_list = imgs_list + imgs_list[::-1][1:-1]
    imgs_list[0].save(
        fp=fp_out,
        format="GIF",
        append_images=imgs_list[1:],
        save_all=True,
        duration=duration,
        loop=0,
    )


def plot_meshes(mesh_list, title_list=None):
    if title_list is None:
        title_list = [""] * len(mesh_list)
    pil_imgs = []
    for mesh, title in zip(mesh_list, title_list):
        pil_img = plot_mesh(
            mesh,
            show=False,
            save_filepath=None,
            eye=dict(x=-0.5, y=-0.5, z=1.5),
            lightposition_kwargs=dict(x=-100, y=-100, z=0),
            title_text=title,
        )
        pil_imgs.append(pil_img)
    return pil_imgs


class load_pca_json:
    def __init__(self, pca_json_path=None) -> None:
        pca_json_dict = utilities.read_dict_in_json(pca_json_path)
        self.n_components = pca_json_dict["n_components"]
        self.explained_variance_ratio_ = np.array(
            pca_json_dict["explained_variance_ratio_"]
        )
        self.explained_variance_ = np.array(pca_json_dict["explained_variance_"])
        self.components_ = np.array(pca_json_dict["components_"])
        self.mean_ = np.array(pca_json_dict["mean_"])


class compute_mahalanobis:
    def __init__(self, pca=None, pca_json_path=None, take_N_components=None) -> None:
        _c = np.array([pca is None, pca_json_path is None], dtype=int)
        assert _c.sum() == 1, "specify only one input option"

        if pca is None:
            pca_json_dict = utilities.read_dict_in_json(pca_json_path)
            self.n_components = pca_json_dict["n_components"]
            self.explained_variance_ratio_ = np.array(
                pca_json_dict["explained_variance_ratio_"]
            )
            self.explained_variance_ = np.array(pca_json_dict["explained_variance_"])
            self.components_ = np.array(pca_json_dict["components_"])
            self.mean_ = np.array(pca_json_dict["mean_"]).reshape(-1, 1)
        else:
            self.n_components = pca.n_components
            self.explained_variance_ratio_ = pca.explained_variance_ratio_
            self.explained_variance_ = pca.explained_variance_
            self.components_ = pca.components_
            self.mean_ = pca.mean_.reshape(-1, 1)

        if take_N_components is None:
            # take all there is
            take_N_components = self.components_.shape[0]
        self.explained_variance_ = self.explained_variance_[:take_N_components]
        self.components_ = self.components_[:take_N_components, :]
        # prepare inversed C for Mahalanobis distance computation
        self.lambda_matrix = np.diag(self.explained_variance_)
        self.lambda_matrix_inv = np.diag(1 / self.explained_variance_)
        self.C_inv = self.components_.T @ self.lambda_matrix_inv @ self.components_

    def compute(self, x_array):
        x_array = x_array.reshape(-1, 1)

        assert np.all(
            np.array(x_array.shape) == np.array(self.mean_.shape)
        ), "wrong shape"

        diff = x_array - self.mean_
        dist_squared = diff.T @ self.C_inv @ diff

        assert np.all(
            np.array(dist_squared.shape) == np.array([1, 1])
        ), "something is wrong"
        return np.sqrt(dist_squared[0, 0])

    def approximate(self, x_array):
        x_array = x_array.reshape(-1, 1)
        diff = x_array - self.mean_

        X = self.components_.T @ np.sqrt(self.lambda_matrix)
        theta = np.linalg.inv(X.T @ X) @ X.T @ diff

        fitted = self.mean_ + X @ theta
        return fitted.reshape(3, -1).T


def compute_mahalanobis_dir(
    in_dirpath,
    dataset,
    pca_json_path,
    nnunet_results_csv_path,
    take_N_components,
    suffix="_target_with_correspondence_gpa.ply",
    scale50=False,
):
    in_fps = sorted(
        utilities.listdir_fpaths(dir_path=in_dirpath, only_files=True, suffix=suffix)
    )
    if nnunet_results_csv_path is not None:
        df = pd.read_csv(nnunet_results_csv_path)
        df.set_index(["fname", "organ_name", "metric"], inplace=True)
    else:
        df = None
    m_dist = compute_mahalanobis(
        pca_json_path=pca_json_path, take_N_components=take_N_components
    )

    results = []
    for fp in tqdm(in_fps, total=len(in_fps), desc="computing mahalanobis distance"):
        if "case_01_OAR_Parotid_L" in fp:
            continue
        # img_name = Path(fp).name.replace(suffix, "_0000")
        img_name = Path(fp).name.replace(suffix, "")
        try:
            img_name = img_name.replace("-dataset-paper", "-NONcurated")
            dice = df.xs((img_name, "Parotid_L", "volumetric_dice"))["value"]
            hd95 = df.xs((img_name, "Parotid_L", "hausdorff"))["value"]
        except:
            dice = None
            hd95 = None

        pcd = trimesh.load(fp)
        if scale50:
            pcd = scale_pcd(pcd, centroid=np.array([50.0, 50.0, 50.0]), max_dist=50.0)
        dist = m_dist.compute(pcd.vertices.flatten("F"))
        fitted = m_dist.approximate(pcd.vertices.flatten("F"))
        _ = trimesh.PointCloud(fitted).export(fp.replace(".ply", "_fitted.ply"))

        results.append(
            dict(
                path=fp,
                mahalanobis_distance=dist,
                take_N_components=take_N_components,
                dice=dice,
                hd95=hd95,
            )
        )
    out_csv = pd.DataFrame(results)
    out_csv["dataset"] = dataset
    csv_path = join(in_dirpath, "mahalanobis_distance.csv")
    out_csv.to_csv(csv_path)
    print(csv_path)
    utilities.write_dict_as_json(results, join(in_dirpath, "mahalanobis_distance.json"))


def compute_and_save_pca(pcd_fps, json_path, scale50=False):
    # scale pointclouds
    pcds = [trimesh.load(fp) for fp in pcd_fps]
    if scale50:
        pcds = scale_all_pcds(
            pcds, centroid=np.array([50.0, 50.0, 50.0]), max_dist=50.0
        )

    # 3n-element vector v1 = (x1, …, xn, y1, …, yn, z1, …, zn)
    # stack them in columns [v1, v2, v3,...]
    # shape is N_vertices x N_samples
    X = np.stack([pcd.vertices.flatten("F") for pcd in pcds], axis=1)

    # mean_X = X.mean(axis=1).reshape(3, -1).T
    # _ = trimesh.Trimesh(mean_X).export(join(out_dirpath, "mean_pca.ply"))
    # _ = new_functions.pcd_to_mesh(mean_X).export(join(out_dirpath, "mean_pca.obj"))

    n_components = 0.95
    # (n_samples, n_features)
    # pca.explained_variance_ holds true eigenvalues!!!
    pca = PCA(n_components)
    # do not forget to transpose
    pca.fit(X.T)
    pca_json_dict = dict(
        n_components=n_components,
        explained_variance_ratio_=pca.explained_variance_ratio_.tolist(),
        explained_variance_=pca.explained_variance_.tolist(),
        components_=pca.components_.tolist(),
        mean_=pca.mean_.tolist(),
    )
    # pca.explained_variance_ratio_
    # pca.explained_variance_ / pca.explained_variance_.sum()
    os.makedirs(Path(json_path).parent, exist_ok=True)
    utilities.write_dict_as_json(pca_json_dict, json_path)
    return pca


def create_gif_from_pca(
    pca_components_,
    pca_explained_variance_,
    pca_mean_,
    COMPONENT_NUM,
    SIGMA_factor,
    N_STEPS,
    SAVE_ply_and_obj,
    out_dirpath,
):
    weighted_component = pca_components_[COMPONENT_NUM, :] * np.sqrt(
        pca_explained_variance_[COMPONENT_NUM]
    )

    pil_imgs = []
    for enum, curr_sigma in enumerate(np.linspace(-1, 1, N_STEPS + 1)):
        fname = f"comp_{COMPONENT_NUM}_{str(enum).zfill(2)}"
        # v = pca_mean_.reshape(-1, 1)
        # tmp_vertices = (
        #     pca_mean_ + SIGMA_factor * curr_sigma * weighted_component
        # ).reshape(-1, 1)
        # print(mahala.compute(tmp_vertices), SIGMA_factor * curr_sigma)
        cur_vertices = (
            (pca_mean_ + SIGMA_factor * curr_sigma * weighted_component)
            .reshape(3, -1)
            .T
        )
        if curr_sigma * SIGMA_factor in np.arange(
            -SIGMA_factor, SIGMA_factor + 1, dtype=float
        ):
            # curr_sigma_title = r"$\mu"
            curr_sigma_title = "\u03bc"
            if int(curr_sigma * SIGMA_factor):
                # curr_sigma_title += r" + " + str(int(curr_sigma * SIGMA_factor)) + r"\sigma$"
                curr_sigma_title += (
                    r" + " + str(int(curr_sigma * SIGMA_factor)) + "\u03c3"
                )
        else:
            curr_sigma_title = ""

        # save pointcloud as .ply file
        if SAVE_ply_and_obj:
            _ = trimesh.Trimesh(cur_vertices).export(join(out_dirpath, fname + ".ply"))
        # convert pointcloud to mesh
        mesh = pcd_to_mesh(vertices_np=cur_vertices, k_smooth_normals=10)
        if SAVE_ply_and_obj:
            _ = mesh.export(join(out_dirpath, fname + ".obj"))
        # smooth mesh to suppress weird mesh at the frontal part of parotids
        mesh_smooth = trimesh.smoothing.filter_laplacian(mesh)
        if SAVE_ply_and_obj:
            _ = mesh_smooth.export(join(out_dirpath, fname + "_smooth.obj"))
        # plot mesh
        pil_img = plot_mesh(
            mesh_smooth,
            show=False,
            save_filepath=join(out_dirpath, fname + "_smooth.png")
            if SAVE_ply_and_obj
            else None,
            eye=dict(x=-0.5, y=-0.5, z=1.5),
            lightposition_kwargs=dict(x=-100, y=-100, z=0),
            # title_text=curr_sigma_title,
        )
        from PIL import ImageDraw, ImageFont

        width, height = pil_img.size
        unicode_font = ImageFont.truetype("DejaVuSans.ttf", 20)
        ImageDraw.Draw(pil_img).text(  # Image
            xy=(width / 2, 0.9 * height),
            text=curr_sigma_title,
            fill=(0, 0, 0),  # Coordinates  # Text  # Color
            align="center",
            font=unicode_font,
        )
        pil_imgs.append(pil_img)

    save_to_gif(
        pil_imgs,
        join(
            out_dirpath,
            f"component{COMPONENT_NUM}_sigma{SIGMA_factor}_steps{N_STEPS}.gif",
        ),
        duration=500,
    )
