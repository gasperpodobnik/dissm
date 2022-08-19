import trimesh
import numpy as np
import plotly.graph_objects as go


def fitPlaneLTSQ(XYZ):
    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  #X
    G[:, 1] = XYZ[:, 1]  #Y
    Z = XYZ[:, 2]
    (a, b, c),resid,rank,s = np.linalg.lstsq(G, Z)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return np.array([0.0, 0.0, c]), normal


def compare_indices(a, b):
    # check each column if any indices are the same
    matrix_of_same_points = np.stack(
        [np.in1d(a[:, 0], b[:, 0]), np.in1d(a[:, 1], b[:, 1]), np.in1d(a[:, 2], b[:, 2])]).T
    # Sum those indices and check, where sum is 3, then squeeze to get ndarray!
    out = ((np.sum(matrix_of_same_points, axis=1) == 3) == 1)*1
    return out
directory = '047'
rootdir = '/media/BigDataMama/Data/aneuryms_neck_100_meshes/100-aneurysm-meshes'
rootdir2 = '/media/BigDataMama/Data/aneuryms_neck_100_meshes/aneurysm_neck_detection_results_cascade_folds'
rootdir2 = '/media/BigDataMama/Data/aneuryms_neck_100_meshes/old_files/cascade_test_fold_2/60'

main_path = rootdir + '/' + directory
main_path_pred = rootdir2 + '/' + directory
path = main_path+'/aneurysm.stl'
path_dome = main_path + '/aneurysm_dome.stl'
mesh = trimesh.load(path)
mesh_dome = trimesh.load(path_dome)
pred_gold = compare_indices(np.array(mesh.vertices), np.array(mesh_dome.vertices))
true_segmentation = np.c_[np.array(mesh.vertices), pred_gold]
faces=np.array(mesh.faces)
#pred_segmentation = np.loadtxt('/media/BigDataMama/Data/aneuryms_neck_100_meshes/aneurysm_neck_detection_results_fold_1/031')
pred_segmentation = np.loadtxt(rootdir2 + '/aneurysm_' + str(f"{int(directory[1:3])-1:02}") + '.txt' )
# preverimo koliko točk dejansko najde procentualno pravilno pri različnih thresholdih

intensitiys = ((pred_segmentation[:,3][faces[:,0]]
                +pred_segmentation[:,3][faces[:,1]]
                +pred_segmentation[:,3][faces[:,2]])/3)

data=go.Mesh3d(
        # 8 vertices of a cube
        x=pred_segmentation[:, 0],
        y=pred_segmentation[:, 1],
        z=pred_segmentation[:, 2],
        #colorbar_title='z',

        colorscale=[[0.0, "rgb(170,170,170)"],
                   [1.0, "rgb(72,144,255)"]],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity = intensitiys,
        intensitymode='cell',
        color='grey',
        # i, j and k give the vertices of triangles
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name='y',
        showscale=True
    )


"""

data=go.Scatter3d(
        # 8 vertices of a cube
        x=pred_segmentation[:, 0],
        y=pred_segmentation[:, 1],
        z=pred_segmentation[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=np.abs(pred_segmentation[:,3]-0.5)>0.25,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=1
        ),

    )



rootdir3 = '/media/BigDataMama/Data/aneuryms_neck_100_meshes/'
normala = np.loadtxt(rootdir3 + 'plains_from_jerman/normals/normal' + directory[1:] + '.txt')
tocka = np.loadtxt(rootdir3 + 'plains_from_jerman/points/point' + directory[1:] + '.txt')

# x = np.linspace(10, 15, 2) 0009
#y = np.linspace(1.3,1.45 , 2)

x = np.linspace(-5.2, 2, 2)
y = np.linspace(-1.4, 0, 2)

X, Y = np.meshgrid(x, y)
Z = -(normala[0]*(X-tocka[0]) + normala[1]*(Y-tocka[1]) - normala[2]*tocka[2] ) / normala[2]


over_b = np.zeros(np.shape(pred_segmentation[:, 0]))
over_b2 = np.zeros(np.shape(pred_segmentation[:, 0]))
pred_jerman = np.zeros(np.shape(pred_segmentation[:, 0]))
tocka_b, normala_b = fitPlaneLTSQ(np.squeeze(pred_segmentation[np.where((pred_segmentation[:,3]>0)*(pred_segmentation[:,3]<1)),0:3]))

for i in range(np.shape(pred_segmentation)[0]):

    over_b[i] = (np.dot(tocka_b - pred_segmentation[i, 0:3], normala_b) < 0) * 1
    over_b2[i] = (np.dot(tocka_b - pred_segmentation[i, 0:3], -normala_b) < 0) * 1

pred_bizjak_plane = over_b
pred_bizjak_plane2 = over_b2
if np.sum(pred_gold * pred_bizjak_plane) < np.sum(pred_gold * pred_bizjak_plane2):
    normala_b = -normala_b


X1, Y1 = np.meshgrid(x, y)
Z1 = -(normala_b[0]*(X1-tocka_b[0]) + normala_b[1]*(Y1-tocka_b[1]) - normala_b[2]*tocka_b[2] ) / normala_b[2]



data_2=go.Mesh3d(
        # 8 vertices of a cube
        x=[X[0,0], X[0,1], X[1,0], X[1,1]],
        y=[Y[0,0], Y[0,1], Y[1,0], Y[1,1]],
        z=[Z[0,0], Z[0,1], Z[1,0], Z[1,1]],
        intensitymode='cell',
        # i, j and k give the vertices of triangles
        i=[0, 0],
        j=[1, 2],
        k=[3, 3],
        opacity= 0.6,
        color = 'black'

    )


data_3=go.Mesh3d(
        # 8 vertices of a cube
        x=[X1[0,0], X1[0,1], X1[1,0], X1[1,1]],
        y=[Y1[0,0], Y1[0,1], Y1[1,0], Y1[1,1]],
        z=[Z1[0,0], Z1[0,1], Z1[1,0], Z1[1,1]],
        intensitymode='cell',
        # i, j and k give the vertices of triangles
        i=[0, 0],
        j=[1, 2],
        k=[3, 3],
        opacity= 0.6,
        color = 'darkorange'

    )
    # -----------------------------------------------------------------
# corner points:
possible_neighbors = mesh.vertex_neighbors[np.where(pred_gold)]
corner_points = []
for i in range(np.size(possible_neighbors)):
    if np.sum(np.abs(pred_gold-1)[np.array(possible_neighbors[i])])>0:
        corner_points.append(np.where(pred_gold)[0][i])
data3=go.Scatter3d(
    x=pred_segmentation[corner_points, 0],
    y=pred_segmentation[corner_points, 1],
    z=pred_segmentation[corner_points, 2],
    mode='markers',
    marker=dict(
        size=4,               # set color to an array/list of desired values
        opacity=1,
        color='red'
    )
)

    
"""

fig = go.Figure(data=[data])

fig.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(255, 255, 255)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",
                         visible=False),
                    yaxis = dict(
                        backgroundcolor="rgb(255, 255,255)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",
                        visible=False),
                    zaxis = dict(
                        backgroundcolor="rgb(255, 255,255)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",
                        visible=False),),
                    margin=dict(
                    r=10, l=10,
                    b=10, t=10)
                  )
fig.show()
