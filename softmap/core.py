from typing import Optional, Sequence
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np
from numpy.typing import NDArray
import trimesh
import lap
from numpy import linalg
from scipy.spatial import distance_matrix


def principal_component_alignment(mesh1, mesh2, mirror):
    mirror = 1
    X = mesh1.vertices.T
    Y = mesh2.vertices.T
    UX, DX, VX = linalg.svd(X, full_matrices=False)
    UY, DY, VY = linalg.svd(Y, full_matrices=False)
    P=[]
    R=[]
    P.append(np.array([1, 1, 1]))
    P.append(np.array([1, -1, -1]))
    P.append(np.array([-1, -1, 1]))
    P.append(np.array([-1, 1, -1]))
    if (mirror == 1):
        P.append(np.array([-1, 1, 1]))
        P.append(np.array([1, -1, 1]))
        P.append(np.array([1, 1, -1]))
        P.append(np.array([-1, -1, -1]))
    for i in P:
        R.append(np.dot(np.dot(UX, np.diag(i)), UY.T))
    return R

def best_pairwise_PCA_alignment(mesh1, mesh2, mirror):
    R = principal_component_alignment(mesh1, mesh2, mirror)
    permutations = []
    min_cost = np.ones(len(R)) * np.inf
    for rot, i in zip(R, range(len(R))):
        cost = distance_matrix(mesh1.vertices, np.dot(rot, mesh2.vertices.T).T)**2
        trash, V2_ind, garbage = lap.lapjv(cost)
        min_cost[i] = trash
        permutations.append(V2_ind)

    best_rot_ind = np.argmin(min_cost)
    best_permutation = permutations[best_rot_ind]
    best_rot = R[best_rot_ind]
    d = min_cost[best_rot_ind]

    return best_permutation, best_rot, d


def permutation_from_rotation(mesh1, mesh2, R):
    cost = distance_matrix(mesh1.vertices, np.dot(R, mesh2.vertices.T).T)**2
    trash, V2_ind, garbage = lap.lapjv(cost)
    return V2_ind

def locgpd(mesh1, mesh2, R_0=None, M_0=None, max_iter=1000, mirror=False):
    # best_permutation and best_rot come from PCA
    if M_0 is None:
        M_0 = np.ones([len(mesh1.vertices), len(mesh1.vertices)])
    
    if R_0 is None:
        best_permutation, best_rot, d = best_pairwise_PCA_alignment(mesh1, mesh2, mirror)
    else:
        best_rot = R_0
        best_permutation = permutation_from_rotation(mesh1, mesh2, R_0)

    V1 = mesh1.vertices.T
    V2 = mesh2.vertices.T
    newV2 = V2

    i = 0
    while True:
        newV2= V2[:,best_permutation]
        err = V1 - np.dot(best_rot, newV2)

        # Do Kabsch
        cur_rot = Kabsch(V1.T, newV2.T)
        err = V1 - np.dot(cur_rot, V2)

        # Do Hungary
        cur_cost = distance_matrix(V1.T, np.dot(cur_rot, V2).T)**2
        cur_trash, cur_permutation, garbage = lap.lapjv(cur_cost)

        #if i > max_iter or cur_trash - best_trash > 0 or ((abs(cur_trash - best_trash)<1e-5*best_trash)):
        if np.linalg.norm(err) < 0.00000001 or i > max_iter or np.sum((cur_permutation - best_permutation) != 0) < 1:
            break
        # update
        best_permutation = cur_permutation
        best_rot = cur_rot
        i += 1

    d = np.sqrt(cur_trash)
    Rotate = best_rot
    Permutate = cur_permutation
    gamma = 1.5 * ltwoinf(V1 - newV2)

    return {'d': d, 'r': Rotate, 'p': Permutate, 'g': gamma}

def ltwoinf(X):
    """l2-inf norm of x, i.e.the maximum 12 norm of the columns of x"""
    d = np.sqrt(max(np.square(X).sum(axis=0)))
    return d

def Kabsch(A, B):
    assert len(A) == len(B)

    N = A.shape[0]  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.dot(AA.T, BB)
    U, S, Vt = linalg.svd(H)
    R = np.dot(U, Vt)
    return R

def Centralize(mesh, scale=None):
    Center = np.mean(mesh.vertices, 0).reshape(1,3)
    foo = np.matlib.repmat(Center, len(mesh.vertices), 1)
    mesh.vertices -= foo
    if scale != None:
        #mesh.vertices = mesh.vertices * np.sqrt(1 / mesh.area)
        mesh.vertices = mesh.vertices * (1 / np.linalg.norm(mesh.vertices, 'fro'))
    return mesh, Center

def load_meshes(data_path, center_scale=True):
    meshes = []
    names = []
    for entry in os.listdir(data_path):
        if os.path.isfile(os.path.join(data_path, entry)):
            m = trimesh.load(os.path.join(data_path, entry))
            if center_scale:
                Centralize(m, scale=1)
            names.append(entry.split('/')[0])
            meshes.append(m)
    return meshes, names

def subsample_meshes(meshes, count):
    low_res_meshes = []
    for i in range(len(meshes)):
        m = meshes[i]
        v, t = trimesh.sample.sample_surface_even(m, count, radius=None)
        lmesh = trimesh.Trimesh(vertices=v, faces=None, process=False)
        low_res_meshes.append(lmesh)
    return low_res_meshes
        

def compute_kernel(distances, eps):
    kernel = distances.copy()
    kernel.data = np.exp(-(kernel.data**2) / eps)
    kernel.setdiag(1.0)

    return kernel   

def process(
    meshes: Sequence,
    low_res: int = 400,
    knn: int = 10,
    eps: Optional[float] = None,
    center_scale: bool = True
) -> NDArray[np.float64]:
    """Generate soft correspondence maps between meshes using Auto3DGM alignment.

    Parameters
    ----------
    meshes : Sequence
        Sequence of meshes to compute soft maps between
    low_res : int, default=400
        Number of vertices to subsample each mesh to
    knn : int, default=10
        Number of nearest neighbors for kernel computation
    eps : float, optional
        Kernel bandwidth parameter. If None, computed automatically from alignment
    center_scale : bool, default=True
        Whether to center and scale meshes before processing

    Returns
    -------
    NDArray[np.float64]
        NxN array of sparse soft correspondence maps between all mesh pairs
    """
    if center_scale:
        for mesh in meshes:
            Centralize(mesh, scale=1)
    sub_meshes = subsample_meshes(meshes, low_res)


        
    n = len(sub_meshes)
    maps = np.array([[None for _ in range(n)] for _ in range(n)])
    for i in range(n):
        print(i)
        for j in range(n):
            print(j)

            aa = locgpd(sub_meshes[i], sub_meshes[j], R_0=None, M_0=None, max_iter=1000, mirror=False)
            if eps is None:
                eps = 2 * aa['g']**2
            eps += 0.0000000001
            aligned_mesh = (aa['r'] @ sub_meshes[j].vertices.T).T[aa['p']] 
            nn = NearestNeighbors(
                n_neighbors=knn, algorithm="auto", metric="euclidean", n_jobs=-1
            )
            nn.fit(sub_meshes[i].vertices)
            distances = nn.kneighbors_graph(aligned_mesh, mode="distance")
            maps[i,j] = compute_kernel(distances, eps)
    return np.array(maps)


if __name__ == "__main__":
    process()