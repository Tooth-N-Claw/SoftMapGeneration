from typing import Optional, Sequence
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np
from numpy.typing import NDArray
import trimesh
import lap
from numpy import linalg
from scipy.spatial import distance_matrix
import pyvista as pv
from joblib import Parallel, delayed
from itertools import product

def _generate_sign_permutations(ndim, mirror):
    """Generate all sign flip permutations for PCA alignment.

    Parameters
    ----------
    ndim : int
        Number of dimensions (2 or 3)
    mirror : bool
        Whether to include mirror reflections (all sign permutations)

    Returns
    -------
    list of np.ndarray
        List of sign permutation arrays
    """
    if mirror:
        # All possible sign combinations: 2^ndim permutations
        return [np.array(signs) for signs in product([1, -1], repeat=ndim)]
    else:
        # Only even number of sign flips (proper rotations, no reflections)
        all_perms = [np.array(signs) for signs in product([1, -1], repeat=ndim)]
        return [p for p in all_perms if np.prod(p) == 1]

def principal_component_alignment(mesh1, mesh2, mirror):
    """Compute PCA-based alignments between two meshes.

    Parameters
    ----------
    mesh1, mesh2 : np.ndarray
        Point clouds to align (shape: n_points x n_dimensions)
    mirror : bool
        Whether to include mirror reflections

    Returns
    -------
    list of np.ndarray
        List of rotation matrices for different alignment orientations
    """
    mirror = 1
    X = mesh1.T
    Y = mesh2.T

    # Detect dimensionality from input
    ndim = X.shape[0]

    UX, DX, VX = linalg.svd(X, full_matrices=False)
    UY, DY, VY = linalg.svd(Y, full_matrices=False)

    # Generate sign permutations based on dimensionality
    P = _generate_sign_permutations(ndim, mirror)

    # Compute rotation matrices
    R = []
    for signs in P:
        R.append(np.dot(np.dot(UX, np.diag(signs)), UY.T))
    return R

def best_pairwise_PCA_alignment(mesh1, mesh2, mirror):
    R = principal_component_alignment(mesh1, mesh2, mirror)
    permutations = []
    min_cost = np.ones(len(R)) * np.inf
    for rot, i in zip(R, range(len(R))):
        cost = distance_matrix(mesh1, np.dot(rot, mesh2.T).T)**2
        trash, V2_ind, garbage = lap.lapjv(cost)
        min_cost[i] = trash
        permutations.append(V2_ind)

    best_rot_ind = np.argmin(min_cost)
    best_permutation = permutations[best_rot_ind]
    best_rot = R[best_rot_ind]
    d = min_cost[best_rot_ind]

    return best_permutation, best_rot, d


def permutation_from_rotation(mesh1, mesh2, R):
    cost = distance_matrix(mesh1, np.dot(R, mesh2.T).T)**2
    trash, V2_ind, garbage = lap.lapjv(cost)
    return V2_ind

def locgpd(mesh1, mesh2, R_0=None, M_0=None, max_iter=1000, mirror=False):
    # best_permutation and best_rot come from PCA
    if M_0 is None:
        M_0 = np.ones([len(mesh1), len(mesh1)])
    
    if R_0 is None:
        best_permutation, best_rot, d = best_pairwise_PCA_alignment(mesh1, mesh2, mirror)
    else:
        best_rot = R_0
        best_permutation = permutation_from_rotation(mesh1, mesh2, R_0)

    V1 = mesh1.T
    V2 = mesh2.T
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
    Center = np.mean(mesh, 0).reshape(1,mesh.shape[1])
    foo = np.matlib.repmat(Center, len(mesh), 1)
    mesh -= foo
    if scale != None:
        #mesh = mesh * np.sqrt(1 / mesh.area)
        mesh = mesh * (1 / np.linalg.norm(mesh, 'fro'))
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


def subsample(points, n_samples, important_indices=None):
    selected = np.zeros(n_samples, dtype=int)
    distances = np.ones(len(points)) * np.inf    
    
    # initial
    if important_indices is not None:
        selected[:len(important_indices)] = important_indices
        
        for idx in important_indices:
            dist_to_point = np.linalg.norm(points - points[idx], axis=1)
            distances = np.minimum(distances, dist_to_point)
        
        i = len(important_indices)
    else:
        selected[0] = np.random.randint(len(points))
        i = 1
    
    # pick points that is furthest away
    for i in range(i, n_samples):
        if i > 0:
            last_selected = selected[i-1]
            dist_to_last = np.linalg.norm(points - points[last_selected], axis=1)
            distances = np.minimum(distances, dist_to_last)
        
        selected[i] = np.argmax(distances)
    
    return points[selected]

# def subsample(points, n_samples, important_indices=None):

#     indices = np.random.choice(len(points), n_samples, replace=False)
#     return points[indices]


def _compute_alignment(i, j, sub_meshes):
    if i == j:
        return i, j, sub_meshes[i]
    else:
        aa = locgpd(sub_meshes[i], sub_meshes[j], R_0=None, M_0=None, max_iter=1000, mirror=False)
        return i, j, (aa['r'] @ sub_meshes[j].T).T[aa['p']]


def _compute_maps_for_i(i, aligned, knn, eps):
    n = len(aligned)
    nn = NearestNeighbors(
        n_neighbors=knn, algorithm="auto", metric="euclidean", n_jobs=-1
    )
    nn.fit(aligned[i])

    maps_row = []
    for j in range(n):
        distances = nn.kneighbors_graph(aligned[j], mode="distance")

        kernel = compute_kernel(distances, eps)
        kernel.eliminate_zeros()

        # normalize rows
        row_sums = np.array(kernel.sum(axis=1)).flatten()
        row_indices, _ = kernel.nonzero()
        kernel.data /= row_sums[row_indices]

        maps_row.append(kernel)

    return i, maps_row




def process(
    meshes: Sequence,
    eps: Optional[float] = 0.01,
    low_res: int = 400,
    knn: int = 10,
    center_scale: bool = True,
    n_jobs: int = -1,
    verbose: bool = True,
    landmark_indices: Optional[Sequence[int]] = None,
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
    n_jobs : int, default=-1
        Number of parallel jobs to use. -1 uses all available cores.
    verbose : bool, default=True
        Whether to print progress information
    landmark_indices : Optional[Sequence[int]], default=None
        Indices of landmark points to always include in subsampling
        Assumed to be passed in the same order as the meshes

    Returns
    -------
    NDArray[np.float64]
        NxN array of sparse soft correspondence maps between all mesh pairs
    """
    if verbose:
        verbose = 10
    else:
        verbose = 0
        
    if center_scale:
        for i, mesh in enumerate(meshes):
            meshes[i], _ = Centralize(mesh, scale=1)
            
    min_val = min([len(mesh) for mesh in meshes])
    low_res = min(low_res, min_val)
    sub_meshes = []
    for i in range(len(meshes)):
        if landmark_indices is not None:
            sub_meshes.append(subsample(meshes[i], low_res, important_indices=landmark_indices[i]))
        else:
            sub_meshes.append(subsample(meshes[i], low_res))
    sub_meshes = np.array(sub_meshes)

        
    n = len(sub_meshes)
    maps = np.array([[None for _ in range(n)] for _ in range(n)])
    aligned = np.array([None for _ in range(n)])

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_compute_alignment)(0, i, sub_meshes)
        for i in range(0,n)
    )

    for i, j, result in results:
        aligned[j] = result 
        

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_compute_maps_for_i)(i, aligned, knn, eps)
        for i in range(n)
    )

    for i, maps_row in results:
        for j in range(n):
            maps[i, j] = maps_row[j]

    return np.array(maps), aligned

if __name__ == "__main__":
    process()