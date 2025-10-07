###############################################
'''
Align shapes in local folder based on 
Auto3dgm
'''
###############################################

import os
import numpy as np
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
        

##### user parameters
def process():
    data_path = '116_teeth_data/'
    print(data_path)
    output_dir = 'out/'
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    low_res = 400

    #### loading data into "data_col"
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    orig_meshes, names = load_meshes(data_path, center_scale=True)
    print('data loaded: num of shapes = ', str(len(orig_meshes)))
    
    sub_meshes = subsample_meshes(orig_meshes, low_res)
    
    mesh1 = sub_meshes[0]
    for i in range(1,len(sub_meshes)):
        mesh2 = sub_meshes[i]
        orig = orig_meshes[i]
        aa = locgpd(mesh1, mesh2, R_0=None, M_0=None, max_iter=1000, mirror=False)
        v2T = aa['r'] @ orig.vertices.T
        
        aligned_mesh = (aa['r'] @ mesh2.vertices.T).T[aa['p']] # S2' = P @ S2 @ R
        
        # calculate pairwise distance using exp(-d_ij^2/eps)
        
        d = np.linalg.norm(mesh1.vertices - aligned_mesh, axis=1)

        eps = aa['g']**4
        k = np.exp(-d**2 / eps)
        np.set_printoptions(threshold=10000, linewidth=30)
        print(k)
        exit()
        
        # if np.linalg.det(aa['r']) < 0: # to make the orientation of the faces correct
        #     orig.faces=orig.faces[:,[0,2,1]]
        # aligned_mesh = trimesh.Trimesh(vertices=v2T.T, faces=orig.faces, process=False)
        # name = names[i]
        # aligned_mesh.export(output_dir + name)
        
        

    print("Aligned meshes saved \n" )
    print('Completed')
    return 0

if __name__ == "__main__":
    process()