import os
import trimesh
import softmap


def load_meshes(data_path, center_scale=True):
    meshes = []
    for entry in os.listdir(data_path):
        if os.path.isfile(os.path.join(data_path, entry)):
            m = trimesh.load(os.path.join(data_path, entry))
            meshes.append(m)
    return meshes

meshes = load_meshes("116_teeth_data")

maps = softmap.process(meshes)
