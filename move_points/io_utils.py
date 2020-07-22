import open3d as o3d
import numpy as np

def read_pointcloud(pointcloud, normals = False):

    if pointcloud.endswith('.ply') or pointcloud.endswith('.xyz'):
        ptc = o3d.io.read_point_cloud(pointcloud)

    if normals:
        return np.asarray(ptc.points), np.asarray(ptc.normals)
    return np.asarray(ptc.points)

def read_2Dplane(pointcloud):

    if pointcloud.endswith('.txt'):
        ptc = np.loadtxt(pointcloud)
        temp = np.zeros(ptc.shape[0])
        points3D = np.append(ptc, temp.reshape(-1, 1), axis=1)

    return ptc, points3D

