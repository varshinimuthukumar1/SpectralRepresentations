import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import h5py

def veclen(vectors):
    """ return L2 norm (vector length) along the last axis, for example to compute the length of an array of vectors """
    return np.sqrt(np.sum(vectors**2, axis=-1))

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def rotate3Dplane(pt_cld, theta, axis):

    qx = axis[0] * np.sin(theta / 2)
    qy = axis[1] * np.sin(theta / 2)
    qz = axis[2] * np.sin(theta / 2)
    qw = np.cos(theta / 2)

    rot = R.from_quat([qx, qy, qz, qw])

    point_xy = np.zeros((pt_cld.shape[0], pt_cld.shape[1]))
    point_2D = np.zeros((pt_cld.shape[0], 2))
    point_2D_xyz = np.zeros((pt_cld.shape[0], pt_cld.shape[1]))

    for i in range(pt_cld.shape[0]):
        pt = pt_cld[i,:]
        point = rot.apply(pt)
        point_xy[i] = point
        point_2D[i] = np.array([point[0], point[1]])
        point_2D_xyz[i] = np.array([point[0], point[1] ,0])

    points_new_plane = point_2D

    return points_new_plane, point_2D_xyz


def rotate3Dplane2(pt_cld, normal, point=[0,0,0]):

    # get theta and axis
    pl_normal = np.cross(pt_cld[2,:] - pt_cld[0,:], pt_cld[1,:] - pt_cld[0,:])

    if pl_normal == normalize(normal) or pl_normal == - normalize(normal):
        return pt_cld, pt_cld[:,:-1]

    axis = np.cross(pl_normal, normal)
    axis = axis/np.linalg.norm(axis)

    theta = np.arccos(np.dot(normal,pl_normal)/(np.linalg.norm(normal) * np.linalg.norm(pl_normal)))

    #
    qx = axis[0] * np.sin(theta / 2)
    qy = axis[1] * np.sin(theta / 2)
    qz = axis[2] * np.sin(theta / 2)
    qw = np.cos(theta / 2)

    rot = R.from_quat([qx, qy, qz, qw])

    point_xy = np.zeros((pt_cld.shape[0], pt_cld.shape[1]))
    point_2D = np.zeros((pt_cld.shape[0], 2))
    point_2D_xyz = np.zeros((pt_cld.shape[0], pt_cld.shape[1]))

    for i in range(pt_cld.shape[0]):
        pt = pt_cld[i,:]
        point = rot.apply(pt)
        point_xy[i] = point
        point_2D[i] = np.array([point[0], point[1]])
        point_2D_xyz[i] = np.array([point[0], point[1] ,0])

    points_new_plane = point_2D

    return points_new_plane, point_2D_xyz

def bat_rotate3Dplane2(pt_cld, normal, point=[0,0,0]):

    # get theta and axis
    pl_normal = np.cross(pt_cld[2,:] - pt_cld[0,:], pt_cld[1,:] - pt_cld[0,:])

    if pl_normal == normalize(normal) or pl_normal == - normalize(normal):
        return pt_cld, pt_cld[:,:-1]

    axis = np.cross(pl_normal, normal)
    axis = axis/np.linalg.norm(axis)

    theta = np.arccos(np.dot(normal,pl_normal)/(np.linalg.norm(normal) * np.linalg.norm(pl_normal)))

    #
    qx = axis[0] * np.sin(theta / 2)
    qy = axis[1] * np.sin(theta / 2)
    qz = axis[2] * np.sin(theta / 2)
    qw = np.cos(theta / 2)

    rot = R.from_quat([qx, qy, qz, qw])

    point_xy = np.zeros((pt_cld.shape[0], pt_cld.shape[1]))
    point_2D = np.zeros((pt_cld.shape[0], 2))
    point_2D_xyz = np.zeros((pt_cld.shape[0], pt_cld.shape[1]))

    for i in range(pt_cld.shape[0]):
        pt = pt_cld[i,:]
        point = rot.apply(pt)
        point_xy[i] = point
        point_2D[i] = np.array([point[0], point[1]])
        point_2D_xyz[i] = np.array([point[0], point[1] ,0])

    points_new_plane = point_2D

    return points_new_plane, point_2D_xyz


def read_matlab_mat(matlab_matrix, isdiagonal= False, issparase = False):

    f = h5py.File(matlab_matrix, 'r')
    print(list(f.keys))

    matrix = f
    return matrix

def cart2sph(x,y,z):

    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    elev = np.arctan2(z,np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                           # phi
    return r, elev, az