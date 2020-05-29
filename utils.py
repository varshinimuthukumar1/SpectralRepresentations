import re
import numpy as np
from scipy.spatial.transform import Rotation as R


def veclen(vectors):
    """ return L2 norm (vector length) along the last axis, for example to compute the length of an array of vectors """
    return np.sqrt(np.sum(vectors**2, axis=-1))

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