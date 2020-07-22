import open3d as o3d
import numpy as np
import math

def get_weights(dist):
    w1 = 1/dist[1]
    w2 = 1/dist[2]
    w3 = 1/dist[3]

    wv1 = w1 / (w1+w2+w3)
    wv2 = w2 / (w1 + w2 + w3)
    wv3 = w3 / (w1 + w2 + w3)

    return wv1, wv2, wv3


def get_spectrum(basis, pcloud_name, sampled_points):

    # append ptcloud to sample points and create a new point cloud
    ptcloud = o3d.io.read_point_cloud('sphere_fibo_10k.ply')
    print(basis.shape)
    n_points = sampled_points.shape[0]


    # get nearest points
    nneigh = 4
    k_list = []
    idx_list = []
    n_list = []

    basis = np.asarray(basis)
    power = np.zeros(basis.shape[1])
    spect = np.zeros(basis.shape[1])
    for i in np.arange(sampled_points.shape[0]):


        p = np.asarray(sampled_points)[i]
        pcl_points = np.append(sampled_points[i].reshape((1, 3)), np.asarray(ptcloud.points), axis= 0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl_points)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)


        [k, idx, _] = pcd_tree.search_knn_vector_3d(sampled_points[i], nneigh)

        neighbours = np.asarray(pcd.points)[idx, :]
        point = sampled_points[i]
        #point = np.tile(point, (neighbours.shape[0], 1))

        v = neighbours - point

        dist = np.linalg.norm(v, axis=1)


        # get weights
        w1, w2, w3 = get_weights(dist)
        print(w1,w2,w3)


        spect = spect + (((w1 * basis[idx[1]-1,:] )+ (w2 * basis[idx[2]-1,:] ) + (w3 * basis[idx[3]-1,:]))/3)

    power = np.abs(spect) ** 2

    return power
