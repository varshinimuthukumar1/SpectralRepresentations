import open3d as o3d
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d,ConvexHull
import matplotlib.pyplot as plt
import utils
from scipy.sparse.linalg import eigsh
import spectral_analysis_pc as sa_pc
from matplotlib import cm


def voronoi_areas(pt_tgt, normal):

    z = [0,0,1]
    normal = normal /np.linalg.norm(normal)

    theta = np.arccos(np.dot(normal,z))
    rot_axis = np.cross(normal, z)
    np.savetxt('pt.xyz', pt_tgt, fmt='%.10f')

    pt_xyplane, pt_xyplane3D = utils.rotate3Dplane(pt_tgt,theta, rot_axis)
    #np.savetxt('pt_xy.xyz', pt_xyplane3D, fmt='%.10f')


    v = Voronoi(pt_xyplane) #, qhull_options=)
    areas = np.zeros(v.points.shape[0])
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices:
            areas[i] = 0
        else:
            areas[i] = ConvexHull(v.vertices[indices]).volume

    return areas


def get_laplacian_pc(ptcloud_name):

    avg_size = 0.5#0.59999999999
    rho = 3
    eps = 2*  avg_size * rho

    # read pointcloud
    ptcloud = o3d.io.read_point_cloud(ptcloud_name)
    if len(ptcloud.points) == 0:
        print('Point cloud not read correctly')
    else :
        print('POINT CLOUD READ......')

    # plot = True
    # if plot == True:
    #     freq = np.loadtxt('eigen_poisson.txt')
    #     basis = np.loadtxt('basis_poisson.txt')
    #
    #     plt.plot((freq))
    #     plt.title("frequency")
    #     plt.show()
    #
    #     basis1 = basis / np.linalg.norm(basis)
    #     emin = np.min(basis1)
    #     emax = np.max(basis1)
    #     scals = basis1[:, 2]
    #     #scals = scals / np.linalg.norm(scals)
    #
    #     # scals = scals / np.linalg.norm(basis)
    #
    #     color = [cm.jet(x) for x in scals]
    #     color = np.asarray(color)
    #
    #     #ptcloud.colors = o3d.utility.Vector3dVector(color[:, :-1])
    #     #o3d.visualization.draw_geometries([ptcloud])
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection = '3d')
    #
    #     X = np.asarray(ptcloud.points)[:,0]
    #     Y = np.asarray(ptcloud.points)[:,1]
    #     Z = np.asarray(ptcloud.points)[:,2]
    #
    #
    #     map = ax.scatter(X,Y,Z, c= scals, cmap='jet')
    #     fig.colorbar(map, ax = ax)
    #
    #     ax.set_xlabel('x axis')
    #     ax.set_ylabel('y axis')
    #     ax.set_zlabel('z axis')
    #
    #
    #     plt.show()

    # initialize kd tree for the pointcloud
    pcd_tree = o3d.geometry.KDTreeFlann(ptcloud)
    k_list = []
    idx_list = []
    n_list = []

    # get neighbouring points in the point cloud and append it in a long list
    for point in ptcloud.points:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, 0.3)
        k_list.append(k)
        idx_list.append(idx)
        n_list.append(np.asarray(ptcloud.points)[idx,:])
        # the first point in each row is the point in consideration, the rest are neighbours

    shape_pts = np.asarray(ptcloud.points).shape
    pt_area = []
    avg_size_calc = []
    # for each point use the neighbours to construct voronoi diagram and get areas
    for i in np.arange(shape_pts[0]):
        neighbours = n_list[i]
        point = np.asarray(ptcloud.points)[i]
        point = np.tile(point,(neighbours.shape[0], 1))
        normal = np.tile(ptcloud.normals[i], (neighbours.shape[0], 1))
        v = np.subtract( neighbours , point)

        dist = np.multiply(normal, v)
        dist = np.sum(dist, axis=1)

        k = dist[:, None]
        projected_points = neighbours - np.multiply(normal , dist[:, None])

        np.savetxt('strin.xyz', projected_points, delimiter=' ')

        areas = voronoi_areas(projected_points, ptcloud.normals[i])

        pt_area.append([np.asarray(ptcloud.points)[i], areas[0]] )

        # compute average size of the pointcloud parallely
        v = np.linalg.norm(v, axis=1)
        avg_size_calc.append( np.mean(avg_size))

    print('VORONOI AREA CALCULATION COMPLETED..............')
    avg_size = np.mean(np.asarray(avg_size_calc))

    print('AVERAGE SIZE OF POINTS DETERMINED TO BE..............',avg_size )

    #np.savetxt('areas.txt', np.asarray(pt_area), fmt='%.10f')

    # Calculate the q and b matrix using this information
    # pt_area has the area of the points in same order as point cloud,  appended to the point coordinates as well

    #avg_size = 1
    t = 1#avg_size *avg_size #* 0.5 # avverage distance between points in the tangent plane squared
    #######


    q = np.zeros([shape_pts[0],shape_pts[0]])
    b = np.zeros([shape_pts[0],shape_pts[0]])


    for i in np.arange(shape_pts[0]):
        for j in np.arange(shape_pts[0]):
            if i == j :
                break

            e = np.exp(np.linalg.norm(ptcloud.points[i] - ptcloud.points[j]) ** 2)/ (4 * t)
            q[i][j] = ((pt_area[i][1]) * (pt_area[i][1]) * (e))/ np.pi * 4 * t * t

    for i in np.arange(shape_pts[0]):
        q[i][i] = np.sum(q[i], axis=0) * -1
        b[i][i] = pt_area[i][1]

    freq,basis = sa_pc.basis_freq_pc(q,b,ptcloud,t,avg_size)

    return
