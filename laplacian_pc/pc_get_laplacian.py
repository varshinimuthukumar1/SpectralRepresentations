import open3d as o3d
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d,ConvexHull
import matplotlib.pyplot as plt
import utils
from scipy.sparse.linalg import eigsh
import spectral_analysis_pc as sa_pc
from matplotlib import cm
import datetime


def basis_freq_pc(q,b, ptcloud,t,avg_size, plot = True):

    print('STARTING EIGENDECOMPOSITION OF Lmatrix..............')
    freq, basis = eigsh(q, k=25, M=b, sigma=0.00001)
    print('EIGENDECOMPOSITION COMPLETED..............')
    #freq = -1 * freq
    #basis = -1 * basis

    idx = freq.argsort()
    freq = freq[idx]

    basis = basis[:, idx]
    basis = np.around(basis, decimals=9)



    print(freq)

    # plot = True
    # if plot == True:
    #     #freq = np.loadtxt('eigen.txt')
    #     #basis = np.loadtxt('basis.txt')
    #
    #     plt.plot((freq))
    #     plt.title("frequency")
    #     plt.show()
    #
    #
    #     basis1 = basis / np.linalg.norm(basis)
    #     emin = np.min(basis1)
    #     emax = np.max(basis1)
    #     scals = basis1[:, 2]
    #
    #     # scals = scals / np.linalg.norm(basis)
    #
    #     color = [cm.jet(x) for x in scals]
    #     color = np.asarray(color)
    #
    #     ptcloud.colors = o3d.utility.Vector3dVector(color[:, :-1])
    #     o3d.visualization.draw_geometries([ptcloud])
    #
    #     plt.plot((freq))
    #     plt.title("frequency")
    #     plt.show()

    return freq, basis

def voronoi_areas(pt_tgt, normal):

    z = [0,0,1]
    normal = normal /np.linalg.norm(normal)

    # rotate the tangent plane to align with the 2d plane
    theta = np.arccos(np.dot(normal,z))
    rot_axis = np.cross(normal, z)

    pt_xyplane, pt_xyplane3D = utils.rotate3Dplane(pt_tgt,theta, rot_axis)

    # construct voronoi cell
    v = Voronoi(pt_xyplane)

    # get areas by constructing the convex hull
    areas = np.zeros(v.points.shape[0])
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices:
            areas[i] = 0
        else:
            areas[i] = ConvexHull(v.vertices[indices]).volume

    return areas


def get_lbo_pc(ptcloud_name, mydir):

    avg_size = 0.0255#0.59999999999
    rho = 3
    eps = 2*  avg_size * rho

    # read pointcloud
    ptcloud = o3d.io.read_point_cloud(ptcloud_name)
    if len(ptcloud.points) == 0:
        print('Point cloud not read correctly')
    else :
        print('POINT CLOUD READ......')

    # initialize kd tree for the pointcloud
    pcd_tree = o3d.geometry.KDTreeFlann(ptcloud)
    k_list = []
    idx_list = []
    n_list = []

    # get neighbouring points in the point cloud and append it in a long list
    for point in ptcloud.points:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, 0.4)
        k_list.append(k)
        idx_list.append(idx)
        n_list.append(np.asarray(ptcloud.points)[idx,:])
        # the first point in each row is the point in consideration, the rest are neighbours

    shape_pts = np.asarray(ptcloud.points).shape
    pt_area = []
    avg_size_calc = []

    # for each point project the neighbours to construct voronoi diagram and get areas
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

        np.savetxt(mydir +'/'+ 'projected_points.xyz', projected_points, delimiter=' ')

        # get voronoi areas
        areas = voronoi_areas(projected_points, ptcloud.normals[i])

        pt_area.append([np.asarray(ptcloud.points)[i], areas[0]] )

        # compute average size of the pointcloud parallely
        v = np.linalg.norm(v, axis=1)
        avg_size_calc.append( np.mean(v))

    print('VORONOI AREA CALCULATION COMPLETED..............')
    avg_size = np.mean(np.asarray(avg_size_calc))

    print('AVERAGE SIZE OF POINTS DETERMINED TO BE..............',avg_size )

    #np.savetxt('areas.txt', np.asarray(pt_area), fmt='%.10f')

    # Calculate the q and b matrix using this information
    # pt_area has the area of the points in same order as point cloud,  appended to the point coordinates as well

    #avg_size = 1
    t = 0.26666**(0.48)  #avg_size *avg_size #* 0.5 # avverage distance between points in the tangent plane squared
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

    print('B and Q IS COMPUTED [L = B(-1) Q]..............')
    freq,basis = basis_freq_pc(q,b,ptcloud,t,avg_size)

    print('WRITING BASIS AND EIGENVALUES(FREQUENCIES) TO FILE..............')
    eigen_txt = mydir +'/'+ 'eigen_t_' + str(t) + '_size_' + str(avg_size) + '.txt'
    basis_txt = mydir +'/'+ 'basis_t_' + str(t) + '_size_' + str(avg_size) + '.txt'

    np.savetxt(eigen_txt, freq, fmt='%.10f')
    np.savetxt('eigen.txt', freq, fmt='%.10f')
    np.savetxt(basis_txt, basis, fmt='%.10f')
    np.savetxt('basis.txt', basis, fmt='%.10f')

    return freq,basis
