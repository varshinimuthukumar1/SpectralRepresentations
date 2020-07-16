import open3d as o3d
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d,ConvexHull
import matplotlib.pyplot as plt
import utils
from scipy.sparse.linalg import eigsh
from Spectral_analysis import Spectral_analysis_pc as sa_pc
from matplotlib import cm
import datetime
from matplotlib.colors import LogNorm
from scipy import fftpack


def plot_spectrum(im_fft):

    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()



def get_avg_size(ptcloud, n):

    pcd_tree = o3d.geometry.KDTreeFlann(ptcloud)
    k_list = []
    idx_list = []
    n_list = []

    for point in ptcloud.points:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, n)
        k_list.append(k)
        idx_list.append(idx)
        n_list.append(np.asarray(ptcloud.points)[idx, :])

    shape_pts = np.asarray(ptcloud.points).shape
    avg_size_calc = []
    for i in np.arange(shape_pts[0]):
        neighbours = n_list[i]
        point = np.asarray(ptcloud.points)[i]
        point = np.tile(point,(neighbours.shape[0], 1))
        v = np.subtract(neighbours, point)
        v = np.linalg.norm(v, axis=1)
        avg_size_calc.append(np.mean(v))

    avg_size = np.mean(np.asarray(avg_size_calc))

    return avg_size
        


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

def voronoi_areas(pt_tgt, normal, plot = False):

    z = [0,0,1]
    if np.asarray(normal).all() == np.asarray(z).all():
        pt_xyplane = np.asarray(pt_tgt)[:,0:-1]
        pt_xyplane3D = np.asarray(pt_tgt)

    else:
        normal = normal /np.linalg.norm(normal)

        # rotate the tangent plane to align with the 2d plane
        theta = np.arccos(np.dot(normal,z))
        rot_axis = np.cross(normal, z)

        pt_xyplane, pt_xyplane3D = utils.rotate3Dplane(pt_tgt,theta, rot_axis)

    im = np.zeros((np.max(pt_xyplane),np.max(pt_xyplane) ))
    row_indices = pt_xyplane[:, 0]
    col_indices = pt_xyplane[:, 1]
    im[row_indices, col_indices] = 1
    im_fft = fftpack.fft2(im)

    plt.figure()
    plot_spectrum(im_fft)
    plt.title('Fourier transform')

    # construct voronoi cell
    v = Voronoi(pt_xyplane)
    if plot:
        fig = voronoi_plot_2d(v)
        plt.show()

    # get areas by constructing the convex hull
    areas = np.zeros(v.points.shape[0])
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices:
            areas[i] = 0
        else:
            areas[i] = ConvexHull(v.vertices[indices]).volume



    return areas #/np.sum(areas)


def get_lbo_pc(ptcloud_name, mydir):



    # read pointcloud
    ptcloud = o3d.io.read_point_cloud(ptcloud_name)
    if len(ptcloud.points) == 0:
        print('Point cloud not read correctly')
    else :
        print('POINT CLOUD READ......')

    # Determine average size of the pointcloud

    avg_size = get_avg_size(ptcloud, 10)  # 0.59999999999
    print(avg_size)
    rho = 3
    eps = 2 * avg_size * rho

    # initialize kd tree for the pointcloud
    pcd_tree = o3d.geometry.KDTreeFlann(ptcloud)
    k_list = []
    idx_list = []
    n_list = []

    # get neighbouring points in the point cloud and append it in a long list
    for point in ptcloud.points:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, eps)
        k_list.append(k)
        idx_list.append(idx)
        n_list.append(np.asarray(ptcloud.points)[idx,:])
        # the first point in each row is the point in consideration, the rest are neighbours

    shape_pts = np.asarray(ptcloud.points).shape
    pt_area = []
    avg_size_calc = []
    a_debug = []

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

        #np.savetxt(mydir +'/'+ str(i) +'.xyz', projected_points, delimiter=' ')

        # get voronoi areas
        #  print(i)
        areas = voronoi_areas(projected_points, ptcloud.normals[i])

        pt_area.append([np.asarray(ptcloud.points)[i], areas[0]] )
        a_debug.append(areas[0])

        # compute average size of the pointcloud parallely
        v = np.linalg.norm(v, axis=1)
        avg_size_calc.append( np.mean(v))

    print('VORONOI AREA CALCULATION COMPLETED..............')
    #print(a_debug)
    #print(np.min(np.asarray(a_debug)))
    #print(np.max(np.asarray(a_debug)))
    #print(np.sum(np.asarray(a_debug)))
    #avg_size = np.mean(np.asarray(avg_size_calc))

    print('AVERAGE SIZE OF POINTS DETERMINED TO BE..............',avg_size )

    #np.savetxt('areas.txt', np.asarray(pt_area), fmt='%.10f')

    # Calculate the q and b matrix using this information
    # pt_area has the area of the points in same order as point cloud,  appended to the point coordinates as well

    #avg_size = 1
    t = avg_size**2 #avg_size *avg_size #* 0.5 # avverage distance between points in the tangent plane squared
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
    eigen_txt = mydir +'/'+ 'eigen_t_' + str(round(t,4)) + '_size_' + str(round(avg_size,4)) + '.txt'
    basis_txt = mydir +'/'+ 'basis_t_' + str(round(t,4)) + '_size_' + str(round(avg_size,4)) + '.txt'

    np.savetxt(eigen_txt, freq, fmt='%.10f')
    np.savetxt('eigen.txt', freq, fmt='%.10f')
    np.savetxt(basis_txt, basis, fmt='%.10f')
    np.savetxt('basis.txt', basis, fmt='%.10f')

    return freq,basis
