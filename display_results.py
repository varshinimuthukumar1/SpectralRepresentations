import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def show_plots(ptcloud_name):

    ptcloud = o3d.io.read_point_cloud(ptcloud_name)
    if len(ptcloud.points) == 0:
        print('Point cloud not read correctly')

    plot = True
    if plot == True:
        freq = np.loadtxt('eigen_0.1.txt')
        basis = np.loadtxt('basis_0.1.txt')

        plt.plot((freq))
        plt.title("frequency")
        plt.show()

        basis1 = basis / np.linalg.norm(basis)
        emin = np.min(basis1)
        emax = np.max(basis1)
        scals = basis1[:, 2]
        #scals = scals / np.linalg.norm(scals)

        # scals = scals / np.linalg.norm(basis)

        color = [cm.jet(x) for x in scals]
        color = np.asarray(color)

        #ptcloud.colors = o3d.utility.Vector3dVector(color[:, :-1])
        #o3d.visualization.draw_geometries([ptcloud])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        X = np.asarray(ptcloud.points)[:,0]
        Y = np.asarray(ptcloud.points)[:,1]
        Z = np.asarray(ptcloud.points)[:,2]


        map = ax.scatter(X,Y,Z, c= scals, cmap='jet')
        fig.colorbar(map, ax = ax)

        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')


        plt.show()

        return