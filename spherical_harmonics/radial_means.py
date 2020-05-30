from math import pi
import numpy as np
import math as m
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import open3d as o3d


lmax = 20
avg_spect = np.zeros(lmax)
for i in range(1):

    # Get spherical harmonics
    alm = 0
    spect = np.zeros(lmax)
    N = 1024

    r = 1
    ra = np.random.sample( N)
    ra2 = np.random.sample(N)
    theta = 2*pi * ra
    phi = np.arccos(2 * ra2 - 1.0 )

    # ---------------- visualize ------------------------------------------------------------------
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # x = np.linspace(-3, 3, 401)
    # mesh_x, mesh_y = np.meshgrid(x, y )
    # z = 0

    xyz = np.zeros((np.size(x), 3))
    xyz[:, 0] = np.reshape(x, -1)
    xyz[:, 1] = np.reshape(y, -1)
    xyz[:, 2] = np.reshape(z, -1)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("random.ply", pcd)

    # ------------------------------------------------------------------------------


    for l in np.arange(lmax):
        C1 = 0
        count = 0
        if(True):
            for m in np.arange(-l, l+1):
                print(l,m)
                alm = 0

                for i in np.arange(N):
                    sphecoeff = sph_harm(m,l,  theta[i], phi[i])
                    alm += sphecoeff

                alm *=((4 * pi)/N)

                power = (N/(4* pi)) * np.abs(alm)**2

                C1 += power
                count+=1

        if l != 0:
            C1 /= count
        spect[l] = C1
    avg_spect += spect

plt.plot(avg_spect)
plt.show()