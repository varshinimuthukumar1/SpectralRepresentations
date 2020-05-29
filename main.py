import open3d as o3d


import cot_laplacian_mesh
import Spectral_analysis
import display_results
import trimesh
# import meshio

# import numpy as np
# import matplotlib
# from vtkplotter import trimesh2vtk, show
# from matplotlib import cm
# from colorspacious import cspace_converter
# from sklearn.preprocessing import normalize
# import torch
# from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
# import spherical_harmonics
# from math import pi
import numpy as np
# import math as m
# from scipy.special import sph_harm

# importing my modules
import pc_get_laplacian as pcl
import datetime

def main_mesh():

    mesh_name = 'data/5_unitsphere_10k/unit_sphere_10242.stl'
    point_cloud = 'data/5_unitsphere_10k/unit_sphere_64'

    # Read mesh
    mesh1 = trimesh.load_mesh(mesh_name)

    # Get L matrix, i.e. Cotangent Laplacian matrix
    [L, VA] = cot_laplacian_mesh.get_cot_laplacian(mesh1.vertices, mesh1.faces)

    # np.savetxt('L.txt', str(L), fmt='%.10f')
    # np.savetxt('VA.txt', str(VA), fmt='%.10f')

    # Get frequencies and basis functions
    # L = np.loadtxt('L.txt')
    # VA = np.loadtxt('VA.txt')

    # freq,basis = cot_laplacian_mesh.get_laplacian_basis_svd(mesh1,L,VA)

    freq = np.loadtxt('eigen.txt')
    basis = np.loadtxt('basis.txt')

    trials = 5

    avg_spect = np.zeros(basis.shape[1])

    for i in range(trials):
        pcloud_name = point_cloud + str(i) + ".ply"
        pt_cld = o3d.io.read_point_cloud(pcloud_name)
        spect = Spectral_analysis.get_spectrum(basis, VA, mesh1, pt_cld)
        avg_spect += spect

    avg_spect = avg_spect / trials
    np.savetxt('spectrum.txt', avg_spect, fmt='%.10f')

    # Get radial means and anisotropy

    plt.plot(freq[1:], avg_spect[1:])
    plt.show()
    R = Spectral_analysis.get_radial_means_spherical()
    np.savetxt('radialmeans.txt', R, fmt='%.10f')

    return


def main_pointcloud():

    pcl.get_laplacian_pc('data/8_pc_sphere_2562/sphere_poisson_2900.ply')
    #display_results.show_plots('data/8_pc_sphere_2562/sphere_poisson_2900.ply')
    return


if __name__ == '__main__':
    main_pointcloud()
