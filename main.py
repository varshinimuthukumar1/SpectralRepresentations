import open3d as o3d
import argparse
import matplotlib.pyplot as plt
import datetime
import os
from scipy.linalg import eigh
import scipy

from scipy.sparse.linalg import eigsh


import display_results
import trimesh
#from scipy.sparse.linalg import eigsh
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
import math
# from scipy.special import sph_harm
from scipy.sparse import csr_matrix
from scipy.io import loadmat, savemat
from scipy.sparse import coo_matrix
import scipy.io as sio

# importing my modules
from laplacian_pc import pc_get_laplacian as pcl
from laplacian_mesh import cot_laplacian_mesh
from Spectral_analysis import Spectral_analysis, Spectral_analysis_pc
from sampler import sampler
import test_modules

#import tables
import utils
import display_results

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--sample_patch', type=str, default='dart_throwing_256.txt', help='sample patch input')
    parser.add_argument('--outdir', type=str, default='output/', help='scratch folder for output')

    # sampler input
    parser.add_argument('--datadir', type=str, default='move_points/data/plane/', help='directory to save the data generated')
    parser.add_argument('--N', type=int, default=500, help='number of realizations to be saved')
    parser.add_argument('--sampling', type=str, default='random', help='shape to sample')
    parser.add_argument('--shape', type=str, default='plane', help='shape to sample')
    parser.add_argument('--nsamples', type=int, default=256, help='number of samples')

    return parser.parse_args()


def create_result_folder():
    directory_path = os.getcwd() + '/results/'
    mydir = os.path.join(directory_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.isfile(mydir):
        os.makedirs(mydir)
    return mydir

def main_mesh():

    mesh_name = 'data/1_sphere/sphere.stl'
    point_cloud = 'data/1_sphere/sphere_dense_poisson'

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

    trials = 2

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


def main_pointcloud(mydir):

    freq,basis = pcl.get_lbo_pc('data/8_pc_sphere_2562/sphere_poisson_2900.ply',mydir)

    #display_results.show_plots('data/8_pc_sphere_2562/sphere_poisson_2900.ply')
    return

def pointcloud_charac():
    a = sampler.random_sampler(shape='sphere', r=1, num=None)
    np.savetxt('test.txt', a, delimiter=',')
    directory_path = os.getcwd() + '/data/QF_BF/BF_sphere_fibo_100k.mat'
    ntrials = 3


    # #------------------------- Read B matrix and Q matrix------------------
    # BF = np.loadtxt('BF_sphere_fibo_30k.txt')
    # BF = np.diag(BF)
    #
    # QF = np.loadtxt('QF_sphere_fibo_30k.txt')
    #
    # # subtract 1 from index as the sparse matrix is in MATLAB format
    # QF= coo_matrix((QF[:,2], (QF[:,0]-1, QF[:,1]-1)),shape= (30000,30000))
    # QF= QF.toarray()
    #
    # print(' B and Q matrix READ...')
    #
    # # -------------------------- Solve eigen decomposition -------------------
    # freq, basis = eigsh(QF, k=10000, M=BF, sigma=-0.00001)
    # #freq, basis = eigh(QF, b=BF, subset_by_index=[0, 99999])
    #
    # freq = -1 * freq
    # basis = -1 * basis
    # idx = freq.argsort()
    # freq = freq[idx]
    #
    # basis = basis[:, idx]
    # basis = np.around(basis, decimals=9)
    #
    # print(freq)
    #
    # print(' Eigendecomposition completed...')
    # np.savetxt('basis.txt', basis, delimiter=',')
    # np.savetxt('freq.txt', freq, delimiter=',')

    # ------------------------- Calculate spectrum -----------------------------

    basis = np.loadtxt(os.getcwd() +'/basis.txt', delimiter=',')
    #freq = np.loadtxt('freq.txt', delimiter=',')
    powertotal = np.zeros(basis.shape[1])

    # Run trial 500 times
    for i in range(ntrials):

        string = str(i+1) + '_sampled_poisson_100.ply'
        ptc = o3d.io.read_point_cloud(string)
        sampled_points =  np.asarray(ptc.points) #sampler.random_sampler(num= 100)

        powerspectrum  = Spectral_analysis_pc.get_spectrum(basis, 'sphere_fibo_30k.ply', sampled_points)
        np.savetxt('spectrum.txt', powerspectrum, delimiter=',')
        powertotal = powertotal + (powerspectrum)

    R = Spectral_analysis.get_radial_means_spherical()
    np.savetxt('radialmeans.txt', R, delimiter=',')

    return


if __name__ == '__main__':
    #mydir = create_result_folder()
    opt = parse_arguments()

    ################## run for point clouds

    #pointcloud_charac()
    #main_pointcloud(mydir)
    # display_results.show_radialmeans()

    ############# run for mesh
    #main_mesh()

    ########### create dataset
    #test_modules.samples_generator(opt)
    ######## fourier loss tests

    #test_modules.test_fourier_computation_tf(opt)
    #test_modules.plot_frequencies_mesh_pc()
    #test_modules.test_fourier_computation_tf(opt)
    test_modules.create_target(opt)

