import cot_laplacian_mesh
import Spectral_analysis
import trimesh
import meshio
import open3d as o3d
import numpy as np
import matplotlib
from vtkplotter import trimesh2vtk, show
from matplotlib import cm
from colorspacious import cspace_converter
from sklearn.preprocessing import normalize
import torch
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import spherical_harmonics
from math import pi
import numpy as np
import math as m
from scipy.special import sph_harm



# Read mesh
mesh1  = trimesh.load_mesh('data/1_sphere/unit_sphere.stl')

# Get L matrix, i.e. Cotangent Laplacian matrix
[L,VA] = cot_laplacian_mesh.get_cot_laplacian(mesh1.vertices, mesh1.faces)
#np.savetxt('L.txt', L, fmt='%.10f')
#np.savetxt('VA.txt', VA, fmt='%.10f')




#matA = np.diag(np.divide(1.0,np.sqrt(VA)))
#matA = np.linalg.inv(np.sqrt(VA))

#matL = L.toarray()
# Normalize L matrix with area, or not
#matL = matA * np.asarray(L.toarray()) * matA

#matL = np.linalg.inv(np.diag(matA)) *  np.asarray(L.toarray())

# Get frequencies and basis functins
freq,basis = cot_laplacian_mesh.get_laplacian_basis_svd(mesh1,L,VA)

trials = 10

avg_spect = np.zeros(basis.shape[1])

for i in range(trials):

    pcloud_name = "data/1_sphere/unit_sphere_random"+str(i)+".ply"
    pt_cld = o3d.io.read_point_cloud(pcloud_name)
    spect = Spectral_analysis.get_spectrum(basis, VA, mesh1, pt_cld)
    avg_spect += spect

avg_spect/= trials



#Get radial means and anisotropy

plt.plot (freq[1:], avg_spect[1:])
plt.show()
R = Spectral_analysis.get_radial_means(avg_spect,freq)
np.savetxt('spect_radial.txt', R, fmt='%.10f')
