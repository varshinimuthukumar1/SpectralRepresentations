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


# Read mesh
mesh1  = trimesh.load_mesh('data/2_bunny/sphere_dense.stl')

# Get L matrix, i.e. Cotangent Laplacian matrix
[L,VA] = cot_laplacian_mesh.get_cot_laplacian(mesh1.vertices, mesh1.faces)
#matA = np.diag(np.divide(1.0,np.sqrt(VA)))

matA = (VA.toarray())/mesh1.area
matL = L.toarray()

# Normalize L matrix with area, or not
#matL = matA * np.asarray(L.toarray()) * matA

#matL = np.linalg.inv(np.diag(matA)) *  np.asarray(L.toarray())

# Get frequencies and basis functins
freq,basis = cot_laplacian_mesh.get_laplacian_basis_svd(mesh1,matL,matA)


# Get spectrum
pt_cld1 = o3d.io.read_point_cloud("data/2_bunny/bunny_pointcl_poisson.ply")
pt_cld2 = o3d.io.read_point_cloud("data/1_sphere/sphere_dense_poisson2.ply")
pt_cld3 = o3d.io.read_point_cloud("data/1_sphere/sphere_dense_poisson3.ply")
pt_cld4 = o3d.io.read_point_cloud("data/1_sphere/sphere_dense_poisson4.ply")
pt_cld5 = o3d.io.read_point_cloud("data/1_sphere/sphere_dense_poisson5.ply")

spect1 = Spectral_analysis.get_spectrum(basis,VA,mesh1, pt_cld1)
#spect2 = Spectral_analysis.get_spectrum(basis,VA,mesh1, pt_cld2)
#spect3 = Spectral_analysis.get_spectrum(basis,VA,mesh1, pt_cld3)
#spect4 = Spectral_analysis.get_spectrum(basis,VA,mesh1, pt_cld4)
#spect5 = Spectral_analysis.get_spectrum(basis,VA,mesh1, pt_cld5)

#spect = (spect1 +spect2 + spect3 + spect4 + spect5) /5

#Get radial means and anisotropy

plt.plot (freq, spect1)
plt.show()
R,A = Spectral_analysis.RadialMeanAccurate(spect1,freq)
