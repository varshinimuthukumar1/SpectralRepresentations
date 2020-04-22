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
mesh1  = trimesh.load_mesh('data/1_sphere/sphere_dense.stl')

# Get L matrix, i.e. Cotangent Laplacian matrix
[L,VA] = cot_laplacian_mesh.get_laplacian_new(mesh1.vertices, mesh1.faces)
matA = np.diag(np.divide(1.0,np.sqrt(VA)))

# Normalize L matrix with area, or not
matL = L.toarray() #matA * np.asarray(L.toarray()) * matA


# Get frequencies and basis functins
freq,basis = cot_laplacian_mesh.get_laplacian_basis_svd(mesh1,matL,matA,True)

#Normalize basis
#basis = matA * basis



# Get spectrum
spect = Spectral_analysis.get_spectrum(basis,VA,mesh1)

#Get radial means and anisotropy
R,A = Spectral_analysis.get_radial_means(spect)
