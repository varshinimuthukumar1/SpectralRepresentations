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


mesh1  = trimesh.load_mesh('3d_model_of_Sphere_poisson.stl')

#o3d.visualization.draw_geometries([mesh1])
normalmag = np.linalg.norm(mesh1.face_normals, axis=1)
[L,VA] = cot_laplacian_mesh.get_laplacian_new(mesh1.vertices, mesh1.faces)
matA = np.diag(np.divide(1.0,np.sqrt(VA)))
matL = matA * np.asarray(L.toarray()) * matA

#matL = L.toarray()
freq, basis = np.linalg.eig((matL))


idx = freq.argsort()[::-1]
freq = freq[idx]
basis = basis[:,idx]
basis = matA * basis

scals = np.asarray(basis[:,5])
scals = scals / (np.linalg.norm(scals))
#viridis = cm.get_cmap('bwr', basis.shape[0])

#colormap = viridis(scals)

#radii = np.linalg.norm(mesh1.vertices - mesh1.center_mass, axis=1)
#mesh1.visual.vertex_colors= trimesh.visual.interpolate(radii, color_map='viridis')#trimesh.visual.color.linear_color_map(normalmag) # o3d.utility.Vector3dVector(colormap[:,:-1])


vtmesh = trimesh2vtk(mesh1)
vtmesh.pointColors(scals, cmap='jet')
#o3d.visualization.draw_geometries([mesh1])
show(vtmesh, bg='w')
spect = Spectral_analysis.get_spectrum(basis,VA,mesh1)
print('1')

R,A = Spectral_analysis.get_radial_means_copy(spect)
