import numpy as np
from utils import veclen
from scipy import sparse
import scipy
import torch
from vtkplotter import trimesh2vtk, show
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse.linalg import eigsh
from matplotlib import cm
import tensorflow as tf

def get_laplacian_tensor(mesh1,matL,matA, plot=True):
    matL = tf.convert_to_tensor(matL.toarray())
    matA = tf.convert_to_tensor(matA.toarray())

    #U, freq, basis = linalg.svd(matL)
    #freq, basis = eigs(A=matL, k=1000, M= matA)

    #freq, basis = scipy.linalg.eig(matL, matA)
    matA = matA/np.sum(np.sum(matA))
    #freq, basis = eigsh(matL, 100, matA)

    freq, basis = eigsh(matL, k=500, M=matA,sigma=-0.000001)



    #basis = basis * matA[None,:]
    area = np.diag(matA)
    #for i in np.arange(len(area)):
    #    basis[i,:] = area[i] * basis[i,:]

    #freq = np.sqrt((np.abs(freq)))#**2

    freq = -1 * freq
    basis = -1 * basis

    idx = freq.argsort()
    freq = freq[idx]

    basis = basis[:, idx]
    basis = np.around(basis, decimals= 9)



    #np.savetxt('text.txt', freq, fmt='%.10f')




    #basis = basis[:,::4]
    #freq = freq[::4]

    if plot == True:
        basis1 = basis / np.linalg.norm(basis)
        emin = np.min(basis1)
        emax = np.max(basis1)
        scals = basis1[:, 0]

        #scals = scals / np.linalg.norm(basis)
        #colm = cm.get_cmap('jet', [emin,emax])

        vtmesh = trimesh2vtk(mesh1)
        vtmesh.pointColors(scals, cmap='jet')

        show(vtmesh, bg='w')

        plt.plot((freq))
        plt.title("frequency")
        plt.show()


    return freq, basis


def get_laplacian_basis_svd(mesh1,matL,matA, plot=True):

    #U, freq, basis = linalg.svd(matL)
    #freq, basis = eigs(A=matL, k=1000, M= matA)

    #freq, basis = scipy.linalg.eig(matL, matA)
    matA = matA/np.sum(np.sum(matA))
    #freq, basis = eigsh(matL, 100, matA)

    freq, basis = eigsh(matL, k=1000, M=matA,sigma=-0.000001)



    #basis = basis * matA[None,:]

    #for i in np.arange(len(area)):
    #    basis[i,:] = area[i] * basis[i,:]

    #freq = np.sqrt((np.abs(freq)))#**2

    freq = -1 * freq
    basis = -1 * basis

    idx = freq.argsort()
    freq = freq[idx]

    basis = basis[:, idx]
    basis = np.around(basis, decimals= 9)


    np.savetxt('eigen.txt', freq, fmt='%.10f')




    #basis = basis[:,::4]
    #freq = freq[::4]

    if plot == True:
        basis1 = basis / np.linalg.norm(basis)
        emin = np.min(basis1)
        emax = np.max(basis1)
        scals = basis1[:,2]

        #scals = scals / np.linalg.norm(basis)
        #colm = cm.get_cmap('jet', [emin,emax])

        vtmesh = trimesh2vtk(mesh1)
        vtmesh.pointColors(scals, cmap='jet')

        show(vtmesh, bg='w')

        plt.plot((freq))
        plt.title("frequency")
        plt.show()

    return freq, basis

def get_cot_laplacian(verts, tris):

    n = len(verts)
    W_ij = np.empty(0)
    I = np.empty(0, np.int32)
    J = np.empty(0, np.int32)
    for i1, i2, i3 in [(0,1,2), (1,2,0), (2,0,1)]:
        vi1 = tris[:, i1]
        vi2 = tris[:, i2]
        vi3 = tris[:, i3]

        u = verts[vi2] - verts[vi1]
        v = verts[vi3] - verts[vi1]
        cotan = (u * v).sum(axis=1)/veclen(np.cross(u,v))

        W_ij = np.append(W_ij, 0.5*cotan)
        I = np.append(I, vi2)
        J = np.append(J, vi3)

        W_ij = np.append(W_ij, 0.5*cotan)
        I = np.append(I, vi3)
        J = np.append(J, vi2)

    L = sparse.csr_matrix((W_ij,(I,J)), shape=(n,n))
    # compute diagonal entries
    L = L - sparse.spdiags(L * np.ones(n), 0, n, n)
    L = L.tocsr()
    # area matrix
    e1 = verts[tris[:, 1]] - verts[tris[:, 0]]
    e2 = verts[tris[:, 2]] - verts[tris[:, 0]]
    n = np.cross(e1, e2)
    triangle_area = .5 * veclen(n)
    # compute per-vertex area
    vertex_area = np.zeros(len(verts))
    ta3 = triangle_area / 3
    for i in range(tris.shape[1]):
        bc = np.bincount(tris[:, i].astype(int), ta3)
        vertex_area[:len(bc)] += bc
    VA = sparse.spdiags(vertex_area, 0, len(verts), len(verts))
    return L, VA

