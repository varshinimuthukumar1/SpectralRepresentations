import numpy as np
from scipy.sparse.linalg import eigsh
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm

import datetime

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

    print('WRITING BASIS AND EIGENVALUES(FREQUENCIES) TO FILE..............')
    eigen_txt = 'eigen_' +str(t) + str(avg_size) + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))+ '.txt'
    basis_txt = 'basis_' +str(t) + str(avg_size) + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.txt'

    np.savetxt(eigen_txt, freq, fmt='%.10f')
    np.savetxt('eigen.txt', freq, fmt='%.10f')
    np.savetxt(basis_txt, basis, fmt='%.10f')
    np.savetxt('basis.txt', basis, fmt='%.10f')

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