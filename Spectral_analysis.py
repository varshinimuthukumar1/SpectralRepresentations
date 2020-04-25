import numpy as np
import matplotlib.pyplot as plt
import math
import trimesh.proximity
import trimesh
import statistics
import open3d as o3d
from scipy.spatial import distance

def get_weights_frompaper():
    valence = 6
    cos2pin = math.cos(2.0 * math.pi / valence)
    beta = (5.0 / 8.0 - (3.0 / 8.0 + 1.0 / 4.0 * cos2pin) * (3.0 / 8.0 + 1.0 / 4.0 * cos2pin)) / valence
    alpha = 1.0 - beta * valence
    gamma = 1.0 - alpha - beta

    return alpha, beta, gamma

def get_meshtriangle_distances(point, vertex_array):

    temp_dist = distance.cdist(point, vertex_array)



    temp_dist = 1./temp_dist.transpose()

    weightv1 = temp_dist[0]
    weightv2 = temp_dist[1]
    weightv3 = temp_dist[2]

    return weightv1,weightv2,weightv3


def get_spectrum_projected(L,VA,mesh1,pt_cld ):

    pt_cld = o3d.io.read_point_cloud("data/1_sphere/sphere_poisson.ply")
    closest_points = trimesh.proximity.closest_point(mesh1, pt_cld.points)

    spect = np.zeros([L.shape[1], 3])

    for j in range(L.shape[1]):
        basis = L[:,j].reshape(1,-1).T
        print(basis.shape)
        print(mesh1.vertices.shape)
        k = np.multiply(basis,mesh1.vertices)


        spect = spect + np.multiply(basis,mesh1.vertices)

    #spect = np.linalg.norm(spect, axis=0)
    spect = spect * mesh1.area/mesh1.vertices.shape[0] #  #/ len(pt_cld.points)  # *mesh1.area #/ L.shape[0]#*np.sum(VA) #/mesh.faces.shape[0] # L.shape[0]#
    #spect = np.sum(L, axis=1)

    plt.plot(spect)
    plt.title('Spectrum plot')
    plt.show()


    return spect


def get_spectrum(L,VA,mesh1, pt_cld):

    #pt_cld = o3d.io.read_point_cloud("data/1_sphere/sphere_poisson.ply")
    closest_points = trimesh.proximity.closest_point(mesh1, pt_cld.points)

    spect = np.zeros(L.shape[1])

    for j in range(len(pt_cld.points)):

        i = closest_points[2][j]
        i0 = mesh1.faces[i, 0]
        i1 = mesh1.faces[i, 1]
        i2 = mesh1.faces[i, 2]

        weightv1,weightv2,weightv3 = get_weights_frompaper()#get_meshtriangle_distances(np.array([pt_cld.points[j]]).reshape(1,-1), np.array([mesh1.vertices[i0],mesh1.vertices[i1],mesh1.vertices[i2]]))

        spect = spect +( (weightv1 * L[i0,:] + weightv2 * L[i1,:] + weightv3 * L[i2,:] )/ (weightv1 + weightv2 + weightv3))

    spect = np.square(np.asarray(spect)) *mesh1.area /len(pt_cld.points)#/len(pt_cld.points)# *mesh1.area #/len(pt_cld.points) #*mesh1.area #/ L.shape[0]#*np.sum(VA) #/mesh.faces.shape[0] # L.shape[0]#

    plt.plot(spect)
    plt.title('Spectrum plot')
    plt.show()
    return spect

# split frequency in bands of 2l +1 (as mentioned in the paper)
def get_radial_means(spectrum):


    spectrum= np.asarray(spectrum)
    size2 = spectrum.shape
    print(size2)
    R = []
    A = []

    ns = 0
    i = 0
    l = 0.75
    ne = 2


    while(ne < len(spectrum)):

        ne = round(ns + (2 * l + 1))
        R.append(statistics.mean(spectrum[ns:ne]))
        A.append(statistics.variance(spectrum[ns:ne]) / R[i] ** 2)
        l += 1
        i += 1
        ns = ne + 1

    R[0] = 0.1
    plt.plot(R)
    plt.title('Radial means')
    plt.show()
    plt.plot(A)
    plt.title('Anisotropy')
    plt.show()

    return R,A


# Implementation as per the code of the paper
def get_radial_means_copy(S):

    R = np.zeros((math.ceil(math.sqrt(len(S)))))
    A = np.zeros((math.ceil(math.sqrt(len(S)))))


    ns = 1
    nl = 2
    i = 1


    while (ns < len(S)) and (i<len(R)):

        ne = min(ns + nl - 1, len(S))
        print(ns)
        print(ne)
        print(S[ns:ne])
        R[i] = statistics.mean(S[ns:ne+1])
        A[i] = statistics.variance(S[ns:ne+1])/ R[i] ** 2
        i = i + 1
        ns = ns + nl
        nl = nl + 2



    R[0]= 0.1

    plt.plot(R)
    plt.title('Radial means')
    plt.show()
    plt.plot(A)
    plt.title('Anisotropy')
    plt.show()

    return R,A


def RadialMeanAccurate(S, Freq):

    size2 = len(S)

    freq = np.sqrt(Freq)
    fmax = freq[size2-1]
    nbin = 100

    R = np.zeros(nbin)
    A = np.zeros(nbin)

    ns = 0
    ne = 1
    idx = 0

    while (1):
        while (ne < size2) and (freq[ne] <= (idx * fmax / nbin)):
            ne = ne + 1

        if ne == size2:
            break

        if idx==nbin:
            break

        print(idx)
        print(ns)
        print(ne)

        R[idx] = statistics.mean(S[ns:ne+1])
        A[idx] = statistics.variance(S[ns:ne+1]) / R[idx]**2
        idx = idx + 1
        ns = ne
        ne = ne + 1


    #R[0] = 0.1

    plt.plot(R)
    plt.title('Radial means')
    plt.show()
    plt.plot(A)
    plt.title('Anisotropy')
    plt.show()

    return R, A
