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

    weightv1 = temp_dist[0]/np.sum(temp_dist)
    weightv2 = temp_dist[1]/np.sum(temp_dist)
    weightv3 = temp_dist[2]/np.sum(temp_dist)

    return weightv1,weightv2,weightv3


def get_spectrum(L,VA,mesh1, pt_cld):

    #pt_cld = o3d.io.read_point_cloud("data/1_sphere/sphere_poisson.ply")
    closest_points = trimesh.proximity.closest_point(mesh1, pt_cld.points)

    spect = np.zeros(L.shape[1])

    for j in range(len(pt_cld.points)):

        i = closest_points[2][j]
        i0 = mesh1.faces[i, 0]
        i1 = mesh1.faces[i, 1]
        i2 = mesh1.faces[i, 2]

        weightv1,weightv2,weightv3 = get_meshtriangle_distances(np.array([pt_cld.points[j]]).reshape(1,-1), np.array([mesh1.vertices[i0],mesh1.vertices[i1],mesh1.vertices[i2]]))

        spect = spect +( (weightv1 * L[i0,:] + weightv2 * L[i1,:] + weightv3 * L[i2,:] ))#/ (weightv1 + weightv2 + weightv3))


    spect *= ((4 * math.pi)/len(pt_cld.points) )
    power = (len(pt_cld.points)/(4 * math.pi) ) * np.abs(spect)**2
    #spect = np.square(np.asarray(spect)) *mesh1.area #/len(pt_cld.points)#/len(pt_cld.points)# *mesh1.area #/len(pt_cld.points) #*mesh1.area #/ L.shape[0]#*np.sum(VA) #/mesh.faces.shape[0] # L.shape[0]#

    plt.plot(power)
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
    plt.plot(A)
    plt.title('Anisotropy')
    plt.legend()
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

    freq = Freq #np.sqrt(Freq)
    fmax = np.amax(freq)
    nbin = 100

    R = np.zeros(nbin)
    A = np.zeros(nbin)

    ns = 0
    ne = 2
    idx = 0

    for idx in range(nbin):

        if ((ns == size2-2) or (ne == size2-2)):
            break

        print(ns)
        print(ne)

        while (float(freq[ne]) < float((idx * fmax /nbin))):
            if (ne == size2-1):
                ne = ne-1
                break
            ne = ne+1




        R[idx] = statistics.mean(S[ns:ne+1]) #np.sum(S[ns:ne])/len(S[ns:ne])#statistics.mean(S[ns:ne])
        A[idx] = statistics.variance(S[ns:ne+1]) / R[idx] ** 2

        ns = ne
        ne = ne + 1


    R[0] = 0.1

    plt.plot(R)
    plt.title('Radial means')
    plt.show()
    plt.plot(A)
    plt.title('Anisotropy')
    plt.show()

    return R, A
