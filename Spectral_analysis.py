import numpy as np
import matplotlib.pyplot as plt
import math
import trimesh.proximity
import trimesh
import statistics
import open3d as o3d

def get_spectrum(L,VA,mesh1):

    pt_cld = o3d.io.read_point_cloud("output.ply")#trimesh.load_mesh('sphere_poisson.stl')
    closest_points = trimesh.proximity.closest_point(mesh1, pt_cld.points)
    valence = 6
    cos2pin = math.cos(2.0 * math.pi / valence)
    beta = (5.0 / 8.0 - (3.0 / 8.0 + 1.0 / 4.0 * cos2pin) * (3.0 / 8.0 + 1.0 / 4.0 * cos2pin)) / valence
    alpha = 1.0 - beta * valence
    gamma = 1.0 - alpha - beta

    spect = np.zeros(L.shape[1])

    for i in closest_points[2]:

        i0 = mesh1.faces[i, 0]
        i1 = mesh1.faces[i, 1]
        i2 = mesh1.faces[i, 2]
        spect = spect + alpha*L[i0,:]+ beta*L[i1,:] + gamma*L[i2,:]

    #spect =np.sum(L, axis=1)
    spect = np.square(np.asarray(spect)) *np.sum(VA) /len(closest_points[2])#/ L.shape[0]#*np.sum(VA) #/mesh.faces.shape[0] # L.shape[0]#
    print(spect.shape)
    #spect=np.multiply(spect,np.sum(VA) ) / mesh.faces.shape[0]
    #print(spect)
    plt.plot(spect)
    plt.title('Spectrum plot')
    plt.show()
    return spect


def get_radialmeans(spectrum):


    spectrum= np.asarray(spectrum)
    size2 = spectrum.shape
    print(size2)
    R = []
    A = []
    #R = np.zeros((1, math.ceil(math.sqrt(len(spectrum)))))
    #A = np.zeros((1, math.ceil(math.sqrt(len(spectrum)))))

    ns = 0
    i = 0
    l = 1
    ne = 2


    while(ne < len(spectrum)):

        ne = round(ns + (2 * l + 1))
        R.append(statistics.mean(spectrum[ns:ne]))
        A.append(statistics.variance(spectrum[ns:ne]) / R[i] ** 2)
        l += 1
        i += 1
        ns = ne + 1

    plt.plot(R)
    plt.title('Radial means')
    plt.show()
    plt.plot(A)
    plt.title('Anisotropy')
    plt.show()

    return R,A

def get_radial_means_copy(S):


    #S = np.asarray(S).T
    print(S.shape)
    #R = np.zeros((round(math.sqrt(size2[1]))))
    #A = np.zeros((round(math.sqrt(size2[1]))))

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

