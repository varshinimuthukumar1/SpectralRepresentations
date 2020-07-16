import numpy as np
import matplotlib.pyplot as plt
import math
import trimesh.proximity
import trimesh
import statistics

from scipy.spatial import distance


def get_weights_frompaper():
    valence = 6
    cos2pin = math.cos(2.0 * math.pi / valence)
    beta = (5.0 / 8.0 - (3.0 / 8.0 + 1.0 / 4.0 * cos2pin) * (3.0 / 8.0 + 1.0 / 4.0 * cos2pin)) / valence
    alpha = 1.0 - beta * valence
    gamma = 1.0 - alpha - beta

    return alpha, beta, gamma

def get_meshtriangle_distances(point, vertex_array):

    temp_dist = distance.cdist(point, vertex_array, metric='euclidean')

    temp_dist = 1./temp_dist#.transpose()

    weightv1 = temp_dist[:,0]/np.sum(temp_dist)
    weightv2 = temp_dist[:,1]/np.sum(temp_dist)
    weightv3 = temp_dist[:,2]/np.sum(temp_dist)

    print(weightv3 + weightv2 + weightv1)

    return weightv1,weightv2,weightv3


def get_spectrum(L,VA,mesh1, pt_cld):

    #L = np.loadtxt('basis.txt')
    #pt_cld = o3d.io.read_point_cloud("data/1_sphere/sphere_poisson.ply")
    closest_points = trimesh.proximity.closest_point(mesh1, pt_cld.points)

    spect = np.zeros(L.shape[1])

    for j in range(len(pt_cld.points)):

        i = closest_points[2][j]
        i0 = mesh1.faces[i, 0]
        i1 = mesh1.faces[i, 1]
        i2 = mesh1.faces[i, 2]

        a = mesh1.vertices[i0]
        b = mesh1.vertices[i1]
        arr = np.array([mesh1.vertices[i0], mesh1.vertices[i1], mesh1.vertices[i2]])
        weightv1,weightv2,weightv3 = get_meshtriangle_distances(np.array([pt_cld.points[j]]).reshape(1,-1), np.array([mesh1.vertices[i0],mesh1.vertices[i1],mesh1.vertices[i2]]))

        spect = spect +( (weightv1 * L[i0,:] + weightv2 * L[i1,:] + weightv3 * L[i2,:] ))#/ (weightv1 + weightv2 + weightv3))




    #spect = np.square(np.asarray(spect)) *mesh1.area #/len(pt_cld.points)#/len(pt_cld.points)# *mesh1.area #/len(pt_cld.points) #*mesh1.area #/ L.shape[0]#*np.sum(VA) #/mesh.faces.shape[0] # L.shape[0]#


    spect *= ((mesh1.area) / len(pt_cld.points))

    power = (len(pt_cld.points) / (mesh1.area)) * np.abs(spect)** 2
    power /= (mesh1.area)


    plt.plot(power[1:])
    plt.title('Spectrum plot')
    plt.show()
    return power



def RadialMeanAccurate(Freq):
    S = np.loadtxt('spectrum.txt')
    #Freq = np.loadtxt('eigen.txt')
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




        while (float(freq[ne]) < float((idx * fmax /nbin))):
            if (ne == size2-1):
                ne = ne-1
                break
            ne = ne+1

        print(ne - ns)
        print(S[ns], S[ne])




        R[idx] = np.sum(S[ns:ne])/len(S[ns:ne+1]) #np.sum(S[ns:ne])/len(S[ns:ne])#statistics.mean(S[ns:ne])
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

def get_radial_means_spherical():

    spect = np.loadtxt('spectrum.txt')


    l = 0
    m = 0
    i = 0

    ne =0
    ns = 0

    R =[]


    while(ne < len(spect)):

        while(m< 2* l +1):
            ne = ne+1
            m = m+1
        l+=1
        m=0

        print(spect[ns:ne])
        print(sum(spect[ns:ne]))
        print(len(spect[ns:ne]))
        print(np.sum(spect[ns:ne])/len(spect[ns:ne]))


        R.append(np.sum(spect[ns:ne])/len(spect[ns:ne]))

        ns = ne
        if ne == len(spect):
            break


    #R[0] = 0.1
    plt.plot(R)
    plt.title('Radial means')
    plt.show()

    return R


