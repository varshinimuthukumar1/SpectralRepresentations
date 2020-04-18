import numpy as np
import matplotlib.pyplot as plt
import math
import trimesh
import statistics

def get_spectrum(L,VA):

    mesh = trimesh.load_mesh('3d_model_of_Sphere.stl')
    valence = 6
    cos2pin = math.cos(2.0 * math.pi / 6)
    beta = (5.0 / 8.0 - (3.0 / 8.0 + 1.0 / 4.0 * cos2pin) * (3.0 / 8.0 + 1.0 / 4.0 * cos2pin)) / 6
    alpha = 1.0 - beta * valence
    gamma = 1.0 - alpha - beta

    spect = np.zeros(L.shape[1])
    for i in range(0, 3071):

        i0 = mesh.faces[i, 0]
        i1 = mesh.faces[i, 1]
        i2 = mesh.faces[i, 2]
        spect = spect + alpha*L[i0,:]+ beta*L[i1,:] + gamma*L[i2,:]

    #spect = np.square(np.asarray(spect))
    print(spect.shape)
    #spect=np.multiply(spect,VA )#/L.shape[0])
    #print(spect)
    plt.plot(spect)
    plt.show()
    return spect


def get_radialmeans(spectrum,freq):

    size2 = spectrum.shape
    R = np.zeros((1, math.ceil(math.sqrt(len(spectrum)))))
    A = np.zeros((1, math.ceil(math.sqrt(len(spectrum)))))

    ns = 1
    nl = 1
    i = 1
    while(ns < len(spectrum)):

        ne = min(ns+nl-1, spectrum.shape[0])
        R[i] = statistics.mean(spectrum[ns:ne])
        A[i] = statistics.variance(spectrum[ns:ne])/ R[i]^2
        i += 1
        ns = ns + nl
        nl = nl+2


    plt.plot(R)
    plt.show()
    plt.plot(A)
    plt.show()

    return R,A

