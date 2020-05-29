import numpy as np

def get_bunny_40k():
    ##               40k bunny.stl eigenmap #######################################################################################

    eigen1 = np.loadtxt('eigen1_0to28968.txt')
    basis1 = np.loadtxt('basis0to28968.txt')

    eigen2 = np.loadtxt('eigen2_18836to45312.txt')
    eigen2 = eigen2[919:]
    basis2 = np.loadtxt('basis2_18836to45312.txt')
    basis2 = basis2[:,919:]

    eigen3 = np.loadtxt('eigen3.txt')
    eigen3 = eigen3[692:]
    basis3 = np.loadtxt('basis3.txt')
    basis3 = basis3[:,692:]

    eigen4 = np.loadtxt('eigen4.txt')
    eigen4 = eigen4[213:]
    basis4 = np.loadtxt('basis4.txt')
    basis4 = basis4[:,213:]

    eigen4_5 = np.loadtxt('eigen4.5.txt')
    eigen4_5 = eigen4_5[365:]
    basis4_5 = np.loadtxt('basis4.5.txt')
    basis4_5 = basis4_5[:, 365:]

    eigen5 = np.loadtxt('eigen5.txt')
    eigen5 = eigen5[508:]
    basis5 = np.loadtxt('basis5.txt')
    basis5 = basis5[:, 508:]

    eigen6 = np.loadtxt('eigen6.txt')
    eigen6 = eigen6[1048:]
    basis6 = np.loadtxt('basis6.txt')
    basis6 = basis6[:, 1048:]

    freq = np.concatenate((eigen1, eigen2, eigen3, eigen4, eigen4_5, eigen5, eigen6), axis=0)
    basis = np.concatenate((basis1, basis2, basis3, basis4, basis4_5, basis5, basis6), axis=1)

    return freq, basis