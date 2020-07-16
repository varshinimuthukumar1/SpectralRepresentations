import torch
import math
import numpy as np
import matplotlib.pyplot as plt

## import my modules




def compute_fourier_spectrum2D(points):

    n_points = points.shape[0]
    dim_points = points.shape[1]

    spectrum_resolution = 64
    freqstep = 1

    xlow =  - spectrum_resolution /2
    xhigh =  spectrum_resolution/ 2
    u = np.arange(xlow, xhigh, freqstep)
    v = np.arange(xlow, xhigh, freqstep)

    uu,vv = np.meshgrid(u,v)
    grid = np.asarray([uu, vv])

    fp = np.tensordot(points, grid, axes=((1),(0)))


    angle = 2* math.pi * fp
    realCoeff = np.cos(angle)
    imagCoeff = np.sin(angle)

    realCoeff = np.sum(realCoeff, axis=0)
    imagCoeff = np.sum(imagCoeff, axis=0)

    power = (realCoeff **2 + imagCoeff**2)/points.shape[0]

    return power

def compute_radial_means(points=None, spectrum=None):
    if spectrum is not None:
        y, x = np.indices((spectrum.shape))  # first determine radii of all pixels
        center = int(spectrum.shape[0]/2), int(spectrum.shape[1]/2)
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        ind = np.argsort(r.flat)  # get sorted indices
        sr = r.flat[ind]  # sorted radii
        sim = spectrum.flat[ind]  # image values sorted by radii
        ri = sr.astype(np.int32)  # integer part of radii (bin size = 1)
        # determining distance between changes
        deltar = ri[1:] - ri[:-1]  # assume all radii represented
        rind = np.where(deltar)[0]  # location of changed radius
        nr = rind[1:] - rind[:-1]  # number in radius bin
        csim = np.cumsum(sim, dtype=np.float64)  # cumulative sum to figure out sums for each radii bin
        tbin = csim[rind[1:]] - csim[rind[:-1]]  # sum for image values in radius bins
        radial_means = tbin / nr  # the answer
        plt.plot(radial_means)
        plt.show()


    return radial_means

def bat_compute_fourier_spectrum2D(bat_points):

    n_patches = bat_points.shape[0]
    n_points = bat_points.shape[1]
    dim_points = bat_points.shape[2]

    spectrum_resolution = 64
    freqstep = 1

    xlow = - spectrum_resolution / 2
    xhigh = spectrum_resolution / 2
    u = torch.arange(xlow, xhigh, freqstep)
    v = torch.arange(xlow, xhigh, freqstep)

    uu, vv = torch.meshgrid(u, v)
    grid =torch.stack((uu, vv))
    grid = grid.float()

    grid = grid.expand(n_patches, dim_points, spectrum_resolution,spectrum_resolution)
    #grid = grid.to("cuda")
    fp = torch.tensordot(bat_points, grid, dims=([2], [1]))

    angle = torch.mul(fp, 2 * math.pi)
    angle = torch.mean(angle, dim=2)
    realCoeff = torch.cos(angle)
    imagCoeff = torch.sin(angle)

    realCoeff = torch.sum(realCoeff, 1)
    imagCoeff = torch.sum(imagCoeff, 1)

    power = (realCoeff ** 2 + imagCoeff ** 2) / n_points
    power = torch.mean(power, dim=0)

    return power

