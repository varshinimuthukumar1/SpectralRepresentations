import os,sys




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
    cancel_DC = True

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

    #torch.cuda.empty_cache()
    angle = torch.mul(fp, 2 * math.pi)
    angle = torch.mean(angle, dim=2)
    realCoeff = torch.cos(angle)
    imagCoeff = torch.sin(angle)

    realCoeff = torch.sum(realCoeff, 1)
    imagCoeff = torch.sum(imagCoeff, 1)

    power = (realCoeff ** 2 + imagCoeff ** 2) / n_points
    power = torch.mean(power, dim=0)

    return power

def bat_compute_radialmeans(bat_points):



    def get_spectrum(grid, bat_points):

        grid = grid.expand(n_patches, dim_points, spectrum_resolution, spectrum_resolution)
        # grid = grid.to("cuda")
        fp = torch.tensordot(bat_points, grid, dims=([2], [1]))

        # torch.cuda.empty_cache()
        angle = torch.mul(fp, 2 * math.pi)
        angle = torch.mean(angle, dim=2)
        realCoeff = torch.cos(angle)
        imagCoeff = torch.sin(angle)

        realCoeff = torch.sum(realCoeff, 1)
        imagCoeff = torch.sum(imagCoeff, 1)

        power = (realCoeff ** 2 + imagCoeff ** 2) / n_points
        power = torch.mean(power, dim=0)

        return power

    n_patches = bat_points.shape[0]
    n_points = bat_points.shape[1]
    dim_points = bat_points.shape[2]

    spectrum_resolution = 64
    freqstep = 1
    cancel_DC = True

    xlow = - spectrum_resolution / 2
    xhigh = spectrum_resolution / 2
    u = torch.arange(xlow, xhigh, freqstep)
    v = torch.arange(xlow, xhigh, freqstep)

    uu, vv = torch.meshgrid(u, v)
    grid = torch.stack((uu, vv))
    grid = grid.float()
    grid2 = grid.clone().detach().numpy()
    plt.scatter(grid2[0, :, :].flatten(), grid2[1, :, :].flatten())
    plt.savefig('grid.png')
    plt.clf()

    power = get_spectrum(grid, bat_points)

    # compute radial means
    r = torch.sqrt(grid[0,:,:] **2 + grid[1,:,:] **2)
    max_r = np.sqrt(2 * (spectrum_resolution/2 )**2)
    step_r = max_r/100
    radialmeans = torch.zeros(power.shape)
    radialmeans = radialmeans.unsqueeze(0)
    print(power[32,32])

    for i in np.arange(0, max_r, step_r ):

        if i ==0:

            masked = torch.where(((r<(i+ step_r)) ), power, torch.zeros(power.shape) )
            masked = torch.where(r>i, masked, torch.zeros(power.shape))
            masked /= torch.nonzero(masked).size(0)
            masked = masked.unsqueeze(0)

            radialmeans = masked

        else:

            masked = torch.where(((r < (i + step_r))), power, torch.zeros(power.shape))
            masked = torch.where(r > i, masked, torch.zeros(power.shape))
            masked /= torch.nonzero(masked).size(0)
            masked = masked.unsqueeze(0)

            radialmeans = torch.cat((radialmeans, masked), dim=0)


    radialmeans =torch.sum(radialmeans, dim=(1,2))
    radialmeans = torch.where(radialmeans== radialmeans, radialmeans, torch.zeros(radialmeans.shape))
    return radialmeans, power

def bat_compute_radialmeanscirc(bat_points):

    assert bat_points is not None, "No input to the function bat_compute_radialmeans"

    n_patches = bat_points.shape[0]
    n_points = bat_points.shape[1]
    dim_points = bat_points.shape[2]

    spectrum_resolution = 64
    freqstep = 1
    cancel_DC = True

    xlow = - spectrum_resolution / 2
    xhigh = spectrum_resolution / 2
    u = torch.arange(xlow, xhigh, freqstep)
    v = torch.arange(xlow, xhigh, freqstep)

    r = torch.arange(xlow, xhigh, freqstep)
    theta = torch.arange(0, 2 * math.pi, math.pi/32)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    r = r.unsqueeze(1)
    cos_theta = cos_theta.unsqueeze(0)
    sin_theta = sin_theta.unsqueeze(0)

    uu = torch.matmul(r, cos_theta)
    print(uu.shape)
    vv = torch.matmul(r, sin_theta)

    #uu, vv = torch.meshgrid(u, v)
    grid = torch.stack((uu, vv))
    grid = grid.float()
    grid2 = grid.clone().detach().numpy()
    plt.scatter(grid2[0, :, :].flatten(), grid2[1, :, :].flatten())
    plt.savefig('grid.png')
    plt.clf()

    grid = grid.expand(n_patches, dim_points, spectrum_resolution, spectrum_resolution)
    # grid = grid.to("cuda")
    fp = torch.tensordot(bat_points, grid, dims=([2], [1]))

    # torch.cuda.empty_cache()
    angle = torch.mul(fp, 2 * math.pi)
    angle = torch.mean(angle, dim=2)
    realCoeff = torch.cos(angle)
    imagCoeff = torch.sin(angle)

    realCoeff = torch.sum(realCoeff, 1)
    imagCoeff = torch.sum(imagCoeff, 1)

    power = (realCoeff ** 2 + imagCoeff ** 2) / n_points
    power = torch.mean(power, dim=0)
    radial_means = 0
    return radial_means,power