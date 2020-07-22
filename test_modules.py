##
import scipy.misc
import torch
import tensorflow as tf
import numpy as np
import imageio
import matplotlib.pyplot as plt
import math
import cv2
import os

## my modules
<<<<<<< HEAD
import move_points.io_utils as io_utils
from move_points.Fourier import *
from move_points.Fourier import fourier_loss
=======
import io_utils
from Fourier import *
from Fourier import fourier_loss
>>>>>>> 853f09313fb4406c8395a06420856335baba00ff
import utils
from sampler import sampler

def test_fourier_computation_tf(opt):


    points_2d, _ = io_utils.read_2Dplane(opt.sample_patch)
    ptcloud_points_t = torch.from_numpy(points_2d)
    ptcloud_points_t = ptcloud_points_t.float()

    ptcloud_points_t = ptcloud_points_t.unsqueeze(0)


    power = fourier_loss.bat_compute_fourier_spectrum2D(ptcloud_points_t)

    power_n = power.numpy()
    imageio.imwrite(opt.outdir +'power.jpg',    power_n)
    cv2.imwrite('image_32.exr', power_n)

    radial_average = fourier_loss.radialSpectrumMC(ptcloud_points_t)

    radialmeans = radial_average.eval(session=tf.compat.v1.Session())
    np.savetxt('radialmeans.txt', radialmeans, delimiter=',')
    plt.plot(radialmeans[10:])
    plt.title("Radial means")
    plt.savefig('radialmeans.png')
    plt.show()


    return

<<<<<<< HEAD
def test_radialmeans_computation(opt):


    points_2d, _ = io_utils.read_2Dplane(opt.sample_patch)
    ptcloud_points_t = torch.from_numpy(points_2d)
    ptcloud_points_t = ptcloud_points_t.float()

    ptcloud_points_t = ptcloud_points_t.unsqueeze(0)


    radialmeans, power = fourier_loss.bat_compute_radialmeans(ptcloud_points_t)

    power_n = power.numpy()
    imageio.imwrite(opt.outdir +'power.jpg',    power_n)
    cv2.imwrite('image_32.exr', power_n)

    #radial_average = fourier_loss.radialSpectrumMC(ptcloud_points_t)

    #radialmeans = radial_average.eval(session=tf.compat.v1.Session())
    np.savetxt('radialmeans.txt', radialmeans, delimiter=',')
    plt.plot(radialmeans[1:])
    plt.title("Radial means")
    plt.savefig('radialmeans.png')
    plt.show()


    return

=======
>>>>>>> 853f09313fb4406c8395a06420856335baba00ff

def fouriercomputation_2Dpoints(opt):
    #ptcloud_points = io_utils.read_pointcloud(opt.sample_patch)
    #points_2d, points_3d =   utils.bat_rotate3Dplane2(ptcloud_points, [0, 0, 1])
     # ptcloud_points_t.unsqueeze(0)
    #points_2d = ptcloud_points[:,:-1]
    points_2d, _ = io_utils.read_2Dplane(opt.sample_patch)
    power = fourier_loss.compute_fourier_spectrum2D(points_2d)
    power = np.float32(power)
    imageio.imwrite(opt.outdir + 'power.jpg', power)
    cv2.imwrite(opt.outdir + 'spectrum.exr', power)

    radial_means = fourier_loss.compute_radial_means(spectrum=power)

    np.savetxt(opt.outdir + 'radialmeans.txt' , radial_means, fmt='%1.5f')


    return


def plot_frequencies_mesh_pc():

    an_freq = np.zeros((10000))
    i = 0
    m = 0
    while(True):

        l = m + 2*i +1
        an_freq[m:l] = i * (i+1)


        i +=1
        m = l

        if l == 10000:
            break

    freq = np.loadtxt('freq.txt')
    freq_mesh = np.loadtxt('freq_mesh.txt')
    freq_mesh /= (4 * math.pi)

    plt.plot(an_freq[:100], 'r')
    plt.plot(freq[:100], 'g')
    plt.plot(freq_mesh[:100], 'b')
    plt.show()

    plt.plot(an_freq, 'r')
    plt.plot(freq, 'g')
    plt.plot(freq_mesh, 'b')

    plt.show()
    return


def samples_generator(opt):
    if opt.sampling == 'random':
        for j in np.arange(opt.N):
            points2D, points3D = sampler.random_sampler(shape=opt.shape, num = opt.nsamples)

            path = os.path.join(opt.datadir,opt.shape)
            if not os.path.exists(path):
                os.makedirs(path)
            filename =  str(j) + '_plane_random_' + str(opt.nsamples) + '.xyz'
            filename = os.path.join(path,filename)
            np.savetxt(filename, points3D,  fmt='%1.5f')

            filename = str(j) + '_plane_random_' + str(opt.nsamples) + '.txt'
            filename = os.path.join(path, filename)
            np.savetxt(filename, points2D, fmt='%1.5f')


    return

<<<<<<< HEAD
def create_target(opt):

    for i in np.arange(15):

        if i ==0:
            filename = str(i)+'_'  + opt.sample_patch
            path = os.path.join(opt.datadir, filename)
            points_2d, _ = io_utils.read_2Dplane(path)
            ptcloud_points_t = torch.from_numpy(points_2d)
            ptcloud_points_t = ptcloud_points_t.float()

            ptcloud_points_t = ptcloud_points_t.unsqueeze(0)

            bat_points = ptcloud_points_t

        else:
            filename = str(i) + '_' + opt.sample_patch
            path = os.path.join(opt.datadir, filename)
            points_2d, _ = io_utils.read_2Dplane(path)
            ptcloud_points_t = torch.from_numpy(points_2d)
            ptcloud_points_t = ptcloud_points_t.float()

            ptcloud_points_t = ptcloud_points_t.unsqueeze(0)

            bat_points = torch.cat((bat_points, ptcloud_points_t), dim=0)



    radialmeans, power = fourier_loss.bat_compute_radialmeans(bat_points)

    power_n = power.numpy()
    imageio.imwrite(opt.outdir + 'power.jpg', power_n)
    cv2.imwrite('target_spectrum.exr', power_n)

    # radial_average = fourier_loss.radialSpectrumMC(ptcloud_points_t)

    # radialmeans = radial_average.eval(session=tf.compat.v1.Session())
    np.savetxt('target_radialmeans.txt', radialmeans, delimiter=',')
    plt.plot(radialmeans[1:])
    plt.title("Radial means")
    plt.savefig('radialmeans.png')
    plt.show()

    return




=======
>>>>>>> 853f09313fb4406c8395a06420856335baba00ff


