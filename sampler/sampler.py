import numpy as np
import random
import math


def random_sampler(shape = 'sphere', r = 1, num = None):
    samples_number = [64, 128, 256, 512]

    if num == None:
        num = random.choice(samples_number)

    if shape == 'sphere':
        phi = np.random.uniform(0, 2 * math.pi, num)
        costheta = np.random.uniform(-1, 1, num)
        u = np.random.uniform(0, 1, num)

        theta = np.arccos(costheta)
        r = 1

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        x = x.reshape(num, 1)
        y = y.reshape(num, 1)
        z = z.reshape(num, 1)

        points = np.append(x, y, axis=1)
        points = np.append(points, z, axis=1)
        np.savetxt('sphere_sam.xyz', points)

    if shape == 'plane':
        x = np.random.uniform(0, 1, num)
        y = np.random.uniform(0, 1, num)

        points = np.append(x.reshape(-1,1),y.reshape(-1,1), axis=1)
        temp = np.zeros(num)
        points3D = np.append(points, temp.reshape(-1,1) , axis=1)

    else :
        points = 0

    return points, points3D

#def poisson_sampler(shape = 'sphere', r = 1, num = None, filename= None):







