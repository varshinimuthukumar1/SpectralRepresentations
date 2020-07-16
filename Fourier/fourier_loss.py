import torch
import tensorflow as tf
import math

def compute_spectrum(points):

    batchCount = points.shape[0]
    pointCount = points.shape[1]
    dimCount = points.shape[2]

    assert dimCount == 2, " points not 2d for fourier spectrum"

    # compute power spectrum

    spectrumRes = 64
    freqStep = 1

    xlow = - spectrumRes * freqStep * 0.5
    xhigh = spectrumRes * freqStep * 0.5
    halfres = int(spectrumRes * 0.5)
    ylow = xlow
    yhigh = xhigh

    u = torch.range(xlow, xhigh, freqStep)
    v = torch.range(xlow, xhigh, freqStep)

    uu, vv = torch.meshgrid(u, v)
    grid = torch.stack([uu, vv], dim=0)
    #grid = torch.meshgrid([torch.arange(0, 5), torch.arange(0, 10)])

    ## redo this line #########################################
    batchGrid = grid.expand(grid.shape[0], grid.shape[1], points.shape[0]).transpose(0, 2)
    print('batchGrid size')
    print(batchGrid.shape)

    dotXU = torch.dot(points, batchGrid, [2,1])
    angle = dotXU * 2 * (22/7)

    angleout =torch.mean(angle, 2)
    realCoeff = torch.sum(torch.cos(angleout), 1)
    imagCoeff = torch.sum(torch.sin(angleout), 1)



    return realCoeff, imagCoeff



def compute_spectrum_tf(points):

    with tf.name_scope('spectrum2D'):
        batchCount = points.shape[0]
        pointCount = points.shape[1]
        dimCount = points.shape[2]

        assert dimCount == 2, " points not 2d for fourier spectrum"

        # compute power spectrum

        spectrumRes = 64
        freqStep = 1

        xlow = - spectrumRes * freqStep * 0.5
        xhigh = spectrumRes * freqStep * 0.5
        halfres = int(spectrumRes * 0.5)
        ylow = xlow
        yhigh = xhigh

        u = tf.range(xlow, xhigh, freqStep)
        v = tf.range(xlow, xhigh, freqStep)
        uu, vv = tf.meshgrid(u,v)
        grid = tf.to_float([uu,vv])
        batchGrid = tf.tile(tf.expand_dims(grid,0),[batchCount,1,1,1])

        dotXU = tf.tensordot(points, batchGrid, [[2],[1]])
        angle = tf.scalar_mul(2.0*tf.constant(math.pi), dotXU)

        angleout = tf.reduce_mean(angle, 2)
        realCoeff = tf.reduce_sum(tf.cos(angleout), 1)
        imagCoeff = tf.reduce_sum(tf.sin(angleout), 1)
        power = (realCoeff**2 + imagCoeff**2) / 100

        # Average across all mini batches
        power = tf.reduce_mean(power, 0)

        dcPos = int(spectrumRes / 2.)
        dcComp = power[dcPos, dcPos]
        power -= tf.scatter_nd([[dcPos, dcPos]], [dcComp], power.shape)

        return power
