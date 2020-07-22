import os,sys
import io_utils

import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

# import my modules

from samplenet import samplenet
from Fourier import fourier_loss
import cv2



def load_data():

    train = None
    test = None
    for i in range(61):
        filename = 'data/plane/' + str(i) + '_plane_random_256.txt'
        points_2d, _ = io_utils.read_2Dplane(filename)
        points_2d = np.expand_dims(points_2d, axis=0)

        if train is None:
            train = points_2d
        else:
            train = np.concatenate((train, points_2d), axis=0)

    ptcloud_points_t = torch.tensor(train, requires_grad=True)
    points_train = ptcloud_points_t.float()

    for i in range(61, 81):

        filename = 'data/plane/' + str(i) + '_plane_random_256.txt'
        points_2d, _ = io_utils.read_2Dplane(filename)
        points_2d = np.expand_dims(points_2d, axis=0)

        if test is None:
            test = points_2d
        else:
            test = np.concatenate((test, points_2d), axis=0)

    print(train.shape)
    print(test.shape)
    ptcloud_points_t = torch.tensor(test, requires_grad=True)
    points_test = ptcloud_points_t.float()

    return points_train, points_test

def save_pred(pred):

    for i in range(pred.shape[0]):
        ptc = pred[i,:,:]
        temp = np.zeros(ptc.shape[0])
        points3D = np.append(ptc, temp.reshape(-1, 1), axis=1)
        filename = 'pred'+ str(i) + '.xyz'
        np.savetxt(filename, points3D)

def bat_wrap_toroidal(bat_patch):
    less_zero = torch.empty(bat_patch.shape)
    greater_one = torch.empty(bat_patch.shape)
    wrapped = torch.empty(bat_patch.shape)

    less_zero = -bat_patch - torch.floor(- bat_patch)
    ones = torch.ones(bat_patch.shape)
    less_zero = ones - less_zero

    wrapped= torch.where(bat_patch <0, less_zero, bat_patch)


    greater_one = greater_one - torch.floor(greater_one)
    wrapped = torch.where(bat_patch >1, greater_one, wrapped)

    return wrapped

def test_samplenet():

    points_train, points_test = load_data()
    n_patches = points_train.shape[0]
    n_points = points_train.shape[1]
    n_dim =  points_train.shape[2]
    # fix


    criterion = torch.nn.L1Loss()


    # fix target
    points_2d, target3d = io_utils.read_2Dplane('data/plane/dart_throwing_256.txt')
    ptcloud_points_t = torch.tensor(points_2d, requires_grad=True)
    ptcloud_points_t = ptcloud_points_t.float()
    target = ptcloud_points_t.unsqueeze(0)
    #target = target.to("cuda")
    target_fourier = fourier_loss.bat_compute_fourier_spectrum2D(target)
    #target_fourier = target_fourier.to("cuda")

    # train



    # eval
    model = samplenet(num_points=n_points, hidden_size=n_points)

    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    print(model.modules)
    #points_test = points_test.to("cuda")
    pred = model(points_test)
    pred_fourier = fourier_loss.bat_compute_fourier_spectrum2D(pred)
    pred = bat_wrap_toroidal(pred)
    testloss = criterion(pred_fourier, target_fourier)

    # detach variables
    # save target
    # np.savetxt('target', target3d)

    target_save = target_fourier.detach().numpy()
    cv2.imwrite('target_fourier.exr', target_save)

    #pred = bat_wrap_toroidal(pred)
    pred = pred.detach().numpy()
    points_test = points_test.detach().numpy()
    plt.scatter(pred[0,:,0],pred[0,:,1] , c = 'red')
    plt.scatter(points_test[0,:,0], points_test[0,:,1] , c = 'blue')
    plt.savefig('output.png')

    pred_fourier = pred_fourier.detach().numpy()

    cv2.imwrite('pred_fourier.exr', pred_fourier)

    save_pred(pred)

    print('test loss: {}'.format(testloss.item()))



if __name__ == '__main__':
    mydir = test_samplenet()