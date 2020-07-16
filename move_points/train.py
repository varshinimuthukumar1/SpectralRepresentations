import io_utils

import numpy as np
import torch

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


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



def train_samplenet():

    points_train, points_test = load_data()
    n_patches = points_train.shape[0]
    n_points = points_train.shape[1]
    n_dim =  points_train.shape[2]
    # fix
    model = samplenet(num_points= n_points, hidden_size= n_points)
    #model.to("cuda")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.00001)

    # fix target
    points_2d, target3d = io_utils.read_2Dplane('data/plane/dart_throwing_256.txt')
    ptcloud_points_t = torch.tensor(points_2d, requires_grad=True)
    ptcloud_points_t = ptcloud_points_t.float()

    target = ptcloud_points_t.unsqueeze(0)
    target_fourier = fourier_loss.bat_compute_fourier_spectrum2D(target)

    # save target
    np.savetxt('target', target3d)
    target_save = target_fourier.detach().numpy()
    cv2.imwrite('target_fourier.exr', target_save)


    # train
    print(model.modules)
    model.train()
    epoch = 100

    for epoch in range(epoch):

        optimizer.zero_grad()

        #Forward pass

        #points_train= points_train.to("cuda")
        pred = model(points_train)

        #Compute loss

        loss = criterion(fourier_loss.bat_compute_fourier_spectrum2D(pred), target_fourier)

        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))

        # Backward pass
        loss.backward(retain_graph=True)
        optimizer.step()

    # eval

    model.eval()
    #points_test = points_test.to("cuda")
    pred = model(points_test)
    pred_fourier = fourier_loss.bat_compute_fourier_spectrum2D(pred)
    testloss = criterion(pred_fourier, target_fourier)

    # detach variables
    pred = pred.detach().numpy()
    pred_fourier = pred_fourier.detach().numpy()

    cv2.imwrite('pred_fourier.exr', pred_fourier)
    save_pred(pred)

    print('test loss: {}'.format(testloss.item()))


if __name__ == '__main__':
    mydir = train_samplenet()