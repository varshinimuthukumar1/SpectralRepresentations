<<<<<<< HEAD
import os,sys
=======
>>>>>>> 853f09313fb4406c8395a06420856335baba00ff
import io_utils

import numpy as np
import torch
<<<<<<< HEAD
import argparse
import matplotlib.pyplot as plt
=======

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

>>>>>>> 853f09313fb4406c8395a06420856335baba00ff

# import my modules

from samplenet import samplenet
from Fourier import fourier_loss
import cv2

<<<<<<< HEAD
def parse_arguments():
    parser = argparse.ArgumentParser()


    # training parameters
    parser.add_argument('--ground_truth', type=str, default='', help='folder with target spectrums')
    parser.add_argument('-- target_spectrum' , type=str, default='target_spectrum.exr', help='arget spectrum')
    parser.add_argument('-- target_radialmeans', type=str, default='target_radialmeans.txt', help='arget spectrum')
    parser.add_argument('--lr', type=float, default=0.000001, help='learning rate')

    # model hyperparameters

    parser.add_argument('--points_per_patch', type=int,
                        default=256, help='max. number of points per patch')# 50

    return parser.parse_args()

=======
>>>>>>> 853f09313fb4406c8395a06420856335baba00ff
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

<<<<<<< HEAD
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

def train_samplenet(opt):
=======


def train_samplenet():
>>>>>>> 853f09313fb4406c8395a06420856335baba00ff

    points_train, points_test = load_data()
    n_patches = points_train.shape[0]
    n_points = points_train.shape[1]
    n_dim =  points_train.shape[2]
    # fix
    model = samplenet(num_points= n_points, hidden_size= n_points)
<<<<<<< HEAD

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr= opt.lr)
=======
    #model.to("cuda")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.00001)
>>>>>>> 853f09313fb4406c8395a06420856335baba00ff

    # fix target
    points_2d, target3d = io_utils.read_2Dplane('data/plane/dart_throwing_256.txt')
    ptcloud_points_t = torch.tensor(points_2d, requires_grad=True)
    ptcloud_points_t = ptcloud_points_t.float()
<<<<<<< HEAD
    target = ptcloud_points_t.unsqueeze(0)
    #target = target.to("cuda")
    target_fourier = fourier_loss.bat_compute_fourier_spectrum2D(target)
    #target_fourier = target_fourier.to("cuda")
    #target_fourier = target_fourier.to("cuda")

    target_radmeans_np = np.loadtxt('target_radialmeans.txt')
    target_radmeans = torch.tensor(target_radmeans_np, requires_grad=True)
    target_radmeans  = target_radmeans.float()
=======

    target = ptcloud_points_t.unsqueeze(0)
    target_fourier = fourier_loss.bat_compute_fourier_spectrum2D(target)

    # save target
    np.savetxt('target', target3d)
    target_save = target_fourier.detach().numpy()
    cv2.imwrite('target_fourier.exr', target_save)

>>>>>>> 853f09313fb4406c8395a06420856335baba00ff

    # train
    print(model.modules)
    model.train()
<<<<<<< HEAD
    epoch = 20
=======
    epoch = 100
>>>>>>> 853f09313fb4406c8395a06420856335baba00ff

    for epoch in range(epoch):

        optimizer.zero_grad()

        #Forward pass
<<<<<<< HEAD
        #model.to("cuda")
        #points_train= points_train.to("cuda")
        pred = model(points_train)
        print(pred.device)
        pred = bat_wrap_toroidal(pred)
        #Compute loss
        radialmeans ,_ = fourier_loss.bat_compute_radialmeans(pred)

        print(radialmeans.type())
        print(target_radmeans.type())
        loss = criterion(radialmeans, target_radmeans)
=======

        #points_train= points_train.to("cuda")
        pred = model(points_train)

        #Compute loss

        loss = criterion(fourier_loss.bat_compute_fourier_spectrum2D(pred), target_fourier)
>>>>>>> 853f09313fb4406c8395a06420856335baba00ff

        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))

        # Backward pass
        loss.backward(retain_graph=True)
        optimizer.step()

<<<<<<< HEAD
    # save model
    torch.save(model.state_dict(), 'model.pth')

=======
>>>>>>> 853f09313fb4406c8395a06420856335baba00ff
    # eval

    model.eval()
    #points_test = points_test.to("cuda")
    pred = model(points_test)
<<<<<<< HEAD
    pred = bat_wrap_toroidal(pred)
    predradialmeans ,_ = fourier_loss.bat_compute_radialmeans(pred)
    testloss = criterion(predradialmeans, target_radmeans)

    # detach variables
    # save target
    # np.savetxt('target', target3d)

    target_save = target_fourier.detach().numpy()
    cv2.imwrite('target_fourier.exr', target_save)


    pred = pred.detach().numpy()
    points_test = points_test.detach().numpy()
    plt.scatter(pred[0,:,0],pred[0,:,1] , c = 'red')
    plt.scatter(points_test[0,:,0], points_test[0,:,1] , c = 'blue')
    plt.savefig('output1.png')
    plt.clf()

    plt.scatter(pred[0, :, 0], pred[0, :, 1], c='red')
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c='green')
    plt.savefig('output2.png')
    plt.clf()

    predradialmeans = predradialmeans.detach().numpy()

    np.savetxt('pred_radialmeans.txt', predradialmeans, delimiter=',')
    plt.plot(predradialmeans[1:])
    plt.title("Radial means")
    plt.savefig('predradialmeans.png')
    plt.show()

    save_pred(pred)

    print('test loss: {}'.format(testloss.item()))
    # save model
    torch.save(model.state_dict(), 'model.pth')

    return


if __name__ == '__main__':
    opt = parse_arguments()
    train_samplenet(opt)
=======
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
>>>>>>> 853f09313fb4406c8395a06420856335baba00ff
