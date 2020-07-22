import io_utils
import random
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn

# import my modules
from channelnet import channelnet
from Fourier import fourier_loss
import cv2

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        #nn.init.xavier_uniform(m.weight.data)
        nn.init.ones_(m.weight.data)

    return


def parse_arguments():
    parser = argparse.ArgumentParser()


    # training parameters
    parser.add_argument('--ground_truth', type=str, default='', help='folder with target spectrums')
    parser.add_argument('-- target_spectrum' , type=str, default='target_spectrum.exr', help='arget spectrum')
    parser.add_argument('-- target_radialmeans', type=str, default='target_radialmeans.txt', help='arget spectrum')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')

    # model hyperparameters

    parser.add_argument('--points_per_patch', type=int,
                        default=256, help='max. number of points per patch')# 50

    return parser.parse_args()

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

def train_samplenet(opt):

    ############## load data #############################
    points_train, points_test = load_data()

    n_patches = points_train.shape[0]
    n_points = points_train.shape[1]
    n_dim =  points_train.shape[2]

    points_train = points_train.unsqueeze(1)
    points_test = points_test.unsqueeze(1)

    points_train = points_train.permute(0,3,2,1)
    points_test = points_test.permute(0, 3, 2, 1)



    ################## fix samplenet parameters ######################
    model = channelnet(n_points, n_points, 2)
    random.seed(random.randint(1, 10000))
    torch.manual_seed(random.randint(1, 10000))
    model.apply(weights_init)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr= opt.lr)

    print(model.modules)

    ################ fix target #########################################

    target_radmeans_np = np.loadtxt('target_radialmeans.txt')
    plt.plot(target_radmeans_np)
    plt.title("Radial means")
    plt.savefig('radialmeans.png')
    target_radmeans = torch.tensor(target_radmeans_np, requires_grad=True)

    target_radmeans  = target_radmeans.float()


    ############## train #################################################

    model.train()
    epoch = 20
    for param in model.parameters():
        print(param.data)

    for epoch in range(epoch):

        optimizer.zero_grad()

        ###### Forward pass

        # to cuda
        #model.to("cuda")
        #points_train= points_train.to("cuda")

        print(points_train.shape)
        pred = model(points_train)
        print(pred.shape)
        pred = pred.squeeze(2)
        pred = pred.permute(0, 2, 1)

        pred = bat_wrap_toroidal(pred)

        ###### Compute loss

        radialmeans ,_ = fourier_loss.bat_compute_radialmeans(pred)

        loss = criterion(radialmeans, target_radmeans)

        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))

        # Backward pass
        loss.backward(retain_graph=True)
        optimizer.step()

    ############################## save model ###########################################
    torch.save(model.state_dict(), 'model.pth')

    ########################### evaluate with test set ##################################

    model.eval()
    ######### to cuda
    #points_test = points_test.to("cuda")

    pred = model(points_test)
    pred = pred.squeeze(2)
    pred = pred.permute(0, 2, 1)

    pred = bat_wrap_toroidal(pred)
    predradialmeans ,pred_fourier = fourier_loss.bat_compute_radialmeans(pred)
    testloss = criterion(predradialmeans, target_radmeans)

    # detach variables

    pred_fourier_save = pred_fourier.detach().numpy()
    cv2.imwrite('pred_fourier.exr', pred_fourier_save)


    pred = pred.detach().numpy()
    points_test = points_test.detach().numpy()
    plt.scatter(pred[0,:,0],pred[0,:,1] , c = 'red')
    plt.scatter(points_test[0,:,0], points_test[0,:,1] , c = 'blue')
    plt.savefig('output1.png')
    plt.clf()

    predradialmeans = predradialmeans.detach().numpy()

    np.savetxt('pred_radialmeans.txt', predradialmeans, delimiter=',')
    plt.plot(predradialmeans[1:])
    plt.title("Radial means")
    plt.savefig('predradialmeans.png')
    save_pred(pred)

    print('test loss: {}'.format(testloss.item()))
    # save model
    torch.save(model.state_dict(), 'model2.pth')

    return


if __name__ == '__main__':
    opt = parse_arguments()
    train_samplenet(opt)