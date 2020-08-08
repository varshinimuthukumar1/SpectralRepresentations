import os,sys
import io_utils
import torch.nn as nn
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# import my modules

from samplenet import samplenet
from Fourier import fourier_loss
import cv2

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        #nn.init.xavier_uniform_(m.weight.data)
        #nn.init.uniform_(m.weight.data)
        nn.init.ones_(m.weight.data)

    return

def load_data():

    train = None
    test = None
    for i in range(2):
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
    greater_one = torch.empty(bat_patch.shape).to('cuda')
    wrapped = torch.empty(bat_patch.shape)

    less_zero = -bat_patch - torch.floor(- bat_patch)
    ones = torch.ones(bat_patch.shape).to('cuda')
    less_zero = ones - less_zero


    wrapped= torch.where(bat_patch <0, less_zero, bat_patch)


    greater_one = greater_one - torch.floor(greater_one)
    wrapped = torch.where(bat_patch >1, greater_one, wrapped)

    return wrapped

def compute_bat_loss(pred, target_radmeans):

    loss = 0
    for points in pred:


        radialmeans, _ = fourier_loss.bat_compute_radialmeans(points)

        loss += (torch.nn.L1Loss(radialmeans, target_radmeans) / pred.shape[0])

        # predradialmeanssave = radialmeans.detach().numpy()

        # np.savetxt(str(epoch) + 'pred_radialmeans.txt', predradialmeanssave, delimiter=',')
        # plt.plot(predradialmeanssave[5:])
        # plt.title("Radial means")
        # plt.savefig(str(epoch) + 'predradialmeans.png')
        # plt.clf()

    return loss


def train_samplenet(opt):
    writer = SummaryWriter('runs/experiment_1')


    ############## load data #############################
    points_train, points_test = load_data()
    n_patches = points_train.shape[0]
    n_points = points_train.shape[1]
    n_dim =  points_train.shape[2]


    ################## fix samplenet parameters ######################
    model = samplenet(num_points= n_points, hidden_size= n_points*8)
    model.apply(weights_init)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr= opt.lr)

    ################ fix target #########################################

    target_radmeans_np = np.loadtxt('target_radialmeans.txt')
    target_radmeans = torch.tensor(target_radmeans_np, requires_grad=True)
    target_radmeans  = target_radmeans.float().to('cuda')

    ############## train #################################################

    model.train()
    epoch = 100

    for epoch in range(epoch):

        optimizer.zero_grad()

        #Forward pass
        model.to("cuda")
        points_train= points_train.to("cuda")
        pred = model(points_train)

        pred = bat_wrap_toroidal(pred)
        #Compute lossx

        loss = 0
        for points in pred:

            points = points.unsqueeze(0)
            radialmeans, _ = fourier_loss.bat_compute_radialmeans(points)

            loss += (criterion(radialmeans, target_radmeans) / pred.shape[0])

            writer.add_scalar('training loss',
                              loss / 1000,
                              epoch )

            # predradialmeanssave = radialmeans.detach().numpy()

            # np.savetxt(str(epoch) + 'pred_radialmeans.txt', predradialmeanssave, delimiter=',')
            # plt.plot(predradialmeanssave[5:])
            # plt.title("Radial means")
            # plt.savefig(str(epoch) + 'predradialmeans.png')
            # plt.clf()

        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))

        # Backward pass
        loss.backward(retain_graph=True)
        optimizer.step()

    ############################## save model ###########################################
    torch.save(model.state_dict(), 'model.pth')
    #for param in model.parameters():
        #print(param.data)
    ########################### evaluate with test set ##################################

    model.eval()
    points_test = points_test.to("cuda")
    pred = model(points_test)
    pred = bat_wrap_toroidal(pred)

    testloss = torch.zeros(1).to('cuda')
    for points in pred:

        points = points.unsqueeze(0)
        radialmeans, _ = fourier_loss.bat_compute_radialmeans(points)
        predradialmeans = radialmeans.clone().detach().cpu().numpy()

        testloss += (criterion(radialmeans, target_radmeans) / pred.shape[0])

        # predradialmeanssave = radialmeans.detach().numpy()

        # np.savetxt(str(epoch) + 'pred_radialmeans.txt', predradialmeanssave, delimiter=',')
        # plt.plot(predradialmeanssave[5:])
        # plt.title("Radial means")
        # plt.savefig(str(epoch) + 'predradialmeans.png')
        # plt.clf()




    ##tensorboard visualization
    #writer.add_graph(model)
    writer.close()




    # detach variables
    # save target
    # np.savetxt('target', target3d)

    #target_save = target_fourier.detach().numpy()
    #cv2.imwrite('target_fourier.exr', target_save)


    pred = pred.detach().cpu().numpy()
    points_test = points_test.detach().cpu().numpy()
    plt.scatter(pred[0,:,0],pred[0,:,1] , c = 'red')
    plt.scatter(points_test[0,:,0], points_test[0,:,1] , c = 'blue')
    plt.savefig('output1.png')
    plt.clf()

    # plt.scatter(pred[0, :, 0], pred[0, :, 1], c='red')
    # plt.scatter(points_2d[:, 0], points_2d[:, 1], c='green')
    # plt.savefig('output2.png')
    # plt.clf()

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
