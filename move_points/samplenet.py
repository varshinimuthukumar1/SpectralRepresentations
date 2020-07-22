import os,sys



import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable



class samplenet(nn.Module):
    
    def __init__(self, num_points, hidden_size):

        super(samplenet, self).__init__()
        self.num_points = num_points
        self.hidden_size = hidden_size

        self.fc1x = torch.nn.Linear(1, self.hidden_size)
        #torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,2))#
        self.fc2x = torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.fc3x = torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.fc4x = torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.fc5x = torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.fc6x = torch.nn.Linear(self.hidden_size,self.hidden_size)

        self.fc7x = torch.nn.Linear(self.hidden_size, 2 )

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.fc1y = torch.nn.Linear(1, self.hidden_size)
        # torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,2))#
        self.fc2y = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3y = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4y = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc5y = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6y = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.fc7y = torch.nn.Linear(self.hidden_size, 2)

        return

    def forward(self, x,y):

        print(x.shape)
        x  = self.fc1x(x)
        print(x.shape)
        x  = self.relu(x)
        print(x.shape)
        x= self.fc2x(x)
        print(x.shape)
        x = self.relu(x)
        x = self.fc3x(x)
        x = self.relu(x)
        x = self.fc4x(x)
        x = self.relu(x)
        x = self.fc5x(x)
        x = self.relu(x)
        x = self.fc6x(x)
        x = self.relu(x)
        x = self.fc7x(x)

        #y seperately
        print(y.shape)
        y = self.fc1y(y)
        print(x.shape)
        y = self.relu(y)
        print(x.shape)
        y = self.fc2y(y)
        print(x.shape)
        y = self.relu(y)
        y = self.fc3y(y)
        y = self.relu(y)
        y = self.fc4y(y)
        y = self.relu(y)
        y = self.fc5y(y)
        y = self.relu(y)
        y = self.fc6y(y)
        y = self.relu(y)
        y = self.fc7y(y)

        return x,y

