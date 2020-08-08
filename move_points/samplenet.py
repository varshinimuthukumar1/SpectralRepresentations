import os, sys

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

        self.fc1 = torch.nn.Linear(num_points*2, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size*3)
        self.fc4 = torch.nn.Linear(self.hidden_size*3, self.hidden_size*3)
        self.fc5 = torch.nn.Linear(self.hidden_size*3, self.hidden_size)
        self.fc6 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc7 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc8 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc9 = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.fc10 = torch.nn.Linear(self.hidden_size, num_points*2)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input):


        x= input.view(-1,input.shape[1]*2, 1)

        x= x.squeeze(2)

        x = self.fc1(x)

        x  = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        #x = self.fc6(x)
        #x = self.fc7(x)
        #x = self.fc8(x)
        #x = self.fc9(x)

        x = self.fc10(x)
        x = self.relu(x)

        #x = self.sigmoid(x)

        x= x.unsqueeze(2)
        x = x.view(-1,input.shape[1],2)


        return x