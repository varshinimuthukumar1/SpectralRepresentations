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

        self.fc1 = torch.nn.Linear(2,self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = torch.nn.Linear(self.hidden_size, 2)

    def forward(self, x):

        print(x.shape)
        x  = self.fc1(x)

        x  = F.relu(x)
        x= self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x

