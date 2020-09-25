import numpy as np  # linear algebra

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid


def weights_init_kaimingUniform(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, a=0, b=1)
            nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.)


class NN(nn.Module):

    def __init__(self, x_dim):
        super(NN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(x_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            nn.Linear(10, 3),
            nn.Tanh()
        )
        weights_init_kaimingUniform(self)

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        #x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class Attention_NN(nn.Module):

    def __init__(self, x_dim):
        super(Attention_NN, self).__init__()
        self.mask_layers = nn.Sequential(
            nn.Linear(x_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            nn.Linear(64, x_dim),
            nn.ReLU(),
            nn.Softmax()
        )

        self.input_mask_l = nn.Linear(64, x_dim)
        self.input_mask_activate = nn.Softmax()
        self.output_mask_l = nn.Linear(64, x_dim),
        self.output_mask_activate = nn.Softmax()

        self.reg_layers = nn.Sequential(
            nn.Linear(x_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            nn.Linear(10, 3),
            nn.Tanh()
        )
        weights_init_kaimingUniform(self)

    def forward(self, x):
        mask = self.mask_layers(x)
        #weighted_x = torch.mul(mask, x)
        weighted_x = mask * x
        x = self.reg_layers(weighted_x)
       # print(mask)
        return x  # , mask


class Extra_Attention_NN(nn.Module):

    def __init__(self, x_a_dim, x_dim):
        super(Extra_Attention_NN, self).__init__()
        self.mask_layers = nn.Sequential(
            nn.Linear(x_a_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, x_dim),
            nn.Softmax()
        )

        self.reg_layers = nn.Sequential(
            nn.Linear(x_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
            nn.Tanh()
        )
        weights_init_kaimingUniform(self)

    def forward(self, x_a, x):
        mask = self.mask_layers(x_a)
        weighted_x = torch.mul(mask, x)
        x = self.reg_layers(weighted_x)
        mask_arr = mask.cpu().detach().numpy()
        # for m in mask_arr:
        #    print(m)
        return x  # , mask


class Attention_NN_dual(nn.Module):

    def __init__(self, x_dim):
        super(Attention_NN_dual, self).__init__()
        self.mask_layers = nn.Sequential(
            nn.Linear(x_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=0.35),
            nn.ReLU(),
        )

        self.input_mask_l = nn.Linear(64, x_dim)
        self.input_mask_acti = nn.Softmax()
        self.output_mask_l = nn.Linear(64, 3)
        self.output_mask_acti = nn.Softmax()

        self.reg_layers = nn.Sequential(
            nn.Linear(x_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            nn.Linear(10, 3),
            nn.Tanh()
        )
        weights_init_kaimingUniform(self)

    def forward(self, x):
        mask = self.mask_layers(x)
        in_mask = self.input_mask_l(mask)
        in_mask = self.input_mask_acti(in_mask)
        weighted_x = in_mask * x  # + x

        out_mask = self.output_mask_l(mask)
        out_mask = self.output_mask_acti(out_mask)

        #weighted_x = torch.mul(mask, x)

        x = self.reg_layers(weighted_x)
        x = out_mask * x
        return x  # , in_mask, out_mask


class Extra_Attention_NN_dual(nn.Module):

    def __init__(self, x_a_dim, x_dim):
        super(Extra_Attention_NN_dual, self).__init__()
        self.mask_layers = nn.Sequential(
            nn.Linear(x_a_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            #nn.Linear(64, x_dim),
            # nn.Sigmoid()
        )

        self.input_mask_l = nn.Linear(64, x_dim)
        self.input_mask_acti = nn.Softmax()
        self.output_mask_l = nn.Linear(64, 3)
        self.output_mask_acti = nn.Softmax()

        self.reg_layers = nn.Sequential(
            nn.Linear(x_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.Dropout(p=0.35),
            nn.ReLU(),
            nn.Linear(10, 3),

        )
        self.final_activation = nn.Tanh()
        weights_init_kaimingUniform(self)

    def forward(self, x_a, x):
        """
        mask = self.mask_layers(x_a)
        weighted_x = torch.mul(mask, x)
        x = self.reg_layers(weighted_x)
        """
        mask = self.mask_layers(x_a)
        in_mask = self.input_mask_l(mask)
        in_mask = self.input_mask_acti(in_mask)
        weighted_x = in_mask * x  # + x

        out_mask = self.output_mask_l(mask)
        out_mask = self.output_mask_acti(out_mask)

        #weighted_x = torch.mul(mask, x)

        x = self.reg_layers(weighted_x)
        x = out_mask * x
        x = self.final_activation(x)
        #print("in_mask", in_mask[:, 0].shape)
        #print(out_mask, "\n_____________________")
        return x  # , in_mask, out_mask


class Attention_CNN(object):
    """docstring for Attention_CNN"""

    def __init__(self, arg):
        super(Attention_CNN, self).__init__()
        self.arg = arg


class Attention_Hybrid(object):
    """docstring for Attention_Hybrid"""

    def __init__(self, arg):
        super(Attention_Hybrid, self).__init__()
        self.arg = arg
