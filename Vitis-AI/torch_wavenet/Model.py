# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Model::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[layer1]/input.2
        self.module_2 = py_nndct.nn.Module('pad') #Model::Model/ZeroPad2d[layer2_pad]/input.3
        self.module_3 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[1, 2], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[layer2]/input.4
        self.module_4 = py_nndct.nn.Module('pad') #Model::Model/ZeroPad2d[layer3_pad]/input
        self.module_5 = py_nndct.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[1, 2], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[layer3]/42
        self.module_6 = py_nndct.nn.Add() #Model::Model/44

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_2 = self.module_2(input=self.output_module_1, pad=[0,1,0,0], mode='constant', value=0.0)
        self.output_module_3 = self.module_3(self.output_module_2)
        self.output_module_4 = self.module_4(input=self.output_module_3, pad=[0,1,0,0], mode='constant', value=0.0)
        self.output_module_5 = self.module_5(self.output_module_4)
        self.output_module_6 = self.module_6(alpha=1, other=self.output_module_1, input=self.output_module_5)
        return self.output_module_6
