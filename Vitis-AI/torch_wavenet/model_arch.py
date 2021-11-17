import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_nndct.nn.modules import functional

class WaveBlock(nn.Module):
    def __init__(self, dilation_rate=1, initial=False):
        super(WaveBlock, self).__init__()

        if initial:
            self.pre_process = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (1, 1))
        else:
            self.pre_process = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (1, 1))

        # check padding = 'same' equivalent
        self.f_pad = nn.ZeroPad2d((0, dilation_rate, 0, 0))
        self.filter = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (1, 2), dilation = dilation_rate) 
        self.g_pad = nn.ZeroPad2d((0, dilation_rate, 0, 0))
        self.gating = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (1, 2), dilation = dilation_rate)
        
        self.post_process = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = (1, 1))
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.skip_add = functional.Add()


    def forward(self, x):

        x = self.relu( self.pre_process(x) )
        
        
        x_f = self.filter(self.f_pad(x))
        x_g = self.gating(self.g_pad(x))
    
        z = self.tanh(x_f) * self.sigmoid(x_g)

        z = self.relu(self.post_process(x_g))

        x += z

        return x, z

class WaveNet(nn.Module):
    def __init__(self):
        super(WaveNet, self).__init__()

        self.dilation_rates = [2**i for i in range(3)] * 7 # 14
        self.last_n_steps = 1000
        self.blocks = nn.ModuleList([WaveBlock(self.dilation_rates[0], initial=True),] + [WaveBlock(rate) for rate in self.dilation_rates[1:]])

        # Tail
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels = 16, out_channels = 128, kernel_size = (1, 1))
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = (1, 1))

        # q branch
        self.q_dense_1 = nn.Linear(in_features=self.last_n_steps, out_features=512)
        self.q_dense_2 = nn.Linear(in_features=512, out_features=1)


        #s1,s2,chi,sigma branch
        self.s_dense_1 = nn.Linear(in_features=self.last_n_steps, out_features=512)
        self.s_dense_2 = nn.Linear(in_features=512, out_features=4)

        self.skip_add = functional.Add()

        self.identity = nn.Identity()

    def forward(self, x):
        
        input_len = x.shape[3].item()
 
        skips = []

        for block in self.blocks:
            x, z = block(x)
            skips.append(z)

        out = x - x

        for s in skips:
            out += s
        
        out = self.relu(out)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        # out - [4, 1, 1, 1024]

        # Slice
        pred_seq = out[:,:,:,(input_len-self.last_n_steps):] # torch.Size([4, 1, 1, 1000])

       
        # Flatten
        bs = pred_seq.shape[0]
        out = torch.reshape(pred_seq, (bs,-1)) # torch.Size([4, 1000])
        
        # q branch

        out_1 = self.q_dense_1(out)
        
        pred_q = self.q_dense_2(out_1)

        # s1,s2,chi,sigma branch
        out_2 = self.s_dense_1(out)
        pred_ap = self.s_dense_2(out_2)

        return pred_q, pred_ap