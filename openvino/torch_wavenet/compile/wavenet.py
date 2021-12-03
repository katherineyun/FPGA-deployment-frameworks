import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WaveBlock(nn.Module):
    def __init__(self, dilation_rate=1, initial=False):
        super(WaveBlock, self).__init__()
        
        if initial:
            self.pre_process = nn.Conv1d(in_channels = 1, out_channels = 16, kernel_size = 1)
        else:
            self.pre_process = nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size = 1)
        
        self.filter = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 2, dilation = dilation_rate)
        self.gating = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 2, dilation = dilation_rate)
        
        self.post_process = nn.Conv1d(in_channels = 32, out_channels = 16, kernel_size = 1)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.dil_rate = dilation_rate

    def forward(self, x):

        x = self.relu( self.pre_process(x) )

        x_f = F.pad(self.filter(x), (0, self.dil_rate), mode='constant')
        x_g = F.pad(self.gating(x), (0, self.dil_rate), mode='constant')
        
        z = self.tanh(x_f) * self.sigmoid(x_g)
        z = self.relu( self.post_process(z))

        x = x + z

        return x, z

class WaveNet(nn.Module):
    def __init__(self):
        super(WaveNet, self).__init__()

        self.dilation_rates = [2**i for i in range(13)] * 3
        self.last_n_steps = 1000
        self.blocks = nn.ModuleList([WaveBlock(self.dilation_rates[0], initial=True),] + [WaveBlock(rate) for rate in self.dilation_rates[1:]])

        # Tail
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels = 16, out_channels = 128, kernel_size = 1)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(in_channels = 128, out_channels = 1, kernel_size = 1)

        # q branch
        self.q_dense_1 = nn.Linear(in_features=self.last_n_steps, out_features=512)
        self.q_dense_2 = nn.Linear(in_features=512, out_features=1)


        #s1,s2,chi,sigma branch
        self.s_dense_1 = nn.Linear(in_features=self.last_n_steps, out_features=512)
        self.s_dense_2 = nn.Linear(in_features=512, out_features=4)

    def forward(self, x):
        
        skips = []
        for block in self.blocks:
            x, z = block(x)
            skips.append(z)

        # update...
        out = torch.zeros_like(z)
        for s in skips:
            out = out + s

        out = self.relu(out)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        # Slice  
        pred_seq = out[:,:,-self.last_n_steps:]

        # Flatten
        bs = pred_seq.shape[0]
        out = torch.reshape(pred_seq, (bs,-1))

        # q branch
        out_1 = self.q_dense_1(out)
        pred_q = self.q_dense_2(out_1)

        # s1,s2,chi,sigma branch
        out_2 = self.s_dense_1(out)
        pred_ap = self.s_dense_2(out_2)

        return pred_q, pred_ap

#wavenet = WaveNet()

#x = np.random.random((8,1, 10130)).astype('float32')
#x = torch.tensor(x)

#q, ap = wavenet(x)
