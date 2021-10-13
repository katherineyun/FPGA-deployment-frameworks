import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dilation_rate):
        super(Model, self).__init__()
        self.layer1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (1, 1))

        # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1 -> p = floor(d / 2)

        self.layer2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (1, 2), dilation = dilation_rate, padding = (0, int((dilation_rate) / 2)))
        self.layer3 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = (1, 2), dilation = dilation_rate, padding = (0, int((dilation_rate) / 2)))

    def forward(self, x):
        
        x = self.layer1(x)
        print(x.size())
        x = self.layer2(x)
        print(x.size())
        x = self.layer3(x)
        print(x.size())
        return x
    

model = Model(dilation_rate=1)
input = torch.ones([1, 1, 1, 1024])
output = model(input)