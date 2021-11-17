from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from pytorch_nndct.nn.modules import functional
import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (1, 1))
        
        self.layer2_pad = nn.ZeroPad2d((0, 1, 0, 0))
        self.layer2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (1, 2))
        
        self.layer3_pad = nn.ZeroPad2d((0, 1, 0, 0))
        self.layer3 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = (1, 2))
        
        self.skip_add = functional.Add()

    def forward(self, x):
        
        k = self.layer1(x)
        
        print(k.size())
        x = self.layer2_pad(k)
        x = self.layer2(x)
        
        x = self.layer3_pad(x)
        x = self.layer3(x)
        
        # print(x.size(), k.size())
        # x = self.skip_add(x, k)

        x = x + k
       
        return x

# model to test branching (2 and 3)
# class Model(nn.Module):
#     def __init__(self, dilation_rate):
#         super(Model, self).__init__()
#         self.layer1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (1, 1))

#         # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1 -> p = floor(d / 2)
#         self.layer2_pad = nn.ZeroPad2d((0, dilation_rate, 0, 0))
#         self.layer2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (1, 2), dilation = dilation_rate)
#         self.layer3_pad = nn.ZeroPad2d((0, dilation_rate, 0, 0))
#         self.layer3 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = (1, 2), dilation = dilation_rate)

#         self.layer4_pad = nn.ZeroPad2d((0, dilation_rate, 0, 0))
#         self.layer4 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = (1, 2), dilation = dilation_rate)
        

#     def forward(self, x):
        
#         x = self.layer1(x)
        
        
#         x = self.layer2_pad(x)
#         x = self.layer2(x)
        
        
#         x1 = self.layer3_pad(x)
#         out1 = self.layer3(x1)
        
#         x2 = self.layer4_pad(x)
#         out2 = self.layer4(x2)

#         print(x1.size(), x2.size())
#         return x, out1, out2

model = Model()

# quant_mode = "calib"
# deploy = "False"

quant_mode = "test"
deploy = "True"

batch_size = 4
input = torch.randn([batch_size, 1, 1, 1024])
quantizer = torch_quantizer(
        quant_mode, model, (input), device=device)

quant_model = quantizer.quant_model

def evaluate(model):

  model.eval()
  model = model.to(device)

  x = np.random.random((1, 1, 1, 1024)).astype('float32')
  x = torch.tensor(x)

  pred = model(x)
  return 

evaluate(quant_model)

# handle quantization result
if quant_mode == 'calib':
    quantizer.export_quant_config()
if deploy:
    quantizer.export_xmodel(deploy_check=False)

    