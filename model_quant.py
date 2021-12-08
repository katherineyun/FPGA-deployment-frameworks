import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = "/workspace/test/torch_resnet/resnet18.pth"
model = resnet50(pretrained=True).cpu()
# model.load_state_dict(torch.load(file_path))


# quant_mode = "calib"
# deploy = "False"
quant_mode = "test"
deploy = "True"

batch_size = 4
input = torch.randn([batch_size, 3, 224, 224])
quantizer = torch_quantizer(
        quant_mode, model, (input), device=device)

quant_model = quantizer.quant_model

def evaluate(model):

  model.eval()
  model = model.to(device)
  data_block = torch.randn([4, 3, 224, 224])

  for data in data_block:
    data = data.unsqueeze(0)
    pred = model(data)

  return 

evaluate(quant_model)

# handle quantization result
if quant_mode == 'calib':
    quantizer.export_quant_config()
if deploy:
    quantizer.export_xmodel(deploy_check=False)