from pytorch_nndct.apis import torch_quantizer, dump_xmodel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model import BraggNN
file_path = "./fc16_8_4_2-sz11.pth"

model = BraggNN(imgsz=11, fcsz=(16,8,4,2))
model.load_state_dict(torch.load(file_path, map_location=device))

quant_mode = "test"
deploy = "True"

batch_size = 4
input = torch.randn([batch_size, 1, 11, 11])
quantizer = torch_quantizer(
        quant_mode, model, (input), device=device)

quant_model = quantizer.quant_model

def evaluate(model):

  model.eval()
  model = model.to(device)
  data_block = torch.randn([4, 1, 11, 11])

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