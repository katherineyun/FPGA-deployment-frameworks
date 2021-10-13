from pytorch_nndct.apis import torch_quantizer, dump_xmodel
import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# file_path = "/workspace/test/torch_resnet/resnet18.pth"
# model = resnet18().cpu()
# model.load_state_dict(torch.load(file_path))
from model_arch import WaveNet

model = WaveNet()

quant_mode = "calib"
deploy = "False"

batch_size = 4
input = torch.randn([batch_size, 1, 1, 1024])
quantizer = torch_quantizer(
        quant_mode, model, (input), device=device)

quant_model = quantizer.quant_model

def evaluate(model):

  model.eval()
  model = model.to(device)
  data_block = torch.randn([4, 1, 1, 2024])

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