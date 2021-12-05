import numpy as np
import torch
import time
from wavenet import WaveNet
from openvino.inference_engine import IECore
from pathlib import Path

LENGTH = 10130
BASE_MODEL_NAME = f"wavenet{LENGTH}"
# Paths where PyTorch, ONNX and OpenVINO IR models will be stored
model_path = Path(BASE_MODEL_NAME).with_suffix(".pth")
onnx_path = model_path.with_suffix(".onnx")
ir_path = model_path.with_suffix(".xml")

input_image = np.random.random((1, 1, LENGTH))


# Run inference on the input image
print("###########################################")
model = WaveNet()
with torch.no_grad():
    start = time.perf_counter()
    model(torch.as_tensor(input_image).float())
    end = time.perf_counter()
    time_torch = end - start

print(
    f"PyTorch model on CPU: {time_torch/1:.3f} seconds per image, "
    f"FPS: {1/time_torch:.2f}"
)

print("###########################################")
# Load network to Inference Engine
ie = IECore()
net_onnx = ie.read_network(model=onnx_path)
exec_net_onnx = ie.load_network(network=net_onnx, device_name="CPU") 

input_layer_onnx = next(iter(exec_net_onnx.input_info))
output_layer_onnx = next(iter(exec_net_onnx.outputs))

# Run inference on the input image
start = time.perf_counter()
res_onnx = exec_net_onnx.infer(inputs={input_layer_onnx: input_image})
end = time.perf_counter()
time_onnx = end - start

print(
    f"ONNX model in Inference Engine/CPU: {time_onnx/1:.3f} "
    f"seconds per image, FPS: {1/time_onnx:.2f}"
)
res_onnx_q = res_onnx['844']
res_onnx_ap = res_onnx['846']
print("ONNX Output result: ", res_onnx_q, res_onnx_ap)


print("###########################################")
# Load the network in Inference Engine
ie = IECore()
net_ir = ie.read_network(model=ir_path)
exec_net_ir = ie.load_network(network=net_ir, device_name="CPU")

# Get names of input and output layers
input_layer_ir = next(iter(exec_net_ir.input_info))
output_layer_ir = next(iter(exec_net_ir.outputs))

# Run inference on the input image
start = time.perf_counter()
res_ir = exec_net_ir.infer(inputs={input_layer_ir: input_image})
end = time.perf_counter()
time_ir = end - start

print(
    f"IR model in Inference Engine/CPU: {time_ir/1:.3f} "
    f"seconds per image, FPS: {1/time_ir:.2f}"
)
res_ir_q = res_ir['844']
res_ir_ap = res_ir['846']
print("IR Output result: ", res_ir_q, res_ir_ap)



