import sys
import time
from pathlib import Path
from wavenet import WaveNet
import cv2
import numpy as np
import torch

from IPython.display import Markdown, display
from openvino.inference_engine import IECore
LENGTH = 4096
model = WaveNet()
BASE_MODEL_NAME = f"wavenet{LENGTH}"
# Paths where PyTorch, ONNX and OpenVINO IR models will be stored
model_path = Path(BASE_MODEL_NAME).with_suffix(".pth")
onnx_path = model_path.with_suffix(".onnx")
ir_path = model_path.with_suffix(".xml")


if not onnx_path.exists():
    dummy_input = torch.randn(1, 1, LENGTH)

    # For the Fastseg model, setting do_constant_folding to False is required
    # for PyTorch>1.5.1
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=True
    )
    print(f"ONNX model exported to {onnx_path}.")
else:
    print(f"ONNX model {onnx_path} already exists.")


mo_command = f"""mo
                 --input_model "{onnx_path}"
                 --input_shape "[1,1, 1, {LENGTH}]"
                 --mean_values="[100]"
                 --scale_values="[50]"
                 --data_type FP16
                 --output_dir "{model_path.parent}"
                 """
mo_command = " ".join(mo_command.split())
print("Model Optimizer command to convert the ONNX model to OpenVINO:")
print(mo_command)
display(Markdown(f"`{mo_command}`"))

# if not ir_path.exists():
#     print("Exporting ONNX model to IR... This may take a few minutes.")
#     mo_result = %sx $mo_command
#     print("\n".join(mo_result))
# else:
#     print(f"IR model {ir_path} already exists.")