import os
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger('pyxir')
logger.setLevel(logging.INFO)

import pyxir

import tvm
from tvm import contrib
import tvm.relay as relay
from tvm.contrib.vai import base
from tvm.contrib.vai.relay_transform import PartitioningPass
from tvm.contrib.vai import extern_accel
from tvm.contrib.vai.tvmruntime_util import TVMRuntimeUtil

import time
import cv2

FILE_DIR   = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.getenv('HOME')
######################################################################
# Download Resnet50 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
###############################################################################
from tvm.contrib.download import download_testdata
#from mxnet.gluon.model_zoo.vision import get_model
import tensorflow as tf
from tensorflow import keras
from PIL import Image
#from matplotlib import pyplot as plt
#block = get_model('resnet18_v1', pretrained=True)
block = tf.keras.models.load_model('/workspace/my_model')

img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_name = 'cat.png'
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'imagenet1000_clsid_to_human.txt'
img_path = download_testdata(img_url, 'cat.png', module='data')
synset_path = download_testdata(synset_url, synset_name, module='data')
with open(synset_path) as f:
    synset = eval(f.read())

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

# DOWNLOAD IMAGE FOR TEST
image = Image.open(img_path).resize((224, 224))
image = transform_image(image)

###############################################################################
# MODEL SETTINGS
#
# Parameter settings for compiling a model using tvm-vai flow
# quant_dir      : path to images for quantization
# shape_dict     : dictionary of input names as keys and input shapes as values
#                  dict{input_name:input_shape}
# postprocessing : 'Softmax' if necessary
# target         : hardware accelerator to run the compiled model
#                      options: 'DPUCADX8G', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102'
#                      options(deprecated): 'dpuv1', 'dpuv2-zcu104', 'dpuv2-zcu102'

###############################################################################

quant_dir      = os.path.join(HOME_DIR,'CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min')
shape_dict     = {'input_1': image.shape}
postprocessing = ['Softmax']
target         = 'DPUCADX8G'

###############################################################################
# INPUTS FUNC
#
# Define and inputs function which takes in an iterator value and returns a
# dictionary mapping from input name to array containing dataset inputs. Note
# that the input function should always return image data in NCHW layout as
# all models are converted to NCHW layout internally for Vitis-AI compilation.
#
# This is necessary for quantizating the model for acceleration using Vitis-AI.
###############################################################################

def inputs_func(iter):
    import os

    img_files = [os.path.join(quant_dir, f) for f in os.listdir(quant_dir) if f.endswith(('JPEG', 'jpg', 'png'))][:10]
    size=shape_dict[list(shape_dict.keys())[0]][2:]

    imgs = []
    for path in img_files:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img.astype(np.float32))
    out = []
    for img in imgs:

        img = cv2.resize(img, tuple(size), interpolation=1)
        img = transform_image(img)
        img = img.reshape(img.shape[1:])
        out.append(img)


    res = np.array(out).astype(np.float32)
    print (res.shape)
    input_name = list(shape_dict.keys())[0]
    return {input_name: res}


###############################################################################
# PARTITION & BUILD
#
# Module pass to partition Relay for Vitis-AI acceleration. Targets can be
#  'DPUCADX8G', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102'
# Afterwards build graph, lib and params using standard TVM flow.
##############################################################################

if  target.startswith('DPUCZ') or target.startswith('dpuv2'):
    tvm_target = tvm.target.arm_cpu('ultra96')
    lib_kwargs = {
        'fcompile': contrib.cc.create_shared,
        'cc': "/usr/aarch64-linux-gnu/bin/ld"
    }
else:
    tvm_target = 'llvm'
    lib_kwargs = {}

mod, params = relay.frontend.from_keras(block, shape_dict)

# CUSTOM VAI MODULE PASS

mod = PartitioningPass(target=target, params=params,
                       inputs_func=inputs_func, postprocessing= postprocessing)(mod)

print("Mod", mod)
graph, lib, params = relay.build(
    mod, tvm_target, params=params)

print(" Compilation successfully finished")
###############################################################################
# SAVE OUTPUT
#
# Save the output files for running on the board
##############################################################################
TVM_OUTPUT_DIR = os.path.join(FILE_DIR, "tf_resnet_50")
DPU_OUTPUT_DIR = os.path.join(FILE_DIR, "tf_resnet_50/libdpu")
os.makedirs(DPU_OUTPUT_DIR, exist_ok = True)

lib.export_library(os.path.join(TVM_OUTPUT_DIR,"tvm_dpu_cpu.so"), **lib_kwargs)

with open(os.path.join(TVM_OUTPUT_DIR,"tvm_dpu_cpu.json"),"w") as f:
    f.write(graph)

with open(os.path.join(TVM_OUTPUT_DIR,"tvm_dpu_cpu.params"), "wb") as f:
    f.write(relay.save_param_dict(params))

import glob
from shutil import copy2

if target == 'DPUCADX8G' or target == 'dpuv1':
    DPU_LIBDIR = '/tmp/vai/'
    try:
        copy2(os.path.join(DPU_LIBDIR,"meta.json"), DPU_OUTPUT_DIR)
        copy2(os.path.join(DPU_LIBDIR,"compiler.json"), DPU_OUTPUT_DIR)
        copy2(os.path.join(DPU_LIBDIR,"weights.h5"), DPU_OUTPUT_DIR)
        copy2(os.path.join(DPU_LIBDIR,"quantizer.json"), DPU_OUTPUT_DIR)
    except IOError as e:
        print("Compiled files were not found in directory: ", DPU_LIBDIR)

else:
    DPU_LIBDIR = FILE_DIR
    try:
        copy2(os.path.join(DPU_LIBDIR,"meta.json"), DPU_OUTPUT_DIR)
        copy2(os.path.join(DPU_LIBDIR,glob.glob("libdpu*.so")[0]), DPU_OUTPUT_DIR)
        copy2(os.path.join(DPU_LIBDIR,"weights.h5"), DPU_OUTPUT_DIR)
        copy2(os.path.join(DPU_LIBDIR,"quantizer.json"), DPU_OUTPUT_DIR)
    except IOError as e:
        print("Compiled files were not found in directory: ", DPU_LIBDIR)

