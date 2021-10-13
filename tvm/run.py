import os
import argparse
import numpy as np
import time


from PIL import Image
from tvm.contrib.download import download_testdata
from tvm.contrib.vai import extern_accel
from tvm.contrib.vai.tvmruntime_util import TVMRuntimeUtil

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

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

def run(fdir, dpu_rundir,shape_dict, iterations):

    # SETUP
    extern_accel.setDpuRunDir(dpu_rundir)


    # DOWNLOAD IMAGE FOR TEST
    img_shape=shape_dict[list(shape_dict.keys())[0]][2:]
    print(img_shape)
    image = Image.open(img_path).resize(img_shape)

    # IMAGE PRE-PROCESSING
    image = transform_image(image)

    # RUN #
    inputs = {}
    inputs[list(shape_dict.keys())[0]] = image


    # VAI FLOW
    tru = TVMRuntimeUtil(fdir)
    for i in range(iterations):
        start = time.time()
        res = tru.run(inputs)
        stop = time.time()

        print("VAI iteration: {}/{}, run time: {}".format(i+1, iterations, stop - start))

        # PREDICTIONS #
        for idx, prediction in enumerate(res[0]):
            print('-----------------------')
            top_k = prediction.argsort()[-1:-(5+1):-1]
            print('TVM-VAI prediction top-5:')
            for pred in top_k:
                print (pred,synset[pred])

    del extern_accel.RUNNER_CACHE

###############################################################################
# RUN MXNET_RESNET_18
#
# Before running the mxnet_resnet_18 model, you have to compile the model
# using the script provided at /tvm/tutorials/accelerators/compile/mxnet_resnet_18.py
# The compile script generates an output file name "mxnet_resnet_18"
# Once you setup your device, you could run the model as follows:
#
# Parameter settings for the run script:
# -f           : Path to directory containing TVM compilation files
# -d           : Path to directory containing DPU model lib
# --iterations : The number of iterations to run the model
#
# example:
# ./mxnet_resnet_18.py -f /PATH_TO_DIR/mxnet_resnet_18 -d /PATH_TO_DIR/mxnet_resnet_18/libdpu --iterations 1

##############################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="Path to directory containing TVM compilation files", default=FILE_DIR)
    parser.add_argument("-d", help="Path to directory containing DPU model lib and meta.json", required=True)
    parser.add_argument("--iterations", help="The number of iterations to run.", default=5, type=int)
    args = parser.parse_args()
    fdir = args.f if os.path.isabs(args.f) else os.path.join(os.getcwd(), args.f)
    dpu_rundir = args.d if  os.path.isabs(args.d) else os.path.join(os.getcwd(), args.d)
    iterations = args.iterations
    shape_dict = {'input_1': [1, 3, 224, 224]}

    start = time.time()
    run(fdir, dpu_rundir, shape_dict, iterations)
    print(time.time() - start)

