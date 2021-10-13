import os
import argparse
import numpy as np
import time


from PIL import Image
from tvm.contrib.download import download_testdata
from tvm.contrib.vai import extern_accel
from tvm.contrib.vai.tvmruntime_util import TVMRuntimeUtil
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_name = 'cat.png'
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'imagenet1000_clsid_to_human.txt'
synset_path = download_testdata(synset_url, synset_name, module='data')

with open(synset_path) as f:
    synset = eval(f.read())

img_path = download_testdata(img_url, 'cat.png', module='data')

img_path = os.path.join(os.environ.get("HOME"),'CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min')
img_names = [f for f in os.listdir(img_path) if f.endswith(('JPEG', 'jpg', 'png'))]

def run(fdir, dpu_rundir,shape_dict, iterations):
    # SETUP
    #extern_accel.setDpuRunDir(dpu_rundir)


    # DOWNLOAD IMAGE FOR TEST
    img_shape=shape_dict[list(shape_dict.keys())[0]][2:]
    
    # import labels for accuracy calculation

    gt_label = {}
    with open('./gt.txt', 'r') as golden:
        line = golden.readline()
        while line:
            img_name, label =  line.split(' ')
            gt_label[img_name] = label
            line = golden.readline()
    golden.close()

    model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet', input_shape = (224,224,3))

    # VAI FLOW
    #tru = TVMRuntimeUtil(fdir)
    total_time = 0
    top1_acc, top5_acc = 0, 0
    for i in range(iterations):
        image = cv2.imread(img_path + '/' + img_names[i])
        image = cv2.resize(image, (224,224))
        try:
            image = np.array(image)
            image = image - np.array([103.939, 116.779, 123.68])
            image = image[np.newaxis, :]
            #image = image.transpose([0, 3, 1, 2])
        except Exception:
            continue
 
        #print(image.shape)
        start = time.time()
        #res = tru.run({'input_1': image})
        keras_out = model.predict(image)
        stop = time.time()
        print("VAI iteration: {}/{}, run time: {}".format(i+1, iterations, stop - start))
        # for idx, prediction in enumerate(res[0]):
        #     print('-----------------------')
        #     top_k = prediction.argsort()[-1:-(5+1):-1]
        #     print('TVM-VAI prediction top-5:')
        #     for pred in top_k:
        #         print (pred,synset[pred])
        if i != 1:
            total_time += (stop - start)
        # PREDICTIONS #
        # for idx, prediction in enumerate(res[0]):
        #     #print('-----------------------')
        #     top_k = prediction.argsort()[-1:-(5+1):-1]
        # #     #print('TVM-VAI prediction')
        # #     # update accuracy
        #     gt = gt_label[img_names[i]][:-1]
        #     if top_k[0] == int(gt):
        #         top1_acc += 1
        #     for j in top_k:
        #         if j == int(gt):
        #             top5_acc += 1

        top_k = keras_out.argsort()[0][-1:-(5+1):-1]
        print(top_k) 
        gt = gt_label[img_names[i]][:-1]
        if top_k[0] == int(gt):
           top1_acc += 1
        for j in top_k:
           if j == int(gt):
               top5_acc += 1

    print("Top 1 and Top 5 accuracies are {} and {}.".format(top1_acc/(iterations), top5_acc/iterations))
    print("Average prediction time: ", total_time/iterations)

    #del extern_accel.RUNNER_CACHE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="Path to directory containing TVM compilation files", default=FILE_DIR)
    parser.add_argument("-d", help="Path to directory containing DPU model lib and meta.json", required=True)
    parser.add_argument("--iterations", help="The number of iterations to run.", default=500, type=int)
    args = parser.parse_args()
    fdir = args.f if os.path.isabs(args.f) else os.path.join(os.getcwd(), args.f)
    dpu_rundir = args.d if  os.path.isabs(args.d) else os.path.join(os.getcwd(), args.d)
    iterations = args.iterations
    shape_dict = {'input_1': [1, 3, 224, 224]}

    start = time.time()
    run(fdir, dpu_rundir, shape_dict, iterations)
    #print(time.time() - start)






