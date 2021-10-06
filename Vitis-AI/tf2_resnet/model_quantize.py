from tensorflow_model_optimization.quantization.keras import vitis_quantize
import tensorflow as tf
import numpy as np

model = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights='imagenet',  classes=1000)

eval_dataset = np.random.rand(20, 3, 224, 224)



quantizer = vitis_quantize.VitisQuantizer(model)
quantized_model = quantizer.quantize_model(calib_dataset=eval_dataset)

quantized_model.save('quantized_model.h5')
