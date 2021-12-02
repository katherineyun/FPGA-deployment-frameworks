from funcx.sdk.client import FuncXClient
import time
import numpy as np

fxc = FuncXClient()

# result = fxc.search_function("mnist_func1", offset=0)

target_endpoint = 'ea8eaa35-bfff-4037-8bbf-ffd3f2f7d12f' # fpga-test

func_uuid = '4dc11381-9d7a-4855-8d44-4e0004bf4f88'

data_dir = '/home/myun7/funcx-fpga/data/test_X.npy'
label_dir = '/home/myun7/funcx-fpga/data/test_Y.npy'
data =  np.load(data_dir) # [200,28,28,1]
label =  np.load(label_dir)

idx = 5

t1 = time.time()
res = fxc.run(data[idx], endpoint_id=target_endpoint, function_id=func_uuid)
#print('Function run_id: ', res)
print(time.time()-t1)

time.sleep(3)

t2 = time.time()
result = fxc.get_result(res)
print(time.time()-t2)

print('Output:', result)
print('Prediction:', np.argmax(result))
print('Label:', np.argmax(label[idx]))