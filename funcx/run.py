from funcx.sdk.client import FuncXClient
import time
import numpy as np

fxc = FuncXClient()

target_endpoint = '92565a2f-5e97-4226-909c-21a8875c4a23' # fpga-test
#func_uuid = 'f1b1f962-fe00-457d-ae52-05417b005317'
func_uuid = '44b1a2f6-6765-4648-8dc4-82f9fb21cead'

data_dir = '/home/myun7/funcx-fpga/data/test_X.npy'
label_dir = '/home/myun7/funcx-fpga/data/test_Y.npy'
data =  np.load(data_dir) # [200,28,28,1]
label =  np.load(label_dir)

idx = 23
res = fxc.run(data[idx], endpoint_id=target_endpoint, function_id=func_uuid)
#print('Function run_id: ', res)

time.sleep(0.5)
result = fxc.get_result(res)
print('Output:', result)
print('Prediction:', np.argmax(result))
print('Label:', np.argmax(label[idx]))
