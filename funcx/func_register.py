from funcx.sdk.client import FuncXClient
import time

fxc = FuncXClient()


def mnist_func0(data):
    import numpy as np
    import pyopencl as cl
    import pyopencl.cltypes
    
    binaryFile = '/home/myun7/funcx-fpga/adder.xclbin'
    source_input = data
    size_in = 28*28*1

    if data.size != size_in:
        return 'Input data size should be (28, 28, 1).'

    platform_id = None
    platforms = cl.get_platforms()
    for platform in platforms:
        if platform.name == 'Xilinx':
            platform_id = platforms.index(platform)

    if not platform_id:
        return 'No Xilinx platform is found.'

    devices = platforms[platform_id].get_devices()
    device = devices[0]

    ctx = cl.Context(devices = [device])
    if not ctx:
        return "Fail to create context."
    
    print("Loading xclbin...")
    with open(binaryFile, "rb") as f:
        binary = f.read()
    

    queue = cl.CommandQueue(ctx, device, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    try:
        print('Trying to program device')
        prg = cl.Program(ctx, [device], [binary])
        prg.build()
        print('Device programmed successful!')
    except:
        return 'Fail to program device.'
    
    krnl_mnist = cl.Kernel(prg, 'adder') # create kernel
    

    source_input = (data*2**10).flatten().astype(dtype=np.int16)
    source_output = np.empty((10,), dtype=np.int16)

    # create memory buffers
    mf = cl.mem_flags
    buffer_input = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=source_input)
    buffer_output = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=source_output)
    
    # set kernel args
    krnl_mnist.set_args(buffer_input, buffer_output)
    
    try:
        cl.enqueue_migrate_mem_objects(queue, [buffer_input], flags=0)
        cl.enqueue_nd_range_kernel(queue, krnl_mnist, [1], [1])
        cl.enqueue_migrate_mem_objects(queue, [buffer_output], flags=cl.mem_migration_flags.HOST)
    except:
        return 'Failed to launch kernel.'


    queue.finish()

    source_output = source_output/(2**10)

    return source_output

group_id = 'd1fc601e-8cc7-11eb-ba6a-df4131a75f9b'
func_uuid = fxc.register_function(mnist_func0, group=group_id, description='testing mnist model on fpga', public=False, searchable=True)
print('Function uuid:', func_uuid)


# if __name__ == "__main__":
#     import numpy as np
#     data_dir = '/home/myun7/funcx-fpga/data/test_X.npy'
#     label_dir = '/home/myun7/funcx-fpga/data/test_Y.npy'

#     data =  np.load(data_dir) # [200,28,28,1]
#     label =  np.load(label_dir)

#     result = mnist_func0(data[0])
#     print(result)