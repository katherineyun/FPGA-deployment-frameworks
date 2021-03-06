/**
* Copyright (C) 2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/
// OpenCL utility layer include
#include "xcl2.hpp"
#include <vector>
#include "defines.h"

#include "ap_int.h"
#include "ap_fixed.h"

#define DATA_SIZE 4096
//#define INCR_VALUE 10

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    auto binaryFile = argv[1];

    // Allocate Memory in Host Memory
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel krnl_adder;
    //size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
    typedef ap_fixed<16,6> ap_t;

    size_t vector_size_bytes_1 = sizeof(ap_t) * 784;
    size_t vector_size_bytes_2 = sizeof(ap_t) * 10;
    std::vector<ap_t, aligned_allocator<ap_t> >source_input(784);
    std::vector<ap_t, aligned_allocator<ap_t> >source_output(10);
    std::vector<float, aligned_allocator<float> >label(10);
    //std::vector<int, aligned_allocator<int> > source_hw_results(DATA_SIZE);
    //std::vector<int, aligned_allocator<int> > source_sw_results(DATA_SIZE);


    //input image and label
    source_input = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1843137254901961,0.42745098039215684,0.8313725490196079,0.9921568627450981,0.7529411764705882,0.058823529411764705,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.043137254901960784,0.6078431372549019,0.9137254901960784,0.9882352941176471,0.9882352941176471,0.9882352941176471,0.8705882352941177,0.09803921568627451,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.28627450980392155,0.7490196078431373,0.9882352941176471,0.9921568627450981,0.9882352941176471,0.803921568627451,0.2784313725490196,0.1607843137254902,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.24705882352941178,0.9294117647058824,0.9882352941176471,0.9882352941176471,0.9921568627450981,0.5019607843137255,0.12156862745098039,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.24705882352941178,0.9921568627450981,0.9882352941176471,0.9882352941176471,0.7411764705882353,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.611764705882353,0.9294117647058824,0.9921568627450981,0.9058823529411765,0.6196078431372549,0.058823529411764705,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.28627450980392155,0.9098039215686274,0.9882352941176471,0.9921568627450981,0.6196078431372549,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.24705882352941178,0.9294117647058824,0.9882352941176471,0.9882352941176471,0.7490196078431373,0.058823529411764705,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.43137254901960786,0.9921568627450981,0.9921568627450981,0.9921568627450981,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.16470588235294117,0.9098039215686274,0.9882352941176471,0.9882352941176471,0.5019607843137255,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.10588235294117647,0.8666666666666667,0.9882352941176471,0.9882352941176471,0.803921568627451,0.12156862745098039,0.0,0.043137254901960784,0.22745098039215686,0.7098039215686275,0.7137254901960784,0.7098039215686275,0.6274509803921569,0.1450980392156863,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1450980392156863,0.9882352941176471,0.9882352941176471,0.9882352941176471,0.2784313725490196,0.0,0.0,0.6078431372549019,0.9882352941176471,0.9882352941176471,0.9921568627450981,0.9882352941176471,0.9882352941176471,0.9882352941176471,0.32941176470588235,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7137254901960784,0.9921568627450981,0.9921568627450981,0.8705882352941177,0.1607843137254902,0.24705882352941178,1.0,0.9921568627450981,0.9921568627450981,0.9921568627450981,0.8156862745098039,0.9333333333333333,0.9921568627450981,0.9921568627450981,0.7529411764705882,0.058823529411764705,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7098039215686275,0.9882352941176471,0.9882352941176471,0.21568627450980393,0.611764705882353,0.9294117647058824,0.9921568627450981,0.9882352941176471,0.7843137254901961,0.3803921568627451,0.0784313725490196,0.11764705882352941,0.5843137254901961,0.9882352941176471,0.9921568627450981,0.6235294117647059,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.043137254901960784,0.7490196078431373,0.9882352941176471,0.9882352941176471,0.4235294117647059,0.9098039215686274,0.9882352941176471,0.9921568627450981,0.7019607843137254,0.0784313725490196,0.0,0.0,0.0,0.1450980392156863,0.9882352941176471,0.9921568627450981,0.7019607843137254,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.28627450980392155,0.9882352941176471,0.9882352941176471,0.9882352941176471,0.9882352941176471,0.9882352941176471,0.9882352941176471,0.4235294117647059,0.058823529411764705,0.0,0.0,0.0,0.0,0.1450980392156863,0.9882352941176471,0.9921568627450981,0.7019607843137254,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7137254901960784,0.9921568627450981,0.9921568627450981,0.9921568627450981,0.9294117647058824,0.24313725490196078,0.0,0.0,0.0,0.0,0.0,0.12549019607843137,0.5058823529411764,0.9921568627450981,1.0,0.4666666666666667,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2235294117647059,0.9882352941176471,0.9882352941176471,0.9882352941176471,0.9686274509803922,0.5254901960784314,0.28627450980392155,0.28627450980392155,0.28627450980392155,0.28627450980392155,0.28627450980392155,0.8117647058823529,0.9882352941176471,0.9882352941176471,0.6235294117647059,0.0196078431372549,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.12549019607843137,0.8431372549019608,0.9882352941176471,0.9882352941176471,0.9882352941176471,0.9882352941176471,0.9921568627450981,0.9882352941176471,0.9882352941176471,0.9882352941176471,0.9921568627450981,0.9882352941176471,0.9647058823529412,0.8431372549019608,0.1607843137254902,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.00392156862745098,0.4235294117647059,0.5803921568627451,0.9882352941176471,0.9882352941176471,0.9921568627450981,0.9882352941176471,0.9882352941176471,0.9882352941176471,0.9921568627450981,0.8235294117647058,0.3607843137254902,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    label = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    // clear output vector
    for(int i=0; i < 10; i++) {
    	source_output[i] = 0;
    }



    // OPENCL HOST CODE AREA START
    auto devices = xcl::get_xil_devices();

    // Create Program and Kernel
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_adder = cl::Kernel(program, "adder", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // Allocate Buffer in Global Memory
    OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector_size_bytes_1,
                                           source_input.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, vector_size_bytes_2,
                                            source_output.data(), &err));

    //int inc = INCR_VALUE;
    //int size = DATA_SIZE;
    // Set the Kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_adder.setArg(narg++, buffer_input));
    OCL_CHECK(err, err = krnl_adder.setArg(narg++, buffer_output));
    //OCL_CHECK(err, err = krnl_adder.setArg(narg++, inc));
    //OCL_CHECK(err, err = krnl_adder.setArg(narg++, size));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input}, 0 /* 0 means from host*/));

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_adder));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

    // OPENCL HOST CODE AREA END

    // Compare the results of the Device to the simulation
    int match = 0;
    std::cout <<"Predictions:" << std::endl;
    for (int i = 0; i < 10; i++) {

    	std::cout << source_output[i] << " ";
    }

    std::cout <<"\nLabel:" << std::endl;
    for(int i = 0; i < 10; i++) {
          std::cout << label[i] << " ";
    }

    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE : EXIT_SUCCESS);
}

