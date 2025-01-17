# nn - Neural Networks in C

This repository implements various neural networks in C, focusing mainly on targetting embedded systems or creating hardware accelerators (FPGA-Based, ASIC, etc.) \
This project was created as part of my independent study course, where I am currently researching the design of hardware accelerators for high-performance workloads

### current implementations (project index)
`snn.c` - A simple feedforward neural network written in ~150loc. Depends on c native libraries and [GSL](https://www.gnu.org/software/gsl/) \
`cnn.c` - Implements a fully featured cnn library in ~600loc. Depends solely on C native libraries \
`cnn-hls.c` - The version of `cnn.c` with HLS specific optimizations (Pragmas, Systolic Array Mutliplication, etc.); aims at being synthesized through Vitus HLS to create a FPGA Based CNN Accelerator \
`mnist.c` - Driver code for `cnn.c` which trains on the [MNIST](https://yann.lecun.com/exdb/mnist/) dataset

### usage
`mnist.c` is a great example of how the library is used, but basic usage boils down to a few simple things: \

1) Importing `cnn.c` into your code
2) Creating a network and creating layers:
```c
// an example of a lenet-5 inspired 8 layer network
Network* network = create_network(8);
network->layers[0] = create_input(IMG_HEIGHT, IMG_WIDTH, 1);
network->layers[1] = create_conv(IMG_HEIGHT, IMG_WIDTH, 1, 6, 5, 1, 2);
network->layers[2] = create_maxpool(network->layers[1]->height, network->layers[1]->width, network->layers[1]->channels, 2, 2);
network->layers[3] = create_conv(network->layers[2]->height, network->layers[2]->width, network->layers[2]->channels, 16, 5, 1, 0);
network->layers[4] = create_maxpool(network->layers[3]->height, network->layers[3]->width, network->layers[3]->channels, 2, 2);
network->layers[5] = create_fc(120, network->layers[4]->height * network->layers[4]->width * network->layers[4]->channels, a_sigmoid);
network->layers[6] = create_fc(84, 120, a_sigmoid);
network->layers[7] = create_fc(NUM_CLASSES, 84, a_softmax);
```
3) Forward and backpropogation through the Network!

## Project Overview and Explanation
### Abstract
For my project, I propose an implementation of a Convolutional Neural Network based handwritten digital classifier using the MNIST dataset on a Field Programmable Gate Array (FPGA). I utilize High Level Synthesis (HLS) tool called Vitus HLS developed by [AMD/Xilinx](https://www.xilinx.com/products/boards-and-kits.html) in order to implement the accelerator through C, eliminating the need to write any code in HDL Languages such as Verilog/VHDL. To reduce any performance losses, I implement a systolic array based architecture and utilize techniques such as pipelining, loop unrolling, and memory partitioning. Through this project, I aim to highlight the potential of FPGAs to offer reduced power consumption and latency for machine learning tasks, creating a more sustainable computing environment.
