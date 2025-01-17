# nn - Neural Networks in C

This repository implements various neural networks in C, focusing mainly on targetting embedded systems or creating hardware accelerators (FPGA-Based, ASIC, etc.) \
This project was created as part of my independent study course, where I am currently researching the design of hardware accelerators for high-performance workloads

### Current Implementations (project index)
`snn.c` - A simple feedforward neural network written in ~150loc. Depends on c native libraries and [GSL](https://www.gnu.org/software/gsl/) \
`cnn.c` - Implements a fully featured cnn library in ~600loc. Depends solely on C native libraries \
`cnn-hls.c` - The version of `cnn.c` with HLS specific optimizations (Pragmas, Systolic Array Mutliplication, etc.); aims at being synthesized through Vitus HLS to create a FPGA Based CNN Accelerator \
`mnist.c` - Driver code for `cnn.c` which trains on the [MNIST](https://yann.lecun.com/exdb/mnist/) dataset

### Usage
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

## Project Information
### Abstract
For my project, I propose an implementation of a Convolutional Neural Network based handwritten digital classifier using the MNIST dataset on a Field Programmable Gate Array (FPGA). I utilize High Level Synthesis (HLS) tool called Vitus HLS developed by [AMD/Xilinx](https://www.xilinx.com/products/boards-and-kits.html) in order to implement the accelerator through C, eliminating the need to write any code in HDL Languages such as Verilog/VHDL. To reduce any performance losses, I implement a systolic array based architecture and utilize techniques such as pipelining, loop unrolling, and memory partitioning. Through this project, I aim to highlight the feasibility and viability of FPGAs for low latency, highly energy efficient machine learning workflows, possibly placing them in consideration as a replacement for GPUs for infrence based tasks.
### What is an FPGA?
A Field Programmable Gate Array, or FPGA, is a type of integrated circuit that is made up of a massive collection of unconnected digital logic parts. When someone designs *gateware* for an FPGA, they are essentially connecting these logic blocks together in a way that creates a new piece of hardware. FPGAs are also "field programmable," meaning that they can be reconfigured on-the-fly as per the designer's needs. While often used as tools for rapidly prototyping hardware designs, the nature of an FPGA's highly specialized and customizable hardware design allows them to achieve very low latency, high throughput, and be very energy efficient.
#### What is High Level Synthesis (HLS)?
High Level Synthesis is a method of designing gateware that allows a programmer to write the gateware in a higher level language like C, C++, or even [Python](https://fastmachinelearning.org/hls4ml/). A High Level Synthesis tool takes this description of the intended function of the hardware from a higher level language and then synthesizes it into RTL level code (such as Verilog or VHDL). Since writing in languages such as Verilog can be tedious and time consuming, HLS serves as an alternative for designers who want to efficiently build and verify hardware in a language that is much easier to write, and is also a tool that invites normal programmers with no experience writing HDL languages to start developing hardware. In this project, I chose to use HLS to not only work under my time constraint, but evaluate how well the HLS workflow truly is to an invidual with little to no experience in HDL languages.
### Reflection and Next Steps
This project was an amazing way to get involved with both the FPGA and hardware design/accelerator design space. I was able to gain a lot of hands on experience with the design workflow for developing gateware for an FPGA, and also was able to gain insights on performance optimization concepts and methods such as systolic arrays, loop pipelining/unrolling, and code inlining. Furthermore, I was able to work more with the mathematics and theory behind Deep Learning and Neural Networks, which is very good knowledge to have given the development of artifical intellegence. The next steps of this project include cleaning up and optimizing the code, possibly implementing quantization, batch normalization, and other types of layerz such as residual blocks to further improve performance for the neural network. On the hardware side, next steps include obtaining a physical FPGA development board to actually deploy this program onto, and possibly performing a rewrite of the code to not rely on HLS, but write the neural network from scratch in an HDL language such as Verilog.
