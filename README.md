# nn - implementation of neural networks in c

implements neural networks in c, targets embedded systems (microcontrollers, fpgas) 

#### current implementations
`snn.c` - a simple feedforward neural network written in ~150loc. \
`cnn.c` - TODO, implements a convolutional neural network \
`cnn-hls.c` - TODO, has fpga hls specific types/pragmas in order to synthesize to verilog; run on an fpga \

depends on native c libraries and [gsl](https://www.gnu.org/software/gsl/)

### future goals
cnn w/ pragmas -> successfully compiled to verilog using vivado/vitus \
self-made matrix multiplication library, relying only on native c ones \
code cleanup and optimization
