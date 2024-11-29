# nn - implementation of neural networks in c

implements neural networks in c, targets embedded systems (microcontrollers, fpgas) \
current implemented a simple feedforward nn in `snn.c`, todo is a convolutional neural network (with HLS #pragmas) for an fpga. \

depends on native c libraries and [gsl](https://www.gnu.org/software/gsl/) as the only external one

### future goals
cnn w/ pragams -> successfully compiled to verilog using vivado/vitus \
self-made matrix multiplication library, relying only on native c ones \
code cleanup and optimization \
