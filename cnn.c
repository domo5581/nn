#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef enum {
  input,
  conv,
  max_pool,
  fully_connected,
  output
} ltype;

typedef enum {
  relu,
  softmax,
  sigmoid,
  tanh
} activation;

typedef struct {
  int filters;
  int filter_h;
  int filter_w;
  int stride;
  int zeropadding; // amount of zeropadding (1 = one layer... etc.)
} convparams;

typedef struct {
  int pool_height; // height and width of the pooling window
  int pool_width;
} poolparams;

typedef struct {
  ltype type;
  activation atype;
  
  int input_height;
  int input_width;
  int input_channels;

  int output_height;
  int output_width;
  int output_channels;

  union {
    convparams layerconv;
    poolparams layerpool;
  } params;  

  float* weights;
  float* biases;
} Layer;

Layer* createlayer(ltype type, int height, int width, int channels, void* params) {
  Layer* layer = (Layer*)malloc(sizeof(Layer));
  layer->type = type;
  layer->input_height = height;
  layer->input_width = width;
  layer->input_channels = channels;

  layer->weights = NULL;
  layer->biases = NULL;

  switch(type) {
    case input: {
      layer->output_height = input_height;
      layer->output_width = input_width;
      layer->output_channels = input_channels;
      layer->activation = relu;
      break;
    }
    case conv: {
      convparams* cparams = (convparams*)params;
      layer->params.layerconv = *cparams;
      layer->activation = relu;

      // https://cs231n.github.io/convolutional-networks/#pool - formula to find dimensions
      layer->output_height = ((input_height + 2*conv_params->zero_padding - conv_params->filter_height) / conv_params->stride_height) + 1;
      layer->output_width = ((input_width + 2*conv_params->zero_padding - conv_params->filter_width) / conv_params->stride_width) + 1;

      layer->output_channels = convparams->filters;
      

    }
}
