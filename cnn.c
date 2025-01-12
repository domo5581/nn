// convolutional neural network c header library
// inspired by euske's nn1
// meant to be synthesized into RTL through Vitus HLS for an FPGA implementation

#include <stdlib.h>
#include <math.h>

typedef enum {
	input,
	conv,
	max_pool,
	fully_connected
} ltype;

typedef enum {
	fc_input,
	fc_hidden,
	fc_output,
} fcpos;

typedef enum {
	sigmoid,
	relu,
	softmax,
} activation;

typedef struct {
	ltype type;
	int height;
	int width;
	int channels; // in this case, "channels" are the number of filters that are coming in
	
	union {
		struct {
			int num_filters;
			int filter_size; // single integer b/c filter will usually be square shaped
			int stride;
			int zero_padding; // single integer for how many layers of zero padding
			float (*weights);
			float (*biases);
		} conv_params;

		struct {
			int pool_size; // single integer again
			int stride;
		} pool_params;

		struct {
			int output_size;
			float (*weights);
			float (*biases);
			activation type;
		} fc_params;
	} params;
	float *output;
	float *delta;
} Layer;

float he_init(int fan_in) {
	float scale = sqrt(2.0f / fan_in);
	float random = (float)rand() / RAND_MAX * 2 - 1;
	return random * scale;
}

float glorot_init(int fan_in, int fan_out) {
	float limit = sqrt(6.0f / (fan_in + fan_out));
	float random = (float)rand() / RAND_MAX;
	return random * 2 * limit - limit;
}


Layer* create_input(int height, int width, int channels) {
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	layer->type = input;
	layer->height = height;
	layer->width = width;
	layer->channels = channels;
	layer->output = (float*)calloc(height * width * channels, sizeof(float));
	return layer;
}

Layer* create_conv(int input_height, int input_width, int input_channels, int num_filters, int filter_size, int stride, int padding) {
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	layer->type = conv;
	layer->params.conv_params.num_filters = num_filters;
	layer->params.conv_params.filter_size = filter_size;
	layer->params.conv_params.stride = stride;
	layer->params.conv_params.zero_padding = padding;

	// output dimensions
	// https://cs231n.github.io/convolutional-networks/
	int output_h = (input_height + 2 * padding - filter_size) / stride + 1;
	int output_w = (input_width + 2 * padding - filter_size) / stride + 1;
	layer->height = output_h;
	layer->width = output_w;
	layer->channels = num_filters;

	// conv layer uses relu, use HE init
	int weights_size = num_filters * input_channels * filter_size * filter_size;
	int fan_in = input_channels * filter_size * filter_size;
	layer->params.conv_params.weights = (float*)calloc(weights_size, sizeof(float));
	for (int i = 0; i < weights_size; i++) {
		layer->params.conv_params.weights[i] = he_init(fan_in);
	}

	layer->params.conv_params.biases = (float*)calloc(num_filters, sizeof(float));

	layer->output = (float*) calloc(output_h * output_w * num_filters, sizeof(float));
  layer->delta = (float*) calloc(output_h * output_w * num_filters, sizeof(float));

	return layer;
}

Layer* create_maxpool(int input_height, int input_width, int input_channels, int pool_size, int stride) {
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	layer->type = max_pool;
	layer->params.pool_params.pool_size = pool_size;
	layer->params.pool_params.stride = stride;

	// output dimensions
	// https://cs231n.github.io/convolutional-networks/
	int output_h = (input_height - pool_size) / stride + 1;
	int output_w = (input_width - pool_size) / stride + 1;
	layer->height = output_h;
	layer->width = output_w;
	layer->channels = input_channels;

	layer->output = (float*) calloc(output_h * output_w * input_channels, sizeof(float));
  layer->delta = (float*) calloc(output_h * output_w * input_channels, sizeof(float));

	return layer;
}

Layer* create_fc(int output_size, int input_size, activation type) {
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	layer->type = fully_connected;
	layer->params.fc_params.output_size = output_size;
	layer->params.fc_params.type = type; // activation type can either be sigmoid or softmax (output layer)
	
	// use glorot initalization 
	layer->params.fc_params.weights = (float*)calloc(output_size * input_size, sizeof(float));
	for (int i = 0; i < (output_size * input_size); i++) {
		layer->params.fc_params.weights[i] = glorot_init(input_size, output_size);
	}

	layer->params.fc_params.biases = (float*)calloc(output_size, sizeof(float));

	layer->height = 1;
	layer->width = 1;
	layer->channels = output_size;
	layer->output = (float*) calloc(output_size, sizeof(float));
	layer->delta = (float*) calloc(output_size, sizeof(float));

	return layer;
}

void free_layer(Layer* layer) {
	switch (layer->type) {
		case input:
			free(layer->output);
    	free(layer);
		case conv:
			free(layer->params.conv_params.weights);
    	free(layer->params.conv_params.biases);
    	free(layer->output);
    	free(layer->delta);
    	free(layer);
		case max_pool:
			free(layer->output);
    	free(layer->delta);
    	free(layer);
		case fully_connected:
			free(layer->params.fc_params.weights);
    	free(layer->params.fc_params.biases);
    	free(layer->output);
    	free(layer->delta);
    	free(layer);
	}
}
