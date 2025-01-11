// convolutional neural network c header library
// inspired by euske's nn1
// meant to be synthesized into RTL through Vitus HLS for an FPGA implementation

#include <cstdlib>
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

typedef struct {
	ltype type;
	// spatial extent of layer- l,w,depth (color space)
	int height;
	int width;
	int channels;
	
	// layer params
	union {
		struct {
			int num_filters;
			int filter_height;
			int filter_width;
			int stride;
			int zero_padding; // how many layers of zero padding
			float*** filters; // (width x height) x filters
		} conv_params;

		struct {
			int pool_height;
			int pool_width;
			int stride;
		} pool_params;

		struct {
			int input_neurons;
			int output_neurons;
			float** weights;
			float* biases;
			fcpos position;
		} fc_params;
	} params;
} Layer;

float random_uniform(float min, float max) {
	return min + (max - min) * ((float)rand() / RAND_MAX);
}

float he_uniform(int fan_in) {
	float limit = sqrt(6.0f / fan_in);
	return random_uniform((limit * -1), limit);
}

float glorot_uniform(int fan_in, int fan_out) {
	float limit = sqrt(6.0f / (fan_in + fan_out));
  return random_uniform((limit * -1), limit);
}


Layer* create_input(int height, int width, int channels) {
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	layer->type = input;
	layer->height = height;
	layer->width = width;
	layer->channels = channels;
	return layer;
}

	Layer* create_conv(int height, int width, int channels, int num_filters, int filter_width, int filter_height, int stride, int zero_padding) {
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	layer->type = conv;
	layer->height = height;
	layer->width = width;
	layer->channels = channels;

	layer->params.conv_params.num_filters = num_filters;
	layer->params.conv_params.filter_height = filter_height;
	layer->params.conv_params.filter_width = filter_width;
	layer->params.conv_params.stride = stride;
	layer->params.conv_params.zero_padding = zero_padding;

	// conv layer uses relu - use he init for weights
	layer->params.conv_params.filters = (float***)malloc(num_filters * sizeof(float**));
	int fan_in = filter_height * filter_width * channels;
	for (int f = 0; f < num_filters; f++) {
		layer->params.conv_params.filters[f] = (float**)malloc(filter_height * sizeof(float*));
		for (int h = 0; h < filter_height; h++) {
			layer->params.conv_params.filters[f][h] = (float*)malloc(filter_width * sizeof(float));
			for (int w = 0; w < filter_width; w++) {
				layer->params.conv_params.filters[f][h][w] = he_uniform(fan_in);
			}
		}
	}

	return layer;
}

Layer* create_max_pool(int height, int width, int channels, int pool_height, int pool_width, int stride) {
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	layer->type = max_pool;
	layer->height = height;
	layer->width = width;
	layer->channels = channels;

	layer->params.pool_params.pool_height = pool_height;
	layer->params.pool_params.pool_width = pool_width;
	layer->params.pool_params.stride = stride;

	return layer;
}

Layer* create_fc(int input_neurons, int output_neurons, fcpos position) {
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	layer->type = fully_connected;
	layer->height = 1;
	layer->width = output_neurons;
	layer->channels = 1;

	layer->params.fc_params.input_neurons = input_neurons;
	layer->params.fc_params.output_neurons = output_neurons;
  layer->params.fc_params.position = position;

	// use xav/glorot init b/c of sigmoid
	layer->params.fc_params.weights = (float**)malloc(output_neurons * sizeof(float*));
	for (int i = 0; i < output_neurons; i++) {
		layer->params.fc_params.weights[i] = (float*)malloc(input_neurons * sizeof(float));
		for (int j = 0; j < input_neurons; j++) {
			layer->params.fc_params.weights[i][j] = glorot_uniform(input_neurons, output_neurons);
		}
	}

	return layer;
}

