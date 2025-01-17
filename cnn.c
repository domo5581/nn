// convolutional neural network c header library
// inspired by euske's nn1
// meant to be synthesized into RTL through Vitus HLS for an FPGA implementation

#include <stdlib.h>
#include <math.h>
#include <string.h>

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
	a_sigmoid,
	a_softmax,
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
			int input_height;
			int input_width;
			int input_channels;
			float (*weights);
			float (*biases);
		} conv_params;

		struct {
			int pool_size; // single integer again
			int stride;
			int input_height;
			int input_width;
		} pool_params;

		struct {
			int output_size;
			float* weights;
			float* biases;
			activation type;
		} fc_params;
	} params;
	float* output;
	float* delta;
	float* pre_activation;
	float (*activation_g)(float);
} Layer;

typedef struct {
	Layer** layers;
	int num_layers;
} Network;

Network* create_network(int capacity) {
	Network* network = (Network*)malloc(sizeof(Network));
	network->layers = (Layer**)malloc(capacity * sizeof(Layer*));
	network->num_layers = capacity;
	return network;
}

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

float relu(float x) {
	return x > 0 ? x : 0;
}

float sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

float relu_g(float x) {
	return x > 0 ? 1 : 0;
}

float sigmoid_g(float x) {
	float sig = sigmoid(x);
	return sig * (1 - sig);
}

void softmax(float* input, float* output, int size) {
	float max = input[0];
	for(int i = 1; i < size; i++) {
		if(input[i] > max) {
			max = input[i];
		}
	}
	float sum = 0;
	for(int i = 0; i < size; i++) {
		output[i] = exp(input[i] - max);
		sum += output[i];
	}
	for(int i = 0; i < size; i++) {
		output[i] /= sum;
	}
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
	layer->params.conv_params.input_height = input_height;
	layer->params.conv_params.input_width = input_width;
	layer->params.conv_params.input_channels = input_channels;

	// output dimensions
	// https://cs231n.github.io/convolutional-networks/
	int output_h = (input_height + 2 * padding - filter_size) / stride + 1;
	int output_w = (input_width + 2 * padding - filter_size) / stride + 1;
	layer->height = output_h;
	layer->width = output_w;
	layer->channels = num_filters;
	layer->activation_g = relu_g;

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
	layer->pre_activation = (float*)calloc(output_h * output_w * num_filters, sizeof(float));

	return layer;
}

Layer* create_maxpool(int input_height, int input_width, int input_channels, int pool_size, int stride) {
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	layer->type = max_pool;
	layer->params.pool_params.pool_size = pool_size;
	layer->params.pool_params.stride = stride;
	layer->params.pool_params.input_height = input_height;
	layer->params.pool_params.input_width = input_width;


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
	layer->activation_g = (type == a_sigmoid) ? sigmoid_g : NULL; // null is softmax (doesnt have a gradient)

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
	layer->pre_activation = (float*) calloc(output_size, sizeof(float));

	return layer;
}

void free_layer(Layer* layer) {
	switch (layer->type) {
		case input:
			free(layer->output);
			free(layer);
			break;
		case conv:
			free(layer->params.conv_params.weights);
			free(layer->params.conv_params.biases);
			free(layer->output);
			free(layer->delta);
			free(layer->pre_activation);
			free(layer);
			break;
		case max_pool:
			free(layer->output);
			free(layer->delta);
			free(layer);
			break;
		case fully_connected:
			free(layer->params.fc_params.weights);
			free(layer->params.fc_params.biases);
			free(layer->output);
			free(layer->delta);
			free(layer->pre_activation);
			free(layer);
			break;
	}
}

void destroy_network(Network* network) {
	if (!network) return;
	for (int i = 0; i < network->num_layers; i++) {
		if (network->layers[i]) {
			free_layer(network->layers[i]);
		}
	}
	free(network->layers);
	free(network);
}

void conv_forward(Layer* layer, float* input) {
	int padding = layer->params.conv_params.zero_padding;
	int stride = layer->params.conv_params.stride;
	int filter_size = layer->params.conv_params.filter_size;
	int num_filters =  layer->params.conv_params.num_filters;
	int input_height = layer->params.conv_params.input_height; 
	int input_width = layer->params.conv_params.input_width;   
	int input_channels = layer->params.conv_params.input_channels;  	

	int padded_height = input_height + 2 * padding;
	int padded_width = input_width + 2 * padding;
	float* padded_input = (float*) calloc(padded_height * padded_width * input_channels, sizeof(float));

	for (int c = 0; c < input_channels; c++) {
		for (int h = 0; h < input_height; h++) {
			for (int w = 0; w < input_width; w++) {
				padded_input[c * padded_height * padded_width + (h + padding) * padded_width + (w + padding)] = input[c * input_height * input_width + h * input_width + w];
			}
		}
	}

	int output_height = (padded_height - filter_size) / stride + 1;
	int output_width = (padded_width - filter_size) / stride + 1;
	int output_size = output_height * output_width * num_filters;

	// for every filter
	for(int f = 0; f < num_filters; f++) {
		for(int oh = 0; oh < output_height; oh++) {
			for(int ow = 0; ow < output_width; ow++) {
				float sum = 0;
				for(int c = 0; c < input_channels; c++) {
					for(int fh = 0; fh < filter_size; fh++) {
						for(int fw = 0; fw < filter_size; fw++) {
							int ih = oh * stride + fh;
							int iw = ow * stride + fw;


							if (ih >= 0 && ih < padded_height && iw >= 0 && iw < padded_width) {
								int input_idx = c * padded_height * padded_width + ih * padded_width + iw;
								int weight_idx = f * input_channels * filter_size * filter_size + 
									c * filter_size * filter_size + 
									fh * filter_size + fw;

								sum += padded_input[input_idx] * layer->params.conv_params.weights[weight_idx];
							}
						}
					}
				}
				sum += layer->params.conv_params.biases[f];
				int output_idx = f * output_height * output_width + oh * output_width + ow;
				layer->pre_activation[output_idx] = sum;
				layer->output[output_idx] = relu(sum);
			}
		}
	}	

	free(padded_input);
}

void maxpool_forward(Layer* layer, float* input) {
	int pool_size = layer->params.pool_params.pool_size;
	int stride = layer->params.pool_params.stride;
	// prev layer
	int input_height = layer->height; 
	int input_width = layer->width; 
	int input_channels = layer->channels; 

	int output_height = (input_height - pool_size) / stride + 1;
	int output_width = (input_width - pool_size) / stride + 1;
	int output_size = output_height * output_width * input_channels;

	for(int c = 0; c < input_channels; c++) {
		for(int oh = 0; oh < output_height; oh++) {
			for(int ow = 0; ow < output_width; ow++) {
				float max_val = -INFINITY;
				for(int ph = 0; ph < pool_size; ph++) {
					for(int pw = 0; pw < pool_size; pw++) {
						int ih = oh * stride + ph;
						int iw = ow * stride + pw;
						float val = input[c * input_height * input_width + ih * input_width + iw];
						if(val > max_val) {
							max_val = val;
						}
					}
				}
				layer->output[c * output_height * output_width + oh * output_width + ow] = max_val;
			}
		}
	}
}

void fc_forward(Layer* layer, float* input) {
	int output_size = layer->params.fc_params.output_size;
	int input_size = layer->height * layer->width * layer->channels;

	// flatten
	float* flattened_input = (float*) calloc(input_size, sizeof(float));
	for(int i = 0; i < input_size; i++) {
		flattened_input[i] = input[i];
	}

	// matmul (output = bias + (input * weight))
	float* temp_output = (float*) calloc(output_size, sizeof(float));
	for(int o = 0; o < output_size; o++) {
		float sum = 0;
		for(int i = 0; i < input_size; i++) {
			sum += flattened_input[i] * layer->params.fc_params.weights[o * input_size + i];
		}
		sum += layer->params.fc_params.biases[o];
		temp_output[o] = sum;
	}

	// apply the correct activation (sigmoid for non output layers, softmax for output)
	if(layer->params.fc_params.type == a_sigmoid) {
		for(int o = 0; o < output_size; o++) {
			layer->pre_activation[o] = temp_output[o];
			layer->output[o] = sigmoid(temp_output[o]);
		}
	} else if(layer->params.fc_params.type == a_softmax) {
		softmax(temp_output, layer->output, output_size);
	}

	free(temp_output);
	free(flattened_input);
}

void forward_propagation(Layer* layer, float* input_fc) {
	int input_size;
	switch(layer->type) {
		case input:
			// input to layer->output
			input_size = (layer->height * layer->width * layer->channels);
			for(int i = 0; i < input_size; i++) {
				layer->output[i] = input_fc[i];
			}
			break;
		case conv:
			conv_forward(layer, input_fc);
			break;
		case max_pool:
			maxpool_forward(layer, input_fc);
			break;
		case fully_connected:
			fc_forward(layer, input_fc);
			break;
	}
}

void network_forward(Network* network, float* input) {
	float* current_input = input;
	for (int i = 0; i < network->num_layers; i++) {
		forward_propagation(network->layers[i], current_input);
		current_input = network->layers[i]->output;
	}
}

void fc_backward(Layer* layer, float* prev_delta, float* input, float learning_rate) {
	int output_size = layer->params.fc_params.output_size;
	int input_size = layer->height * layer->width * layer->channels;

	float* gradient;
	if(layer->params.fc_params.type == a_softmax) {
		gradient = (float*)malloc(output_size * sizeof(float));
		for(int i = 0; i < output_size; i++) {
			gradient[i] = layer->output[i];
			if(prev_delta[i] > 0.5) { // one hot encoded
				gradient[i] -= 1.0;
			}
		}
	} else {
		gradient = prev_delta;
	}

	// update weights and biases
	for(int o = 0; o < output_size; o++) {
		for(int i = 0; i < input_size; i++) {
			layer->params.fc_params.weights[o * input_size + i] -= 
				learning_rate * gradient[o] * input[i];
		}
		layer->params.fc_params.biases[o] -= learning_rate * gradient[o];
	}

	// gradient 
	if(layer->activation_g) { 
		for(int i = 0; i < input_size; i++) {
			float sum = 0;
			for(int o = 0; o < output_size; o++) {
				sum += layer->params.fc_params.weights[o * input_size + i] * gradient[o];
			}
			layer->delta[i] = sum * layer->activation_g(layer->pre_activation[i]);
		}
	}

	if(layer->params.fc_params.type == a_softmax) {
		free(gradient);
	}
}



void conv_backward(Layer* layer, float* prev_delta, float* input, float learning_rate) {
	int num_filters = layer->params.conv_params.num_filters;
	int channels = layer->channels;
	int filter_size = layer->params.conv_params.filter_size;
	int input_height = layer->height;
	int input_width = layer->width;
	int padding = layer->params.conv_params.zero_padding;
	int stride = layer->params.conv_params.stride;
	int output_height = (input_height + 2 * padding - filter_size) / stride + 1;
	int output_width = (input_width + 2 * padding - filter_size) / stride + 1;

	// gradient w/respect to filters
	for(int f = 0; f < num_filters; f++) {
		for(int c = 0; c < channels; c++) {
			for(int fh = 0; fh < filter_size; fh++) {
				for(int fw = 0; fw < filter_size; fw++) {
					float grad = 0;
					for(int oh = 0; oh < output_height; oh++) {
						for(int ow = 0; ow < output_width; ow++) {
							int ih = oh * stride + fh - padding;
							int iw = ow * stride + fw - padding;
							if(ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
								grad += input[c * input_height * input_width + ih * input_width + iw] * prev_delta[f * output_height * output_width + oh * output_width + ow];
							}
						}
					}
					int index = f * channels * filter_size * filter_size + c * filter_size * filter_size + fh * filter_size + fw;
					layer->params.conv_params.weights[index] -= learning_rate * grad;
				}
			}
		}
	}

	// gradient w/respect to biases
	for(int f = 0; f < num_filters; f++) {
		float grad = 0;
		for(int oh = 0; oh < output_height; oh++) {
			for(int ow = 0; ow < output_width; ow++) {
				grad += prev_delta[f * output_height * output_width + oh * output_width + ow];
			}
		}
		layer->params.conv_params.biases[f] -= learning_rate * grad;
	}

	// gradient with respect to inputs
	for(int c = 0; c < channels; c++) {
		for(int ih = 0; ih < input_height; ih++) {
			for(int iw = 0; iw < input_width; iw++) {
				float grad = 0;
				for(int f = 0; f < num_filters; f++) {
					for(int fh = 0; fh < filter_size; fh++) {
						for(int fw = 0; fw < filter_size; fw++) {
							int oh = (ih - fh + padding) / stride;
							int ow = (iw - fw + padding) / stride;
							if((ih - fh + padding) % stride == 0 && (iw - fw + padding) % stride == 0 && oh < output_height && ow < output_width) {
								int w_index = f * channels * filter_size * filter_size + c * filter_size * filter_size + fh * filter_size + fw;
								grad += layer->params.conv_params.weights[w_index] * prev_delta[f * output_height * output_width + oh * output_width + ow];
							}
						}
					}
				}
				layer->delta[c * input_height * input_width + ih * input_width + iw] = grad * layer->activation_g(layer->pre_activation[c * input_height * input_width + ih * input_width + iw]);
			}
		}
	}
}

void maxpool_backward(Layer* layer, float* prev_delta, float* input, float learning_rate) {
	int pool_size = layer->params.pool_params.pool_size;
	int stride = layer->params.pool_params.stride;
	int input_height = layer->params.pool_params.input_height;
	int input_width = layer->params.pool_params.input_width;
	int channels = layer->channels;

	// Zero initialize deltas
	memset(layer->delta, 0, input_height * input_width * channels * sizeof(float));

	int output_height = layer->height;
	int output_width = layer->width;

	for(int c = 0; c < channels; c++) {
		for(int oh = 0; oh < output_height; oh++) {
			for(int ow = 0; ow < output_width; ow++) {
				// finds max value
				int maxI = -1, maxJ = -1;
				float maxVal = -INFINITY;

				for(int ph = 0; ph < pool_size; ph++) {
					for(int pw = 0; pw < pool_size; pw++) {
						int ih = oh * stride + ph;
						int iw = ow * stride + pw;

						// checks bounds
						if (ih < input_height && iw < input_width) {
							float val = input[c * input_height * input_width + ih * input_width + iw];
							if(val > maxVal) {
								maxVal = val;
								maxI = ih;
								maxJ = iw;
							}
						}
					}
				}

				// only propagate gradient if a valid max position is found
				if(maxI != -1 && maxJ != -1) {
					int delta_idx = c * output_height * output_width + oh * output_width + ow;
					layer->delta[c * input_height * input_width + maxI * input_width + maxJ] = 
						prev_delta[delta_idx];
				}
			}
		}
	}
}

void backward_propagation(Layer* layer, float* prev_delta, float* input_fc, float learning_rate) {
	switch(layer->type) {
		case fully_connected:
			fc_backward(layer, prev_delta, input_fc, learning_rate);
			break;
		case conv:
			conv_backward(layer, prev_delta, input_fc, learning_rate);
			break;
		case max_pool:
			maxpool_backward(layer, prev_delta, input_fc, learning_rate);
			break;
		case input:
			// No backpropagation for input layer
			break;
	}
}

void network_backward(Network* network, float* label, float learning_rate) {
	// ouput
	Layer* output_layer = network->layers[network->num_layers - 1];
	// output gradient
	for(int o = 0; o < output_layer->channels; o++) {
		output_layer->delta[o] = output_layer->output[o] - label[o];
	}
	// backprop
	for(int i = network->num_layers - 2; i >= 0; i--) {
		Layer* current_layer = network->layers[i];
		Layer* next_layer = network->layers[i + 1];
		backward_propagation(current_layer, next_layer->delta, current_layer->output, learning_rate);
	}
}
