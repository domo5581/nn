#include "ap_fixed.h"
#include "hls_stream.h"
#include "hls_math.h"
#include <string.h>

// Fixed point definitions for better hardware efficiency
typedef ap_fixed<16,8> data_t;  // 16 bits total, 8 integer bits
typedef ap_fixed<16,8> weight_t;
typedef ap_fixed<32,16> acc_t;  // Wider accumulator to prevent overflow

// Enums remain the same
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

// Maximum size definitions for static arrays
#define MAX_LAYER_SIZE 1024
#define MAX_FILTER_SIZE 11
#define MAX_CHANNELS 256
#define MAX_FILTERS 256

// Layer struct optimized for HLS
struct Layer {
	ltype type;
	int height;
	int width;
	int channels;

	union {
		struct {
			int num_filters;
			int filter_size;
			int stride;
			int zero_padding;
			int input_height;
			int input_width;
			int input_channels;
			weight_t weights[MAX_FILTERS][MAX_CHANNELS][MAX_FILTER_SIZE][MAX_FILTER_SIZE];
			data_t biases[MAX_FILTERS];
		} conv_params;

		struct {
			int pool_size;
			int stride;
			int input_height;
			int input_width;
		} pool_params;

		struct {
			int output_size;
			weight_t weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE];
			data_t biases[MAX_LAYER_SIZE];
			activation type;
		} fc_params;
	} params;

	data_t output[MAX_LAYER_SIZE];
	data_t delta[MAX_LAYER_SIZE];
	data_t pre_activation[MAX_LAYER_SIZE];
};

// Helper functions
data_t sigmoid(data_t x) {
	#pragma HLS INLINE
	return 1.0 / (1.0 + hls::exp(-x));
}

data_t relu(data_t x) {
	#pragma HLS INLINE
	return (x > 0) ? x : 0;
}

// Systolic array matrix multiplication for fully connected layers
void systolic_matrix_multiply(
	const weight_t weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE],
	const data_t input[MAX_LAYER_SIZE],
	acc_t output[MAX_LAYER_SIZE],
	int M, int N) {

	#pragma HLS PIPELINE II=1
	#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=16 dim=2
	#pragma HLS ARRAY_PARTITION variable=input cyclic factor=16

	static acc_t pe_array[MAX_LAYER_SIZE];
	#pragma HLS ARRAY_PARTITION variable=pe_array cyclic factor=16

	// Initialize processing elements
	for (int i = 0; i < M; i++) {
		#pragma HLS UNROLL factor=16
		pe_array[i] = 0;
	}

	// Systolic computation
	for (int k = 0; k < N; k++) {
		for (int i = 0; i < M; i++) {
			#pragma HLS PIPELINE II=1
			#pragma HLS UNROLL factor=16
			pe_array[i] += weights[i][k] * input[k];
		}
	}

	// Write results
	for (int i = 0; i < M; i++) {
		#pragma HLS UNROLL factor=16
		output[i] = pe_array[i];
	}
}

// Optimized convolution forward pass
void conv_forward(Layer& layer, const data_t input[MAX_LAYER_SIZE]) {
	#pragma HLS INLINE off

	const int padding = layer.params.conv_params.zero_padding;
	const int stride = layer.params.conv_params.stride;
	const int filter_size = layer.params.conv_params.filter_size;
	const int num_filters = layer.params.conv_params.num_filters;
	const int input_height = layer.params.conv_params.input_height;
	const int input_width = layer.params.conv_params.input_width;
	const int input_channels = layer.params.conv_params.input_channels;

	// Create padded input buffer
	data_t padded_input[MAX_CHANNELS][MAX_FILTER_SIZE][MAX_FILTER_SIZE];
	#pragma HLS ARRAY_PARTITION variable=padded_input complete dim=1

	const int padded_height = input_height + 2 * padding;
	const int padded_width = input_width + 2 * padding;
	const int output_height = (padded_height - filter_size) / stride + 1;
	const int output_width = (padded_width - filter_size) / stride + 1;

	// Main convolution loops
CONV_FILTERS: for(int f = 0; f < num_filters; f++) {
	CONV_OUTPUT_H: for(int oh = 0; oh < output_height; oh++) {
		CONV_OUTPUT_W: for(int ow = 0; ow < output_width; ow++) {
				#pragma HLS PIPELINE II=1

				acc_t sum = 0;

			CONV_CHANNELS: for(int c = 0; c < input_channels; c++) {
				CONV_KERNEL_H: for(int fh = 0; fh < filter_size; fh++) {
					CONV_KERNEL_W: for(int fw = 0; fw < filter_size; fw++) {
							#pragma HLS UNROLL factor=3

							int ih = oh * stride + fh;
							int iw = ow * stride + fw;

							if (ih >= 0 && ih < padded_height && iw >= 0 && iw < padded_width) {
								sum += input[c * input_height * input_width + (ih-padding) * input_width + (iw-padding)] * 
									layer.params.conv_params.weights[f][c][fh][fw];
							}
						}
					}
				}

				sum += layer.params.conv_params.biases[f];
				int output_idx = f * output_height * output_width + oh * output_width + ow;
				layer.pre_activation[output_idx] = sum;
				layer.output[output_idx] = relu(sum);
			}
		}
	}
}

// Optimized max pooling forward pass
void maxpool_forward(Layer& layer, const data_t input[MAX_LAYER_SIZE]) {
	#pragma HLS INLINE off

	const int pool_size = layer.params.pool_params.pool_size;
	const int stride = layer.params.pool_params.stride;
	const int input_height = layer.height;
	const int input_width = layer.width;
	const int input_channels = layer.channels;

	const int output_height = (input_height - pool_size) / stride + 1;
	const int output_width = (input_width - pool_size) / stride + 1;

POOL_CHANNELS: for(int c = 0; c < input_channels; c++) {
	POOL_OUTPUT_H: for(int oh = 0; oh < output_height; oh++) {
		POOL_OUTPUT_W: for(int ow = 0; ow < output_width; ow++) {
				#pragma HLS PIPELINE II=1

				data_t max_val = -INFINITY;

			POOL_WINDOW_H: for(int ph = 0; ph < pool_size; ph++) {
				POOL_WINDOW_W: for(int pw = 0; pw < pool_size; pw++) {
						#pragma HLS UNROLL

						int ih = oh * stride + ph;
						int iw = ow * stride + pw;
						data_t val = input[c * input_height * input_width + ih * input_width + iw];
						max_val = (val > max_val) ? val : max_val;
					}
				}

				layer.output[c * output_height * output_width + oh * output_width + ow] = max_val;
			}
		}
	}
}

// Optimized fully connected forward pass using systolic array
void fc_forward(Layer& layer, const data_t input[MAX_LAYER_SIZE]) {
	#pragma HLS INLINE off

	const int output_size = layer.params.fc_params.output_size;
	const int input_size = layer.height * layer.width * layer.channels;

	// Use systolic array for matrix multiplication
	acc_t temp_output[MAX_LAYER_SIZE];
	systolic_matrix_multiply(layer.params.fc_params.weights, input, temp_output, output_size, input_size);

	// Add biases and apply activation
FC_OUTPUT: for(int o = 0; o < output_size; o++) {
		#pragma HLS PIPELINE II=1

		acc_t sum = temp_output[o] + layer.params.fc_params.biases[o];

		if(layer.params.fc_params.type == a_sigmoid) {
			layer.pre_activation[o] = sum;
			layer.output[o] = sigmoid(sum);
		} else {
			layer.output[o] = sum; // For softmax, store raw values
		}
	}

	// Apply softmax if needed
	if(layer.params.fc_params.type == a_softmax) {
		acc_t max_val = layer.output[0];
		acc_t sum = 0;

		// Find max value for numerical stability
	SOFTMAX_MAX: for(int i = 1; i < output_size; i++) {
			#pragma HLS PIPELINE II=1
			max_val = (layer.output[i] > max_val) ? layer.output[i] : max_val;
		}

		// Compute exponentials and sum
	SOFTMAX_EXP: for(int i = 0; i < output_size; i++) {
			#pragma HLS PIPELINE II=1
			layer.output[i] = hls::exp(layer.output[i] - max_val);
			sum += layer.output[i];
		}

		// Normalize
	SOFTMAX_NORM: for(int i = 0; i < output_size; i++) {
			#pragma HLS PIPELINE II=1
			layer.output[i] /= sum;
		}
	}
}

// Top-level function for HLS synthesis
void cnn_forward(
	data_t input[MAX_LAYER_SIZE],
	data_t output[MAX_LAYER_SIZE],
	Layer layers[],
	int num_layers) {

	#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=layers offset=slave bundle=gmem2
	#pragma HLS INTERFACE s_axilite port=num_layers bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	data_t layer_input[MAX_LAYER_SIZE];
	data_t layer_output[MAX_LAYER_SIZE];

	// Copy input to local buffer
	memcpy(layer_input, input, MAX_LAYER_SIZE * sizeof(data_t));

	// Process each layer
LAYER_LOOP: for(int i = 0; i < num_layers; i++) {
		#pragma HLS LOOP_TRIPCOUNT min=1 max=20

		Layer& current_layer = layers[i];

		switch(current_layer.type) {
			case conv:
				conv_forward(current_layer, layer_input);
				break;
			case max_pool:
				maxpool_forward(current_layer, layer_input);
				break;
			case fully_connected:
				fc_forward(current_layer, layer_input);
				break;
			default:
				break;
		}

		// Copy output to input buffer for next layer
		memcpy(layer_input, current_layer.output, MAX_LAYER_SIZE * sizeof(data_t));
	}

	// Copy final output
	memcpy(output, layer_input, MAX_LAYER_SIZE * sizeof(data_t));
}
