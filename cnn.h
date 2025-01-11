// convolutional neural network c header library
// inspired by euske's nn1
// meant to be synthesized into RTL through vitus HLS for an FPGA implementation

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

typedef struct Layer {
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
