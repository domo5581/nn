#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cnn.c"

#define IMG_HEIGHT 28
#define IMG_WIDTH 28
#define NUM_CLASSES 10
#define BATCH_SIZE 32
#define LEARNING_RATE 0.01
#define NUM_EPOCHS 10

float* read_mnist_images(const char* filename, int* num_images) {
	FILE* fp = fopen(filename, "rb");
	if (!fp) {
		printf("Error opening file %s\n", filename);
		return NULL;
	}

	int magic_number = 0;
	fread(&magic_number, sizeof(int), 1, fp);
	magic_number = ((magic_number & 0xff000000) >> 24) | 
		((magic_number & 0x00ff0000) >> 8) |
		((magic_number & 0x0000ff00) << 8) |
		((magic_number & 0x000000ff) << 24);

	if (magic_number != 2051) {
		printf("Invalid MNIST image file format\n");
		fclose(fp);
		return NULL;
	}

	fread(num_images, sizeof(int), 1, fp);
	*num_images = ((*num_images & 0xff000000) >> 24) |
		((*num_images & 0x00ff0000) >> 8) |
		((*num_images & 0x0000ff00) << 8) |
		((*num_images & 0x000000ff) << 24);

	int rows, cols;
	fread(&rows, sizeof(int), 1, fp);
	fread(&cols, sizeof(int), 1, fp);
	rows = ((rows & 0xff000000) >> 24) |
		((rows & 0x00ff0000) >> 8) |
		((rows & 0x0000ff00) << 8) |
		((rows & 0x000000ff) << 24);
	cols = ((cols & 0xff000000) >> 24) |
		((cols & 0x00ff0000) >> 8) |
		((cols & 0x0000ff00) << 8) |
		((cols & 0x000000ff) << 24);

	if (rows != IMG_HEIGHT || cols != IMG_WIDTH) {
		printf("Invalid image dimensions\n");
		fclose(fp);
		return NULL;
	}

	float* images = (float*)malloc(*num_images * IMG_HEIGHT * IMG_WIDTH * sizeof(float));
	unsigned char* temp = (unsigned char*)malloc(IMG_HEIGHT * IMG_WIDTH);

	for (int i = 0; i < *num_images; i++) {
		fread(temp, 1, IMG_HEIGHT * IMG_WIDTH, fp);
		for (int j = 0; j < IMG_HEIGHT * IMG_WIDTH; j++) {
			images[i * IMG_HEIGHT * IMG_WIDTH + j] = temp[j] / 255.0f;
		}
	}

	free(temp);
	fclose(fp);
	return images;
}

float* read_mnist_labels(const char* filename, int* num_labels) {
	FILE* fp = fopen(filename, "rb");
	if (!fp) {
		printf("Error opening file %s\n", filename);
		return NULL;
	}

	int magic_number = 0;
	fread(&magic_number, sizeof(int), 1, fp);
	magic_number = ((magic_number & 0xff000000) >> 24) |
		((magic_number & 0x00ff0000) >> 8) |
		((magic_number & 0x0000ff00) << 8) |
		((magic_number & 0x000000ff) << 24);

	if (magic_number != 2049) {
		printf("Invalid MNIST label file format\n");
		fclose(fp);
		return NULL;
	}

	fread(num_labels, sizeof(int), 1, fp);
	*num_labels = ((*num_labels & 0xff000000) >> 24) |
		((*num_labels & 0x00ff0000) >> 8) |
		((*num_labels & 0x0000ff00) << 8) |
		((*num_labels & 0x000000ff) << 24);

	float* labels = (float*)calloc(*num_labels * NUM_CLASSES, sizeof(float));
	unsigned char* temp = (unsigned char*)malloc(*num_labels);

	fread(temp, 1, *num_labels, fp);
	for (int i = 0; i < *num_labels; i++) {
		labels[i * NUM_CLASSES + temp[i]] = 1.0f;
	}

	free(temp);
	fclose(fp);
	return labels;
}

int main() {
	// load mnist
	int num_train_images, num_train_labels;
	float* train_images = read_mnist_images("train-images-idx3-ubyte", &num_train_images);
	float* train_labels = read_mnist_labels("train-labels-idx1-ubyte", &num_train_labels);

	// creating a lenet-5 inspired network
	Network* network = create_network(8);
	network->layers[0] = create_input(IMG_HEIGHT, IMG_WIDTH, 1);
	network->layers[1] = create_conv(IMG_HEIGHT, IMG_WIDTH, 1, 6, 5, 1, 2);
	network->layers[2] = create_maxpool(network->layers[1]->height, network->layers[1]->width, network->layers[1]->channels, 2, 2);
	network->layers[3] = create_conv(network->layers[2]->height, network->layers[2]->width, network->layers[2]->channels, 16, 5, 1, 0);
	network->layers[4] = create_maxpool(network->layers[3]->height, network->layers[3]->width, network->layers[3]->channels, 2, 2);
	network->layers[5] = create_fc(120, network->layers[4]->height * network->layers[4]->width * network->layers[4]->channels, a_sigmoid);
	network->layers[6] = create_fc(84, 120, a_sigmoid);
	network->layers[7] = create_fc(NUM_CLASSES, 84, a_softmax);

	// training loop
	for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
		float total_loss = 0.0f;
		int correct = 0;

		for (int i = 0; i < num_train_images; i++) {
			// forward pass
			network_forward(network, &train_images[i * IMG_HEIGHT * IMG_WIDTH]);

			// accuracy
			float* output = network->layers[network->num_layers - 1]->output;
			int predicted = 0;
			float max_prob = output[0];
			for (int j = 1; j < NUM_CLASSES; j++) {
				if (output[j] > max_prob) {
					max_prob = output[j];
					predicted = j;
				}
			}

			int true_label = 0;
			for (int j = 0; j < NUM_CLASSES; j++) {
				if (train_labels[i * NUM_CLASSES + j] > 0.5f) {
					true_label = j;
					break;
				}
			}

			if (predicted == true_label) correct++;

			// backprop 
			network_backward(network, &train_labels[i * NUM_CLASSES], LEARNING_RATE);

			// cross entropy loss
			float loss = 0.0f;
			for (int j = 0; j < NUM_CLASSES; j++) {
				if (train_labels[i * NUM_CLASSES + j] > 0.5f) {
					loss -= log(output[j] + 1e-10);
				}
			}
			total_loss += loss;

			// progress
			if ((i + 1) % 100 == 0) {
				printf("Epoch %d/%d, Step %d/%d, Loss: %.4f, Accuracy: %.2f%%\n",
					 epoch + 1, NUM_EPOCHS, i + 1, num_train_images,
					 total_loss / (i + 1), 100.0f * correct / (i + 1));
			}
		}

		printf("Epoch %d/%d completed, Average Loss: %.4f, Accuracy: %.2f%%\n",
				 epoch + 1, NUM_EPOCHS, total_loss / num_train_images,
				 100.0f * correct / num_train_images);
	}

	// Clean up
	free(train_images);
	free(train_labels);
	destroy_network(network);

	return 0;
}
