#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <gsl/matrix>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>

#define ALPHA 0.2
#define LEARNING_RATE 0.01


typedef struct Layer {
    struct Layer* previous;
    struct Layer* next;
    int neurons; // number of neurons
    gsl_matrix* weights; // make a matrix of size m x n, where m is the number of neurons in the
                         // next layer while n is the number of neurons in the current layer
                         // -> exploit BLAS to matmul and get the results of the next layer
    gsl_matrix* values;  // the layer's values
} Layer;

double uniformrandom(double low, double high) {
    // [low, high)
    return low + ((double)rand() / (RAND_MAX / (high - low)));
}

double relu(double input, double alpha) {
    return (input >= 0) ? input : (alpha * input);
}

Layer* createlayer(Layer* lprev, Layer* lnext, int neurons, gsl_matrix* nvalues) {
    Layer* self = (Layer*) calloc(1, sizeof(Layer));
    if (self == NULL) return NULL;
    self->previous = lprev;
    self->next = lnext;
    // number of neurons MUST be more than zero sigma
    self->neurons = neurons;
    
    assert(neurons == nvalues->size1);
    self->values = nvaules;

    // setup the weights matrix
    assert(lnext != NULL);
    self->weights = gsl_matrix_calloc(lnext->neurons, neurons);
    // make the matrix have uniform random values from -0.5 to 0.5
    gsl_matrix_set_all(self->weights, uniformrandom(-0.5, 0,5));
    return self;
}

void freelayer(Layer* layer) {
    assert(layer != NULL);
    if (layer->weights != NULL) gsl_matrix_free(layer->weights);
    if (layer->values != NULL) gsl_matrix_free(layer->values);
    free(layer);
}

void forwardprop(Layer* layer) {
    assert(layer->next != NULL);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, layer->weights, layer->values, 0, layer->next->values);
    // layer->next->values will only ever have a single row
    for(unsigned int i = 0; i < layer->next->values->size1; i++) {
        double davalue = gsl_matrix_get(layer->next->values, i, 0);
        gsl_matrix_set(layer->next->values, i, 0, relu(davalue, ALPHA));
    }
}

double matrixsum(gsl_matrix* matrix) {
    double result;
    for (unsigned int i = 0; i < matrix->size1; i++) {
        for (unsigned int j = 0; j < matrix->size2; j++) {
            result += gsl_matrix_get(matrix, i, j);
        }
    }
    return result;
}

double msecost(Layer* layer, gsl_matrix* expected) {
    // mean squared error cost fxn
    // only on output layer
    assert(layer->values->size1 == expected->size1);
    gsl_matrix* result = gsl_matrix_alloc(expected->size1, 1);
    gsl_matrix_memcpy(result, layer->values);
    gsl_matrix_sub(result, expected);
    gsl_matrix_mul_elements(result, result); // squares matrix
    double matsum = matrixsum(result);
    gsl_matrix_free(result);
    return (((double)1 / layer->neurons) * matsum);
    // you dont need this for mean squared error. need this if you implement a diff cost function
}

void backprop(Layer* layer, gsl_matrix* expected) {
    // b/c you use mse, you can just do like ouput layer - expected output (matrix subtraction)
    assert(layer->previous != NULL);
        // signifies this is the output layer - previous layer would be hidden layer (ideally)
        gsl_matrix* deltao = gsl_matrix_alloc(layer->neurons, 1);
        gsl_matrix_memcpy(deltao, layer->values);
        gsl_matrix_sub(deltao, expected);
        gsl_matrix* prevlayertranposed = gsl_matrix_alloc(1, layer->previous->values->size2);
        gsl_matrix_transpose_memcpy(prevlayertranposed, layer->previous->values);
        gsl_matrix* updatedweights = gsl_matrix_alloc(layer->neurons, layer->previous->neurons);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, deltao, prevlayertranposed, 0.0, updatedweights);
        gsl_matrix_scale(updatedweights, (double)(LEARNING_RATE * -1.00));
        gsl_matrix_memcpy(layer->previous->weights, updatedweights);
        
    }
}
