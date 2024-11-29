#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <gsl/matrix>

#define ALPHA 0.2

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

void cost(Layer* layer, gsl_matrix* expected) {
    // (for mnist at least) your expected will be a matrix of [10x1]
    assert(layer->values->size1 == expected->size1);

}

void backprop(Layer* layer) {
    assert(layer->previous != NULL);
    
}
