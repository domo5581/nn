#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <gsl/matrix>

/*
neural network
input layer -> hidden layer -> output layer
each neuron has weights, numweights = numneurons
neuron value * weight goes to another neuron, and that sum goes through activation fxn to determine that other neuron"'s output

5 input layers 3 hidden layers
each connection from a neuron carries a weight, 3 connections from a single neuron. new formula, num connections is equal to hidden num neurons * input neurons

each neuron only gets the weighted sum from that number. ie the first hidden neuron only gets the weighted sum from the input values * their respective first weights.

represent that as a matrix with # cols representing hidden neuron amount and # rows representing input neurons

[.1 .2 .3 .4 .5 w1s of each input neuron
 .3 .5 .7 .9 1.1 w2s
 .2 .4 .5 .8 1.2 ] w3s

 just multiply input values by w1 row to get hidden neuron 1's value
 repeat for hn2 and hn3

this works.
*/

typedef struct Layer {
    struct Layer* previous;
    struct Layer* next;
    int neurons; // number of neurons
    gsl_matrix* weights; // make a matrix of size m x n, where m is the number of neurons in the
                         // next layer while n is the number of neurons in the current layer
                         // -> exploit BLAS to matmul and get the results of the next layer
    gsl_matrix* values;  // the layer's values
} Layer;

Layer* createlayer(Layer* lprev, Layer* lnext, int neurons, gsl_matrix* nvalues) {
    Layer* self = (Layer*) calloc(1, sizeof(Layer));
    if (self == NULL) return NULL;
    self->previous = lprev;
    self->next = lnext;
    // number of neurons MUST be more than zero sigma
    self->neurons = neurons;
    
    assert((neurons == nvalues->size2) && (nvalues->size1 == lnext->neurons));
    self->values = nvaules;

    // setup the weights matrix
    assert(lnext != NULL);
    self->weights = gsl_matrix_calloc(lnext->neurons, neurons);
    
    return self;
}

void freelayer(Layer* layer) {
    assert(layer != NULL);
    if (layer->weights != NULL) gsl_matrix_free(layer->weights);
    if (layer->values != NULL) gsl_matrix_free(layer->values);
    free(layer);
}

void forwardprop(Layer* layer) {
    
}
