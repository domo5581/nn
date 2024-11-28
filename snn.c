#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

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

