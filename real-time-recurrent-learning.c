// Implements the RTRL algorithm:

// First, some notation:

// Neurons are indexed by layers l, with a total of L layers
// Each layer contains N_l neurons indexed by n
// Output of each neuron is denoted Y(t) where t is the time step

// The weights W_i,j is FROM unit j TO unit i

// Each neuron recieves a weighted sum of inputs:
//		net_k(t) = sum W_ij Y(t)
// or with full indices:
//		net_[l,n](t) = sum_m W_[l,n],[l-1,m] Y_[l-1,m](t) 

// The above weighted sum then passes through the sigmoid function
//		Y_k(t+1) = sigmoid (net_k(t))
// where k is a generic index

// The individual ERROR is calculated at the output layer L:
//		e_k(t) = target_k(t) - Y_k(t)
// The total error for a single time step is 1/2 the squared sum of this.
// Our TARGET ERROR (ET) function is the integral (sum) over all time steps.

// The gradient of ET is the gradient for the current time step plus the gradient of
// previous time steps:
//		∇ ET(t0, t+1) = ∇ ET(t0,t) + ∇ E(t+1)

// As a time series is presented to the network, we can accumulate the values of the
// gradient, or equivalently, of the weight changes. We thus keep track of the value:
//		∆ W_ij(t) = -η ∂E(t)/∂W_ij
// After the network has been presented with the whole series, we alter each weight W by:
//		sum (over t) ∆ W_ij(t)

// We therefore need an algorithm that computes ∂Y(t)/∂W:
//		....

// Computation of ∂Y(t)/∂W

// ∂Y_k(t+1)/∂W_ij = sigmoid' (net_k(t)) [ sum_h W_kh ∂Y_h(t)/∂W_ij + δ_ik Y_j(t)]

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <time.h>				// time as random seed in create_NN()
#include "RNN.h"

#define Eta 0.001				// learning rate
#define BIASOUTPUT 1.0			// output for bias. It's always 1.

//****************************create neural network*********************//
// GIVEN: how many layers, and how many neurons in each layer
void create_RTRL_NN(RNN *net, int numLayers, int *neuronsOfLayer)
	{
	srand(time(NULL));
	net->numLayers = numLayers;

	assert(numLayers >= 3);

	net->layers = (rLAYER *) malloc(numLayers * sizeof (rLAYER));
	//construct input layer, no weights
	net->layers[0].numNeurons = neuronsOfLayer[0];
	net->layers[0].neurons = (rNEURON *) malloc(neuronsOfLayer[0] * sizeof (rNEURON));

	//construct hidden layers
	for (int l = 1; l < numLayers; l++) //construct layers
		{
		net->layers[l].neurons = (rNEURON *) malloc(neuronsOfLayer[l] * sizeof (rNEURON));
		net->layers[l].numNeurons = neuronsOfLayer[l];
		for (int n = 0; n < neuronsOfLayer[l]; n++) // construct each neuron in the layer
			{
			net->layers[l].neurons[n].weights =
					(double *) malloc((neuronsOfLayer[l - 1] + 1) * sizeof (double));
			for (int i = 0; i <= neuronsOfLayer[l - 1]; i++)
				{
				//construct weights of neuron from previous layer neurons
				//when k = 0, it's bias weight
				extern double randomWeight();
				net->layers[l].neurons[n].weights[i] = randomWeight();
				//net->layers[i].neurons[j].weights[k] = 0.0f;
				}
			}
		}
	}

void free_RTRL_NN(RNN *net, int *neuronsOfLayer)
	{
	// for input layer
	free(net->layers[0].neurons);

	// for each hidden layer
	int numLayers = net->numLayers;
	for (int l = 1; l < numLayers; l++) // for each layer
		{
		for (int n = 0; n < neuronsOfLayer[l]; n++) // for each neuron in the layer
			{
			free(net->layers[l].neurons[n].weights);
			}
		free(net->layers[l].neurons);
		}

	// free all layers
	free(net->layers);
	
	// free the whole net
	free(net);
	}

//**************************** forward-propagation ***************************//

void forward_RTRL(RNN *net, int dim_V, double V[])
	{
	//set the output of input layer
	//two inputs x1 and x2
	for (int i = 0; i < dim_V; ++i)
		net->layers[0].neurons[i].output = V[i];

	//calculate output from hidden layers to output layer
	for (int i = 1; i < net->numLayers; i++)
		{
		for (int j = 0; j < net->layers[i].numNeurons; j++)
			{
			double v = 0; //induced local field for neurons
			//calculate v, which is the sum of the product of input and weights
			for (int k = 0; k <= net->layers[i - 1].numNeurons; k++)
				{
				if (k == 0)
					v += net->layers[i].neurons[j].weights[k] * BIASOUTPUT;
				else
					v += net->layers[i].neurons[j].weights[k] *
						net->layers[i - 1].neurons[k - 1].output;
				}

			// For the last layer, skip the sigmoid function
			// Note: this idea seems to destroy back-prop convergence
			// if (i == net->numLayers - 1)
			//	net->layers[i].neurons[j].output = v;
			// else
				extern double sigmoid(double);
				net->layers[i].neurons[j].output = sigmoid(v);
			}
		}
	}

//****************************** RTRL ***************************//

void RTRL(RNN *net, double *errors)
	{
	int numLayers = net->numLayers;
	rLAYER lastLayer = net->layers[numLayers - 1];

	#define steepness 3.0
	// calculate ∆ for output layer
	for (int n = 0; n < lastLayer.numNeurons; ++n)
		{
		double output = lastLayer.neurons[n].output;
		//for output layer, ∆ = y∙(1-y)∙error
		lastLayer.neurons[n].grad = steepness * output * (1.0 - output) * errors[n];
		}

	// calculate ∆ for hidden layers
	for (int l = numLayers - 2; l > 0; --l)		// for each hidden layer
		{
		for (int n = 0; n < net->layers[l].numNeurons; n++)		// for each neuron in layer
			{
			double output = net->layers[l].neurons[n].output;
			double sum = 0.0f;
			rLAYER nextLayer = net->layers[l + 1];
			for (int i = 0; i < nextLayer.numNeurons; i++)		// for each weight
				{
				sum += nextLayer.neurons[i].weights[n + 1]		// ignore weights[0] = bias
						* nextLayer.neurons[i].grad;
				}
			net->layers[l].neurons[n].grad = steepness * output * (1.0 - output) * sum;
			}
		}

	// update all weights
	for (int l = 1; l < numLayers; ++l)		// except for 0th layer which has no weights
		{
		for (int n = 0; n < net->layers[l].numNeurons; n++)		// for each neuron
			{
			net->layers[l].neurons[n].weights[0] += Eta * 
					net->layers[l].neurons[n].grad * 1.0;		// 1.0f = bias input
			for (int i = 0; i < net->layers[l - 1].numNeurons; i++)	// for each weight
				{	
				double inputForThisNeuron = net->layers[l - 1].neurons[i].output;
				net->layers[l].neurons[n].weights[i + 1] += Eta *
						net->layers[l].neurons[n].grad * inputForThisNeuron;
				}
			}
		}
	}
