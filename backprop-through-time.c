// Try to learn input-output pairs with flexible iteration

// Strategy: let unfold N times until convergence, then back-prop through all unfoldings
// Require: back-prop with ReLU's
// forward-propagate and perhaps no need to record until convergence

// From now on we adopt a simple architecture:  The RNN is a multi-layer feed-forward
// network, with its output layer connected to its input layer.  For learning, we simply
// use traditional back-prop, the key is by allowing the network to iterate as long as
// it needs to converge to an equilibrium point, and then we use the difference between
// the equilibrium point and the target as error.

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <time.h>				// time as random seed in create_NN()
#include "RNN.h"

#define Eta 0.001				// learning rate
#define BIASOUTPUT 1.0			// output for bias. It's always 1.

//************************ create neural network *********************//
// GIVEN: how many layers, and how many neurons in each layer
void create_RNN(RNN *net, int numLayers, int *neuronsOfLayer)
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

void free_RNN(RNN *net, int *neuronsOfLayer)
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
// It seems that how many times I unfold, I should also duplicate the network so many times.
// How about just 2 times?  But then what is the difference between 2-times unfolded and
// a network that is twice-thick?
// The weights seem to be repeated
// ...does the update rule update the weights multiple times?

void forward_RNN(RNN *net, int dim_V, double V[])
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

void unfold_RNN(RNN *net, int numLayers, int *neuronsOfLayer)
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

//*************************** Back-Prop Through Time ***************************//

void backprop_through_time(RNN *net)
	{
	int numLayers = net->numLayers;
	rLAYER lastLayer = net->layers[numLayers - 1];

	#define steepness 3.0
	// calculate ∆ for output layer
	for (int n = 0; n < lastLayer.numNeurons; ++n)
		{
		double output = lastLayer.neurons[n].output;
		double error = lastLayer.neurons[n].error;
		//for output layer, ∆ = y∙(1-y)∙error
		lastLayer.neurons[n].grad = steepness * output * (1.0 - output) * error;
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
