// *********************** Back-Prop Through Time ***************************
// Try to learn input-output pairs with flexible iteration

// As a first step we only unfold once (to learn a 2-step operation).

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <time.h>				// time as random seed in create_NN()
#include "BPTT-RNN.h"

extern double rectifier(double);

#define Eta 0.01				// learning rate
#define BIASOUTPUT 1.0			// output for bias. It's always 1.

//************************ create neural network *********************//
// GIVEN: how many layers, and how many neurons in each layer

void create_BPTT_NN(RNN *net, int numLayers, int *neuronsOfLayer)
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

void free_BPTT_NN(RNN *net, int *neuronsOfLayer)
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
// Propagate throught the *unfolded* network n times.
// Record all activities (output)

void forward_BPTT(RNN *net, int dim_V, double V[], int nfold)
	{
	int numLayers = net->numLayers;
	rLAYER lastLayer = net->layers[numLayers - 1];

	for (int t = 0; t < nfold; ++t) // for each unfolding...
		{
		//set the output of input layer
		for (int k = 0; k < dim_V; ++k)
			net->layers[0].neurons[k].output[t] =
				(t == 0 ?
				V[k] :
				// feed output of last layer back to input
				lastLayer.neurons[k].output[t - 1]
				);

		//calculate output from hidden layers to output layer
		for (int l = 1; l < numLayers; l++)
			{
			for (int n = 0; n < net->layers[l].numNeurons; n++)
				{
				double v = 0; //induced local field for neurons
				//calculate v, which is the sum of the product of input and weights
				for (int k = 0; k <= net->layers[l - 1].numNeurons; k++)
					{
					if (k == 0)
						v += net->layers[l].neurons[n].weights[k] * BIASOUTPUT;
					else
						v += net->layers[l].neurons[n].weights[k] *
							net->layers[l - 1].neurons[k - 1].output[t];
					}

				extern double rectifier(double);
				net->layers[l].neurons[n].output[t] = rectifier(v);
				
				// This is to prepare for back-prop
				#define Leakage 0.0
				if (v < -1.0)
					net->layers[l].neurons[n].grad[t] = -Leakage;
				// if (v > 1.0)
				//	net->layers[l].neurons[n].grad[t] = Leakage;
				else
					net->layers[l].neurons[n].grad[t] = 1.0;
				}
			}
		}
	}

//*************************** Back-Prop Through Time ***************************//

void backprop_through_time(RNN *net, double *errors, int nfold)
	{
	int numLayers = net->numLayers;
	rLAYER lastLayer = net->layers[numLayers - 1];

	for (int t = nfold - 1; t >= 0; --t) // back-prop through time...
		{
		if (t == nfold - 1)
			// calculate ∇ for output layer
			for (int n = 0; n < lastLayer.numNeurons; ++n)
				{
				// double output = lastLayer.neurons[n].output[t];
				//for output layer, ∇ = y∙(1-y)∙error
				lastLayer.neurons[n].grad[t] *= errors[n];
				}
		else
			// for the "recurrent" layer
			for (int n = 0; n < lastLayer.numNeurons; ++n)
				{
				// double output = lastLayer.neurons[n].output[t];
				double sum = 0.0f;
				rLAYER nextLayer = net->layers[1];
				for (int i = 0; i < nextLayer.numNeurons; i++) // for each weight
					{
					sum += nextLayer.neurons[i].grad[t + 1];
					}
				lastLayer.neurons[n].grad[t] *= sum;
				}

		// calculate ∇ for hidden layers
		for (int l = numLayers - 2; l > 0; --l) // for each hidden layer
			{
			for (int n = 0; n < net->layers[l].numNeurons; n++) // for each neuron in layer
				{
				// double output = net->layers[l].neurons[n].output[t];
				double sum = 0.0f;
				rLAYER nextLayer = net->layers[l + 1];
				for (int i = 0; i < nextLayer.numNeurons; i++) // for each weight
					{
					sum += nextLayer.neurons[i].weights[n + 1] // ignore weights[0] = bias
							* nextLayer.neurons[i].grad[t];
					}
				net->layers[l].neurons[n].grad[t] *= sum;
				}
			}
		}

	// update all weights
	for (int t = nfold - 1; t >= 0; --t) // sum over all time...
		{
		for (int l = 1; l < numLayers; ++l) // except for 0th layer which has no weights
			{
			for (int n = 0; n < net->layers[l].numNeurons; n++) // for each neuron
				{
				net->layers[l].neurons[n].weights[0] += Eta *
						net->layers[l].neurons[n].grad[t] * 1.0; // 1.0f = bias input
				for (int i = 0; i < net->layers[l - 1].numNeurons; i++) // for each weight
					{
					double inputForThisNeuron = net->layers[l - 1].neurons[i].output[t];
					net->layers[l].neurons[n].weights[i + 1] += Eta *
							net->layers[l].neurons[n].grad[t] * inputForThisNeuron;
					}
				}
			}
		}
	}
