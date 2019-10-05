// TO-DO:

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>		// constants "true" and "false"
#include <math.h>
#include <assert.h>
#include <time.h>			// time as random seed in create_NN()
#include "QNET.h"

#define Eta 0.01			// learning rate
#define BIASINPUT 1.0		// input for bias. It's always 1.

double randomWeight()		// generate random weight between [+1.0, -1.0]
	{
	// return 0.5 + (rand() / (double) RAND_MAX) * 0.01;
	return (rand() / (double) RAND_MAX) * 2.0 - 1.0;
	}

//****************************create neural network*********************//
// GIVEN: how many layers, and how many neurons in each layer
QNET *create_QNN(int numLayers, int *neuronsPerLayer)
	{
	QNET *net = (QNET *) malloc(sizeof (QNET));
	srand(time(NULL));
	net->numLayers = numLayers;

	assert(numLayers >= 3);

	net->layers = (LAYER *) malloc(numLayers * sizeof (LAYER));
	//construct input layer, no weights
	net->layers[0].numNeurons = neuronsPerLayer[0];
	net->layers[0].neurons = (NEURON *) malloc(neuronsPerLayer[0] * sizeof (NEURON));

	//construct hidden layers
	for (int l = 1; l < numLayers; ++l) //construct layers
		{
		net->layers[l].neurons = (NEURON *) malloc(neuronsPerLayer[l] * sizeof (NEURON));
		net->layers[l].numNeurons = neuronsPerLayer[l];
		net->layers[l].α = randomWeight();
		net->layers[l].β = randomWeight();
		net->layers[l].γ = randomWeight();
		net->layers[l].δ = randomWeight();
		}
	return net;
	}

void re_randomize(QNET *net, int numLayers, int *neuronsPerLayer)
	{
	srand(time(NULL));

	for (int l = 1; l < numLayers; ++l)							// for each layer
		{
		net->layers[l].α = randomWeight();
		net->layers[l].β = randomWeight();
		net->layers[l].γ = randomWeight();
		net->layers[l].δ = randomWeight();
		}
	}

void free_QNN(QNET *net, int *neuronsPerLayer)
	{
	// for input layer
	free(net->layers[0].neurons);

	// for each hidden layer
	int numLayers = net->numLayers;
	for (int l = 1; l < numLayers; l++) // for each layer
		free(net->layers[l].neurons);

	// free all layers
	free(net->layers);

	// free the whole net
	free(net);
	}

//**************************** forward-propagation ***************************//

void forward_prop_quadratic(QNET *net, int dim_V, double V[])
	{
	// set the output of input layer
	for (int i = 0; i < dim_V; ++i)
		net->layers[0].neurons[i].output = V[i];

	// calculate output from hidden layers to output layer
	for (int l = 1; l < net->numLayers; l++)
		{
		for (int n = 0; n < net->layers[l].numNeurons; n++)
			{
			double v = 0.0; //induced local field for neurons
			// calculate v, which is the sum of the product of input and weights
			for (int j = 0; j <= net->layers[l - 1].numNeurons; j++)
				for (int k = 0; k <= net->layers[l - 1].numNeurons; k++)
					{
					if (j == 0)
						input1 = BIASINPUT;
					else
						input1 = net->layer[l-1].neurons[j-1].output;

					if (k == 0)
						input2 = BIASINPUT;
					else
					 	input2 = net->layer[l-1].neurons[k-1].output;

					double weight;
					layer = net->layer[l]
					if (j == k)				// W[n,j,k] is on diagonal
						{
						if (n == j)			// W[n,j,k] is on super-diagonal
							weight = layer.α;
						else
							weight = layer.β;
						}
					else					// off diagonal
						{
						if (n == j || n == k)
							weight = layer.γ;
						else
							weight = layer.δ;
						}
					v += weight * input1 * input2;
					}

			net->layers[l].neurons[n].output = v;
			// No need to calculate the traditional "local gradient" because it ≡ 1.0
			}
		}
	}

//****************************** back-propagation ***************************//
// The error is propagated backwards starting from the output layer, hence the
// name for this algorithm.

// In the update formula, we need to adjust by "η ∙ input ∙ ∇", where η is the learning rate.
// The value of		∇_j = σ'(summed input) Σ_i W_ji ∇_i
// where σ is the sigmoid function, σ' is its derivative.  This formula is obtained directly
// from differentiating the error E with respect to the weights W.

// The meaning of del (∇) is the "local gradient".  At the output layer, ∇ is equal to
// the derivative σ'(summed inputs) times the error signal, while on hidden layers it is
// equal to the derivative times the weighted sum of the ∇'s from the "next" layers.
// From the algorithmic point of view, ∇ is derivative of the error with respect to the
// summed inputs (for that particular neuron).  It changes for every input instance because
// the error is dependent on the NN's raw input.  So, for each raw input instance, the
// "local gradient" keeps changing.  I have a hypothesis that ∇ will fluctuate wildly
// when the NN topology is "inadequate" to learn the target function.

// Some history:
// It was in 1974-1986 that Paul Werbos, David Rumelhart, Geoffrey Hinton and Ronald Williams
// discovered this algorithm for neural networks, although it has been described by
// Bryson, Denham, and Dreyfus in 1963 and by Bryson and Yu-Chi Ho in 1969 as a solution to
// optimization problems.  The book "Talking Nets" interviewed some of these people.

void back_prop_quadratic(QNET *net, double *errors)
	{
	int numLayers = net->numLayers;
	LAYER lastLayer = net->layers[numLayers - 1];

	// calculate gradient for output layer
	for (int n = 0; n < lastLayer.numNeurons; ++n)
		{
		// double output = lastLayer.neurons[n].output;
		// For output layer, ∇ = sign(y)∙error
		// .grad has been prepared in forward-prop
		lastLayer.neurons[n].grad = errors[n];
		}

	// calculate gradient for hidden layers
	for (int l = numLayers - 2; l > 0; --l)		// for each hidden layer
		{
		for (int n = 0; n < net->layers[l].numNeurons; n++)		// for each neuron in layer
			{
			// double output = net->layers[l].neurons[n].output;
			double sum = 0.0f;
			LAYER nextLayer = net->layers[l + 1];
			for (int i = 0; i < nextLayer.numNeurons; i++)		// for each weight
				{
				sum += nextLayer.neurons[i].weights[n + 1]		// ignore weights[0] = bias
						* nextLayer.neurons[i].grad;
				}
			// .grad has been prepared in forward-prop
			net->layers[l].neurons[n].grad *= sum;
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

// Calculate error between output of forward-prop and a given answer Y
double calc_error(QNET *net, double Y[], double *errors)
	{
	// calculate mean square error
	// desired value = Y = K* = trainingOUT
	double sumOfSquareError = 0;

	int numLayers = net->numLayers;
	LAYER lastLayer = net->layers[numLayers - 1];
	// This means each output neuron corresponds to a classification label --YKY
	for (int n = 0; n < lastLayer.numNeurons; n++)
		{
		//error = desired_value - output
		double error = Y[n] - lastLayer.neurons[n].output;
		errors[n] = error;
		sumOfSquareError += error * error / 2;
		}
	double mse = sumOfSquareError / lastLayer.numNeurons;
	return mse; //return mean square error
	}
