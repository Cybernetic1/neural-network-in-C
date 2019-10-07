// TO-DO:

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>		// constants "true" and "false"
#include <math.h>
#include <assert.h>
#include <time.h>			// time as random seed in create_QNN()
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
QNET *create_QNN(int numLayers)
	{
	QNET *net = (QNET *) malloc(sizeof (QNET));
	srand(time(NULL));
	net->numLayers = numLayers;

	assert(numLayers >= 3);

	net->layers = (LAYER *) malloc(numLayers * sizeof (LAYER));
	//construct input layer, no weights
	net->layers[0].neurons = (NEURON *) malloc(dim_V * sizeof (NEURON));

	//construct hidden layers
	for (int l = 1; l < numLayers; ++l) //construct layers
		{
		net->layers[l].neurons = (NEURON *) malloc(dim_V * sizeof (NEURON));
		net->layers[l].alpha = randomWeight();
		net->layers[l].beta = randomWeight();
		net->layers[l].gamma = randomWeight();
		net->layers[l].delta = randomWeight();
		}
	return net;
	}

void re_randomize(QNET *net)
	{
	srand(time(NULL));

	for (int l = 1; l < net->numLayers; ++l)					// for each layer
		{
		net->layers[l].alpha = randomWeight();
		net->layers[l].beta = randomWeight();
		net->layers[l].gamma = randomWeight();
		net->layers[l].delta = randomWeight();
		}
	}

void free_QNN(QNET *net)
	{
	// for each layer
	for (int l = 0; l < net->numLayers; l++)
		free(net->layers[l].neurons);

	// free all layers
	free(net->layers);

	// free the whole net
	free(net);
	}


//**************************** forward-propagation ***************************//

// NOTE: This version does not include the bias term.
// If bias term is included:  If the network "width" is dim_V, the matrix W's width would be
// (dim_V + 1), with the matrix's index 0 reserved for the constant term (ie, the "bias term"
// ≡ 1.0).  But the network vector's index 0 would multiply with the matrix index 1.

void forward_prop_quadratic(QNET *net, double V[])
	{
	// set the output of input layer
	for (int i = 0; i < dim_V; ++i)
		net->layers[0].neurons[i].output = V[i];

	// calculate output from hidden layers to output layer
	for (int l = 1; l < net->numLayers; l++)
		{
		// the indices n, i, j are for the matrix W
		for (int n = 0; n < dim_V; n++)
			{
			double v = 0.0;	// "induced local field" of neurons (not sure why this terminology)
			// calculate v, the sum of product of input and weights
			for (int i = 0; i < dim_V; i++)
				for (int j = 0; j < dim_V; j++)
					{
					double input1, input2;
					input1 = net->layers[l-1].neurons[i].output;
				 	input2 = net->layers[l-1].neurons[j].output;

					double weight;
					LAYER layer = net->layers[l];
					if (i == j)				// W[n,i,j] is on diagonal
						{
						if (n == i)			// W[n,i,j] is on super-diagonal
							weight = layer.alpha;
						else
							weight = layer.beta;
						}
					else					// off diagonal
						{
						if (n == i || n == j)
							weight = layer.gamma;
						else
							weight = layer.delta;
						}
					v += weight * input1 * input2;
					}

			net->layers[l].neurons[n].output = v;
			// No need to calculate the traditional "local gradient" because it ≡ 1.0
			net->layers[l].neurons[n].grad = 1.0;
			}
		}
	}

/* ***************************** back-propagation ***************************
The error is propagated backwards starting from the output layer, hence the
name for this algorithm.

In the update formula, we need to adjust by "η ∙ input ∙ ∇", where η is the learning rate.
The value of		∇_j = σ'(summed input) Σ_i W_ji ∇_i
where σ is the sigmoid function, σ' is its derivative.  This formula is obtained directly
from differentiating the error E with respect to the weights W.

The meaning of del (∇) is the "local gradient".  At the output layer, ∇ is equal to
the derivative σ'(summed inputs) times the error signal, while on hidden layers it is
equal to the derivative times the weighted sum of the ∇'s from the "next" layers.
From the algorithmic point of view, ∇ is derivative of the error with respect to the
summed inputs (for that particular neuron).  It changes for every input instance because
the error is dependent on the NN's raw input.  So, for each raw input instance, the
"local gradient" keeps changing.  I have a hypothesis that ∇ will fluctuate wildly
when the NN topology is "inadequate" to learn the target function.

In classic back-prop, the weight update is given by:
	ΔWᵢⱼ = -η ∂E/∂Wᵢⱼ = -η oᵢδⱼ
In our new, quadratic back-prop the gradient is given by:
	∂E/∂Wₖᵢⱼ = δₖoᵢoⱼ
where
	δₖ = ∑ₗ(δₗ ∑ⱼ Wₗₖⱼ oⱼ)

Some history:
It was in 1974-1986 that Paul Werbos, David Rumelhart, Geoffrey Hinton and Ronald Williams
discovered this algorithm for neural networks, although it has been described by
Bryson, Denham, and Dreyfus in 1963 and by Bryson and Yu-Chi Ho in 1969 as a solution to
optimization problems.  The book "Talking Nets" interviewed some of these people.
*/
void back_prop_quadratic(QNET *net, double *errors)
	{
	int numLayers = net->numLayers;
	LAYER lastLayer = net->layers[numLayers - 1];

	// calculate gradient for output layer
	for (int n = 0; n < dim_V; ++n)
		{
		// double output = lastLayer.neurons[n].output;
		// For output layer, ∇ = sign(y)∙error
		// .grad has been prepared in forward-prop
		lastLayer.neurons[n].grad = errors[n];
		}

	// calculate gradient for hidden layers
	for (int l = numLayers - 2; l > 0; --l)		// for each hidden layer
		{
		for (int n = 0; n < dim_V; n++)			// for each neuron in layer
			{
			// double output = net->layers[l].neurons[n].output;
			double sum = 0.0f;
			LAYER nextLayer = net->layers[l + 1];
			for (int i = 0; i < dim_V; ++i)		// for each weight
				for (int j = 0; j < dim_V; ++j)
					{
					double weight = 1.0;
					// sum += nextLayer.neurons[i].weights[n + 1]		// ignore weights[0] = bias
					//		* nextLayer.neurons[i].grad;
					sum += weight * nextLayer.neurons[i].grad;
					}
			// .grad has been prepared in forward-prop
			net->layers[l].neurons[n].grad = sum;
			}
		}

	// update all weights
	for (int l = 1; l < numLayers; ++l)		// except for 0th layer which has no weights
		{
		for (int n = 0; n < dim_V; n++)		// for each neuron
			{
			double weight;
			weight += Eta * net->layers[l].neurons[n].grad * 1.0;
			// net->layers[l].neurons[n].weights[0] += Eta *
			//		net->layers[l].neurons[n].grad * 1.0;		// 1.0f = bias input
			for (int i = 0; i < dim_V; ++i)						// for each weight
				for (int j = 0; j < dim_V; ++j)
					{
					double o_i = net->layers[l - 1].neurons[i].output;
					double o_j = net->layers[l - 1].neurons[j].output;
					double weight;
					weight += Eta * net->layers[l].neurons[n].grad * o_i * o_j;
					// net->layers[l].neurons[n].weights[i + 1] += Eta *
					//		net->layers[l].neurons[n].grad * inputForThisNeuron;
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
	for (int n = 0; n < dim_V; n++)
		{
		//error = desired_value - output
		double error = Y[n] - lastLayer.neurons[n].output;
		errors[n] = error;
		sumOfSquareError += error * error / 2;
		}
	double mse = sumOfSquareError / dim_V;
	return mse; //return mean square error
	}
