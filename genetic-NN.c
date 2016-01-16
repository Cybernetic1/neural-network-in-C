// Use genetic programming to evolve neural network

// Explanation:
// * bunch of genes encode a network
// * each neuron is encoded by a (full) weight vector (dim = width of network)
// * layer = array of vectors, network = layers of arrays of vectors = N∙L vectors
// * genome would be a very big array.  But that is even for 1 individual, right?
// * if the layers do not interbreed, then each layer has its own population
// * we can select the top-N neurons in each layer's population
// * in that case, the population size for each layer is M > N

// TO-DO:
// * that means we can evole the net given in-out pairs.
// * can it generalize to mutiple-folds?  maybe.
// * as a 1st step, maybe combine with stochastic forward-backward
// * but a single-fold of genetic-NN learning may be more destructive than standard
//	back-prop, so it may be less suited for n-fold?

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>			// time as random seed in create_NN()
#include "feedforward-NN.h"

#define Eta 0.01			// learning rate
#define BIASOUTPUT 1.0		// output for bias. It's always 1.

#define L				4		// L = number of layers
#define N				5		// N = number of neurons per layer
#define M				10		// M = number of "candidate" neurons per layer; M > N
#define NumBits			64
#define MaxGens			100
#define PopSize			100
#define CrossRate		0.98
#define MutationRate	(1.0 / NumBits)

extern int rand(void);
extern void qsort(void *, size_t, size_t, int (*compar)(const void *, const void*));

void randomWeights(float w[])
	{
	for (int i = 0; i < N; ++i)
		w[i] = (rand() / (double) RAND_MAX) * 2.0 - 1.0;	// w ∊ [-1,1]
	}

int fitness(float neuron[])
// Call forward-prop with input-output pairs to evaluate the network
	{
	// build network
	NNet *Net = (NNet *) malloc(sizeof (NNet));
	int neuronsOfLayer[L];
	for (int l = 0; l < L; ++l)
		neuronsOfLayer[l] = N;
	int dimK = N;		// dimension of input-layer vector
	create_gNN(Net, L, neuronsOfLayer, neuron);
	LAYER lastLayer = Net->layers[L - 1];
	double errors[dimK];
	
	// forward_prop
	#define NumTrials	100
	for (i = 0; i < NumTrials; ++i)
		{
		forward_prop(Net, dimK, K);
		}
	}

void binaryTournament(float *selector, float pop[][NumBits])
	{
	int i = (rand() / (double) RAND_MAX) * PopSize;
	int j = (rand() / (double) RAND_MAX) * PopSize;

	float *p1 = pop[i], *p2 = pop[j];

	float *p;
	if (fitness(p1) > fitness(p2))
		p = p1;
	else
		p = p2;

	for (int k = 0; k < NumBits; ++k)
		selector[k] = p[k];
	}

void pointMutation(float *dna, float rate)
	{
	for (int i = 0; i < NumBits; ++i)
		if ((rand() / (double) RAND_MAX) < rate)
			dna[i] = (dna[i] == '0') ? '1' : '0';
	}

void crossOver(float *result, float *parent1, float *parent2, float rate)
	{
	if ((rand() / (double) RAND_MAX) > rate)
		{
		for (int i = 0; i < NumBits; ++i)
			result[i] = parent1[i];
		return;
		}

	int point = (rand() / (double) RAND_MAX) * NumBits;
	int i;
	for (i = 0; i < point; ++i)
		result[i] = parent1[i];
	for (; i < NumBits; ++i)
		result[i] = parent2[i];
	}

// **** Reproduce for 1 generation
void reproduce(float children[][NumBits], float selected[][NumBits], int popSize, float crossRate, float mutationRate)
	{
	float *p1, *p2;

	for (int i = 0; i < PopSize; ++i)
		{
		p1 = selected[i];
		p2 = (i % 2 == 0) ? selected[i + 1] : selected[i - 1];
		if (i == PopSize - 1)
			p2 = selected[0];

		crossOver(children[i], p1, p2, crossRate);
		pointMutation(children[i], mutationRate);
		}
	}

void printCandidate(float candidate[NumBits])
	{
	for (int i = 0; i < NumBits; ++i)
		printf("%c", (candidate[i] == '0') ? ' ' : '*');
	printf("\n");
	}

// Main algorithm for genetic search
void evolve()
	{
	// initialize population
	float population[PopSize][NumBits];
	for (int i = 0; i < PopSize; ++i)
		randomBitString(population[i]);

	float compareDNA(float x[NumBits], float y[NumBits])
		{
		return (fitness(x) < fitness(y));
		}

	// Sort population
	qsort(population, PopSize, NumBits, compareDNA);
	printf("Initial population:\n");
	for (int i = 0; i < PopSize; ++i)
		{
		printCandidate(population[i]);
		}

	float selected[PopSize][NumBits];
	float children[PopSize][NumBits];
	float best[NumBits];

	for (int i = 0; i < MaxGens; ++i)
		{
		printf("gen %03d: \n", i);

		for (int j = 0; j < PopSize; ++j)
			binaryTournament(selected[j], population);

		qsort(selected, PopSize, NumBits, compareDNA);

		reproduce(children, selected, PopSize, CrossRate, MutationRate);

		qsort(children, PopSize, NumBits, compareDNA);

		for (int k = 0; k < PopSize; ++k)
			{
			printCandidate(children[k]);
			}

		printf("fitness of child 0: %d\n", fitness(children[0]));

		if (fitness(children[0]) > fitness(best))
			strncpy(best, children[0], sizeof(best));	// dest, src
		printf("best: %d\n", fitness(best));
		printCandidate(best);

		strncpy(population, children, sizeof(population));

		if (fitness(best) == NumBits)
			{
			printf("Success!!!\n");
			break;
			}

		// Thread.sleep(500)
		getchar();
		}

	printf("Finished.\n");
	}

//****************************create neural network*********************//
// GIVEN: how many layers, and how many neurons in each layer
void create_gNN(NNET *net, int numLayers, int *neuronsOfLayer, float dna[][L])
	{
	srand(time(NULL));
	net->numLayers = numLayers;

	assert(numLayers >= 3);

	net->layers = (LAYER *) malloc(numLayers * sizeof (LAYER));
	//construct input layer, no weights
	net->layers[0].numNeurons = neuronsOfLayer[0];
	net->layers[0].neurons = (NEURON *) malloc(neuronsOfLayer[0] * sizeof (NEURON));

	//construct hidden layers
	for (int l = 1; l < numLayers; l++) //construct layers
		{
		net->layers[l].neurons = (NEURON *) malloc(neuronsOfLayer[l] * sizeof (NEURON));
		net->layers[l].numNeurons = neuronsOfLayer[l];
		for (int n = 0; n < neuronsOfLayer[l]; n++) // construct each neuron in the layer
			{
			net->layers[l].neurons[n].weights =
					(double *) malloc((neuronsOfLayer[l - 1] + 1) * sizeof (double));
			for (int i = 0; i <= neuronsOfLayer[l - 1]; i++)
				{
				//construct weights of neuron from previous layer neurons
				//when k = 0, it's bias weight
				net->layers[l].neurons[n].weights[i] = randomWeight();
				//net->layers[i].neurons[j].weights[k] = 0.0f;
				}
			}
		}
	}

void free_gNN(NNET *net, int *neuronsOfLayer)
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

#ifdef CRAP

//**************************** forward-propagation ***************************//

void forward_prop(NNET *net, int dim_V, double V[])
	{
	// set the output of input layer
	for (int i = 0; i < dim_V; ++i)
		net->layers[0].neurons[i].output = V[i];

	// calculate output from hidden layers to output layer
	for (int l = 1; l < net->numLayers; l++)
		{
		for (int n = 0; n < net->layers[l].numNeurons; n++)
			{
			double v = 0; //induced local field for neurons
			// calculate v, which is the sum of the product of input and weights
			for (int k = 0; k <= net->layers[l - 1].numNeurons; k++)
				{
				if (k == 0)
					v += net->layers[l].neurons[n].weights[k] * BIASOUTPUT;
				else
					v += net->layers[l].neurons[n].weights[k] *
						net->layers[l - 1].neurons[k - 1].output;
				}

			// For the last layer, skip the sigmoid function
			// Note: this idea seems to destroy back-prop convergence
			// if (i == net->numLayers - 1)
			//	net->layers[i].neurons[j].output = v;
			// else
			net->layers[l].neurons[n].output = sigmoid(v);
			}
		}
	}

// Same as above, except with soft_plus activation function
void forward_prop_SP(NNET *net, int dim_V, double V[])
	{
	// set the output of input layer
	for (int i = 0; i < dim_V; ++i)
		net->layers[0].neurons[i].output = V[i];

	// calculate output from hidden layers to output layer
	for (int l = 1; l < net->numLayers; l++)
		{
		for (int n = 0; n < net->layers[l].numNeurons; n++)
			{
			double v = 0.0; // induced local field for neurons
			// calculate v, which is the sum of the product of input and weights
			for (int k = 0; k <= net->layers[l - 1].numNeurons; k++)
				{
				if (k == 0)
					v += net->layers[l].neurons[n].weights[k] * BIASOUTPUT;
				else
					v += net->layers[l].neurons[n].weights[k] *
						net->layers[l - 1].neurons[k - 1].output;
				}

			net->layers[l].neurons[n].output = softplus(v);

			net->layers[l].neurons[n].grad = d_softplus(v);
			}
		}
	}

// Same as above, except with rectifier activation function
// ReLU = "rectified linear unit"
void forward_prop_ReLU(NNET *net, int dim_V, double V[])
	{
	// set the output of input layer
	for (int i = 0; i < dim_V; ++i)
		net->layers[0].neurons[i].output = V[i];

	// calculate output from hidden layers to output layer
	for (int l = 1; l < net->numLayers; l++)
		{
		for (int n = 0; n < net->layers[l].numNeurons; n++)
			{
			double v = 0.0; // induced local field for neurons
			// calculate v, which is the sum of the product of input and weights
			for (int k = 0; k <= net->layers[l - 1].numNeurons; k++)
				{
				if (k == 0)
					v += net->layers[l].neurons[n].weights[k] * BIASOUTPUT;
				else
					v += net->layers[l].neurons[n].weights[k] *
						net->layers[l - 1].neurons[k - 1].output;
				}

			net->layers[l].neurons[n].output = rectifier(v);
			
			// This is to prepare for back-prop
			if (v < -1.0)
				net->layers[l].neurons[n].grad = -Leakage;
			// if (v > 1.0)
			//	net->layers[l].neurons[n].grad = Leakage;
			else
				net->layers[l].neurons[n].grad = 1.0;
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

// There is a neat trick for the calculation of σ':  σ'(x) = σ(x) (1−σ(x))
// For its simple derivation you can see this post:
// http://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
// Therefore in the code, we use "output * (1 - output)" for the value of "σ'(summed input)",
// because output = σ(summed input), where summed_input_i = Σ_j W_ji input_j.

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

void back_prop(NNET *net, double *errors)
	{
	int numLayers = net->numLayers;
	LAYER lastLayer = net->layers[numLayers - 1];

	// calculate gradient for output layer
	for (int n = 0; n < lastLayer.numNeurons; ++n)
		{
		double output = lastLayer.neurons[n].output;
		//for output layer, ∇ = y∙(1-y)∙error
		lastLayer.neurons[n].grad = steepness * output * (1.0 - output) * errors[n];
		}

	// calculate gradient for hidden layers
	for (int l = numLayers - 2; l > 0; --l)		// for each hidden layer
		{
		for (int n = 0; n < net->layers[l].numNeurons; n++)		// for each neuron in layer
			{
			double output = net->layers[l].neurons[n].output;
			double sum = 0.0f;
			LAYER nextLayer = net->layers[l + 1];
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

// Same as above, except with rectifier activation function
// In this case:  σ'(x) = sign(x)
void back_prop_ReLU(NNET *net, double *errors)
	{
	int numLayers = net->numLayers;
	LAYER lastLayer = net->layers[numLayers - 1];

	// calculate gradient for output layer
	for (int n = 0; n < lastLayer.numNeurons; ++n)
		{
		// double output = lastLayer.neurons[n].output;
		//for output layer, ∇ = sign(y)∙error
		// .grad has been prepared in forward-prop
		lastLayer.neurons[n].grad *= errors[n];
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
double calc_error(NNET *net, double Y[], double *errors)
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

#endif
