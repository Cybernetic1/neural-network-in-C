// Use genetic programming to evolve neural network

// Explanation:
// * bunch of genes encode a network
// * each neuron is encoded by a (full) weight vector (dim = width of network)
// * layer = array of vectors, network = layers of arrays of vectors = N∙L vectors
// * genome would be a very big array.  But that is even for 1 individual, right?
// * if the layers do not interbreed, then each layer has its own population
// * we can select the top-N neurons in each layer's population
// * in that case, the population size for each layer is M > N

// How to extend to recurrent case?
// * Perhaps the same idea as stochastic forward-backward?

// TO-DO:
// * that means we can evole the net given in-out pairs.
// * can it generalize to mutiple-folds?  maybe.
// * as a 1st step, maybe combine with stochastic forward-backward
// * but a single-fold of genetic-NN learning may be more destructive than standard
//   back-prop, so it may be less suited for n-fold?

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>			// time as random seed in create_NN()
// #include "feedforward-NN.h"

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

// A question is how to store the current network as well as the entire population.
// Perhaps the data structure should store all the "population rows".
float population[L][M][N];		// each element is a connection weight
float output[L][M];				// output of each neuron
float grad[L][M];				// local gradient for each neuron

int neuronsOfLayer[L] = { N };		// initialize all layers to have N neurons
int dimK = N;						// dimension of input-layer vector

extern int rand(void);
extern void qsort(void *, size_t, size_t, int (*compar)(const void *, const void*));

void randomWeights(float w[])
	{
	for (int i = 0; i < N; ++i)
		w[i] = (rand() / (double) RAND_MAX) * 2.0 - 1.0;	// w ∊ [-1,1]
	}

// Perhaps each neuron is an individual, in the sense that neurons compete with each other.
// The network should consist of the top-N neurons in each population row.
// Fitness of neuron = ∑ (∂E/∂W)² where E = error, relative to one input-output pair, and we sum over such I/O pairs
float fitness(int layer, int neuron)	// input = index of neuron to be scored
// Call forward-prop with input-output pairs to evaluate the current network.
	{
	// initialize network
	float lastLayer[][] = population[L - 1];
	float K[dimK];
	float errors[dimK];

	// forward_prop
	#define NumTrials	100
	for (int i = 0; i < NumTrials; ++i)
		{
		// prepare input and ideal output values
		
		
		forward_gNN(dimK, K);		// forward-propagate the gNN

		// Calculate the error
		for (int n = 0; n < N; ++N)
			{
			errors[n] = output[L - 1][n] - ideal[n];
			}

		// Use back-prop to calculate local gradients
		backprop_gNN(errors);
		// Then fitness = sum of local gradients for a neuron, relative to 1 example.
		// So we need to add up the fitnesses for all examples.
		}
	}

// This seems to be independent of gene expression
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

// Each neuron is an individual, a point mutation mutates a single weight within the neuron
void pointMutation(float *dna, float rate)
	{
	for (int i = 0; i < NumBits; ++i)
		if ((rand() / (double) RAND_MAX) < rate)
			dna[i] = (dna[i] == '0') ? '1' : '0';
	}

// Cross-over of 2 neurons
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
	// No need to create neural network as it is stored in the population
	// initialize population
	for (int l = 0; l < L; ++l)
		for (int m = 0; m < M; ++m)
			for (int n = 0; n < N; ++n)
				population[l][m][n] = (rand() / (double) RAND_MAX) * 2.0 - 1.0;	// w ∊ [-1,1]

	// Compare fitness of 2 neurons
	// The neurons are addressed by layer and position
	float compareFitness(int layer, int x_index, int y_index)
		return (fitness(layer, x_index) < fitness(layer, y_index));

	// Sort population according to fitness
	// Perhaps the fitness of a neuron inside a network can be defined independent of other
	// neurons in the network?  For each input-output pair, the network incurs an error.
	// Can we associate a component of this error to an individual neuron?  Perhaps that is
	// gvien by the "local gradient" of the neuron?
	// The fitness of a neuron can be defined as:  ∑ (∂E/∂W)²
	// where summation is over the weights belonging to the neuron in question;  E is the
	// error w.r.t. a single input-output pair, so we should do another summation over all
	// data points.
	// The rationale for the above formula is that ∂E/∂W measures the error sensitivity of a
	// single connection (ie, weight).
	// Armed with this fitness measure, we can continue with the strategy of maintain M
	// neurons per layer and selecting N out of M to build the actual network.
	qsort(population, PopSize, NumBits, compareFitness);
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

		qsort(selected, PopSize, NumBits, compareFitness);

		reproduce(children, selected, PopSize, CrossRate, MutationRate);

		qsort(children, PopSize, NumBits, compareFitness);

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

/* No need to create neural network as it is stored in the population
 * 
//****************************create neural network*********************
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
*/

//**************************** forward-propagation ***************************//
void forward_gNN(int dim_V, double V[])
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

			net->layers[l].neurons[n].output = sigmoid(v);
			}
		}
	}

//****************************** back-propagation ***************************//
// The meaning of del (∇) is the "local gradient".  At the output layer, ∇ is equal to
// the derivative σ'(summed inputs) times the error signal, while on hidden layers it is
// equal to the derivative times the weighted sum of the ∇'s from the "next" layers.
// From the algorithmic point of view, ∇ is derivative of the error with respect to the
// summed inputs (for that particular neuron).  It changes for every input instance because
// the error is dependent on the NN's raw input.  So, for each raw input instance, the
// "local gradient" keeps changing.  I have a hypothesis that ∇ will fluctuate wildly
// when the NN topology is "inadequate" to learn the target function.

void backprop_gNN(double *errors)
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

#ifdef CRAP

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
