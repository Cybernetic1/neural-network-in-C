// Use genetic programming to evolve neural network

// Explanation:
// * evole the net given in-out pairs
// * bunch of genes encode a network
// * each gene = DFT of weights, serialized as a long vector

// TO-DO:

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>			// time as random seed in create_NN()
#include <stdbool.h>
#include <fftw3.h>			// fastest Fourier Transform in the West
// #include "feedforward-NN.h"

#define Eta 0.01			// learning rate
#define BIASOUTPUT 1.0		// output for bias. It's always 1.

#define numLayers		4
#define neuronsPerLayer	[4, 3, 3, 2]
#define populationSize	100
#define MaxGens			100
#define CrossRate		0.98
#define MutationRate	(1.0 / N)

double fitness[populationSize];

// Sorry I have to use global variables to simplify code
// =============================================================

// A question is how to store the current network as well as the entire population.
// Perhaps the data structure should store all the "population rows".
double population[L][M][N];		// each element is a connection weight
double output[L][M];			// output of each neuron
double grad[L][M];				// local gradient for each neuron
double score[L][M];				// fitness of each neuron

double best[L][N][N];			// best candidate
double selected[L][M][N];		// selected from binary tournament
double children[L][M][N];		// 2nd generation

int neuronsPerLayer[L] = { N };		// initialize all layers to have N neurons
int dimK = N;						// dimension of input-layer vector

extern int rand(void);
extern void qsort(void *, size_t, size_t, int (*comparator)(const void *, const void*));
extern void forward_gNN(int, double []);		// forward-propagate the gNN
extern void backprop_gNN(double []);

double fitness(int layer, int index)
// Call forward-prop with input-output pairs to evaluate the current network.
	{
	// initialize network
	double K[dimK];
	double errors[dimK];

	// forward_prop
	#define NumTrials	100
	double sum_fitness = 0.0;
	for (int i = 0; i < NumTrials; ++i)
		{
		// prepare input and ideal output values
		// Create random K vector (4 + 2 + 2 elements)
		for (int k = 0; k < 4; ++k)
			K[k] = floor((rand() / (double) RAND_MAX) * 10.0) / 10.0;
		for (int k = 4; k < 6; ++k)
			K[k] = (rand() / (double) RAND_MAX) > 0.5 ? 1.0 : 0.0;
		for (int k = 6; k < 8; ++k)
			K[k] = floor((rand() / (double) RAND_MAX) * 10.0) / 10.0;

		// Desired value = K_star
		double K_star[10];
		extern void transition(double [], double []);
		transition(K, K_star);

		forward_gNN(dimK, K);		// forward-propagate the gNN

		// Calculate the error, for back-prop
		for (int k = 0; k < dimK; ++k)
			errors[k] = K_star[k] - K[k];	// error = ideal - actual

		// Use back-prop to calculate local gradients
		backprop_gNN(errors);
		// Then fitness = sum of local gradients for a neuron, relative to 1 example.
		double fitness = 0.0;
		for (int n = 0; n < N; ++n)
			{
			double g = grad[layer][n];
			fitness += g * g;
			}
		// And we need to add up the fitnesses for all examples.
		sum_fitness -= fitness;
		}
	}

// This seems to be independent of gene expression
// INPUT: population
// OUTPUT: selected = the winner (an individual = a neuron)
void binaryTournament(int layer, int candidate)
	{
	// Choose 2 candidates (neurons) in the population
	int i = (rand() / (double) RAND_MAX) * M;
	int j = (rand() / (double) RAND_MAX) * M;

	double *p1 = population[layer][i], *p2 = population[layer][j];

	double *p;
	if (fitness(layer, i) > fitness(layer, j))
		p = p1;
	else
		p = p2;

	for (int n = 0; n < N; ++n)
		selected[layer][candidate][n] = p[n];
	}

// Each neuron is an individual, a point mutation mutates a single weight within the neuron
void pointMutation(double *dna, double rate)
	{
	for (int n = 0; n < N; ++n)
		if ((rand() / (double) RAND_MAX) < rate)
			dna[n] = (dna[n] == '0') ? '1' : '0';
	}

// Cross-over of 2 neurons
void crossOver(double *result, double *parent1, double *parent2, double rate)
	{
	if ((rand() / (double) RAND_MAX) > rate)
		{
		for (int n = 0; n < N; ++n)
			result[n] = parent1[n];
		return;
		}

	int point = (rand() / (double) RAND_MAX) * N;
	int n;
	for (n = 0; n < point; ++n)
		result[n] = parent1[n];
	for (; n < N; ++n)
		result[n] = parent2[n];
	}

// **** Reproduce for 1 generation
void reproduce(int layer, int popSize, double crossRate, double mutationRate)
	{
	double *p1, *p2;

	for (int m = 0; m < M; ++m)
		{
		p1 = selected[layer][m];
		p2 = (m % 2 == 0) ? selected[layer][m + 1] : selected[layer][m - 1];
		if (m == M - 1)
			p2 = selected[layer][0];

		crossOver(children[layer][m], p1, p2, crossRate);
		pointMutation(children[layer][m], mutationRate);
		}
	}

void printCandidate(double candidate[N])
	{
	for (int n = 0; n < N; ++n)
		printf("%c", (candidate[n] == '0') ? ' ' : '*');
	printf("\n");
	}

void FourierTransform(int n, double *network, fftw_complex *gene)
	{
    fftw_plan p;

	p = fftw_plan_dft_r2c_1d(n, network, gene);
    fftw_execute(p);
    fftw_destroy_plan(p);
	}

// Main algorithm for genetic search
void evolve()
	{
	// **** initialize population

	// find total # of neurons per NN
	int numNeurons = 0;
	for (int l = 0; l < numLayers; ++l)						// for each layer
		numNeurons += neuronsPerLayer[l];

	// allocate space for one NN
	double *candidate = malloc(numNeurons * sizeof(double));
	// allocate space for entire genome
	fftw_complex genome[populationSize][] = fftw_malloc(sizeof(fftw_complex) * numNeurons * populationSize);

	// generate population of NNs with random weights
	for (int m = 0; m < populationSize; ++m)				// for each population candidate
		{
		for (int i = 0, l = 0; l < numLayers; ++l)			// for each layer
			for (int n = 0; n < neuronsPerLayer[l]; ++n)	// for each neuron
				{
				++i;
				candidate[i] = (rand() / (double) RAND_MAX) * 2.0 - 1.0;	// w ∊ [-1,1]
				}
		// do Fourier transform
		FourierTransform(numNeurons, candidate, genome[m]);
		}

	// Compare fitness of 2 candidates
	int compareFitness(const void *l, const void *r)
		{
		int x_index = *(const int *)l;
		int y_index = *(const int *)r;
		return (fitness(x_index) < fitness(y_index));
		};

	for (int m = 0; m < populationSize; ++m)		// for each candidate
		fitness[m] = evaluateCandidate(genome[m]);

	// Sort population according to fitness
	qsort(genome, populationSize, numNeurons * sizeof(fftw_complex), compareFitness);

	printf("Initial population:\n");
	for (int m = 0; m < populationSize; ++m)
		printCandidate(genome[m]);

	for (int i = 0; i < MaxGens; ++i)
		{
		printf("gen %03d: \n", i);

		for (int m = 0; m < populationSize; ++m)	// for the size of 1 population
			binaryTournament();

		qsort(selected, populationSize, numNeurons * sizeof(fftw_complex), compareFitness);

		reproduce(l, M, CrossRate, MutationRate);

		qsort(children[l], M, N, compareFitness);

		for (int m = 0; m < populationSize; ++m)
			printCandidate(genome[m]);

		strncpy(population, children, sizeof(population));

		if (fitness(best, 0) >= N)
			{
			printf("Success!!!\n");
			break;
			}

		// Thread.sleep(500)
		getchar();
		}

	printf("Finished.\n");
	}


//**************************** forward-propagation ***************************//
void forward_gNN(int dim_V, double V[])
	{
	// extern double output[][];

	// set the output of input layer
	for (int n = 0; n < dim_V; ++n)
		output[0][n] = V[n];

	// calculate output from hidden layers to output layer
	for (int l = 1; l < L; l++)
		{
		for (int n = 0; n < N; n++)
			{
			double v = 0; //induced local field for neurons
			// calculate v, which is the sum of the product of input and weights
			for (int k = 0; k <= N; k++)
				{
				if (k == 0)
					v += population[l][n][k] * BIASOUTPUT;
				else
					v += population[l][n][k] *
						output[l - 1][k - 1];
				}

			output[l][n] = sigmoid(v);
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
	// calculate gradient for output layer
	for (int n = 0; n < N; ++n)
		{
		double out = output[L - 1][n];
		//for output layer, ∇ = y∙(1-y)∙error
		#define steepness 1.0
		grad[L - 1][n] = steepness * out * (1.0 - out) * errors[n];
		}

	// calculate gradient for hidden layers
	for (int l = L - 2; l > 0; --l)		// for each hidden layer
		{
		for (int n = 0; n < N; n++)		// for each neuron in layer
			{
			double out = output[l][n];
			double sum = 0.0f;
			// nextLayer = l + 1;
			for (int i = 0; i < N; i++)		// for each weight
				{
				sum += population[l + 1][i][n + 1]		// ignore weights[0] = bias
						* grad[l + 1][i];
				}
			grad[l][n] = steepness * out * (1.0 - out) * sum;
			}
		}

	// update all weights
	for (int l = 1; l < L; ++l)		// except for 0th layer which has no weights
		{
		for (int n = 0; n < N; n++)		// for each neuron
			{
			population[l][n][0] += Eta *
					grad[l][n] * 1.0;		// 1.0f = bias input
			for (int i = 0; i < N; i++)		// for each weight
				{
				double inputForThisNeuron = output[l - 1][i];
				population[l][n][i + 1] += Eta *
						grad[l][n] * inputForThisNeuron;
				}
			}
		}
	}

/*
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

*/

/* No need to create neural network as it is stored in the population
 *
//****************************create neural network*********************
// GIVEN: how many layers, and how many neurons in each layer
void create_gNN(NNET *net, int numLayers, int *neuronsPerLayer, double dna[][L])
	{
	srand(time(NULL));
	net->numLayers = numLayers;

	assert(numLayers >= 3);

	net->layers = (LAYER *) malloc(numLayers * sizeof (LAYER));
	//construct input layer, no weights
	net->layers[0].numNeurons = neuronsPerLayer[0];
	net->layers[0].neurons = (NEURON *) malloc(neuronsPerLayer[0] * sizeof (NEURON));

	//construct hidden layers
	for (int l = 1; l < numLayers; l++) //construct layers
		{
		net->layers[l].neurons = (NEURON *) malloc(neuronsPerLayer[l] * sizeof (NEURON));
		net->layers[l].numNeurons = neuronsPerLayer[l];
		for (int n = 0; n < neuronsPerLayer[l]; n++) // construct each neuron in the layer
			{
			net->layers[l].neurons[n].weights =
					(double *) malloc((neuronsPerLayer[l - 1] + 1) * sizeof (double));
			for (int i = 0; i <= neuronsPerLayer[l - 1]; i++)
				{
				//construct weights of neuron from previous layer neurons
				//when k = 0, it's bias weight
				net->layers[l].neurons[n].weights[i] = randomWeight();
				//net->layers[i].neurons[j].weights[k] = 0.0f;
				}
			}
		}
	}

void free_gNN(NNET *net, int *neuronsPerLayer)
	{
	// for input layer
	free(net->layers[0].neurons);

	// for each hidden layer
	int numLayers = net->numLayers;
	for (int l = 1; l < numLayers; l++) // for each layer
		{
		for (int n = 0; n < neuronsPerLayer[l]; n++) // for each neuron in the layer
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
