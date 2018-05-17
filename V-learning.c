// V-learner -- simple value learning in reinforcement learning
// for use with "tic-tac-toe.cpp"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "feedforward-NN.h"

extern double sigmoid(double v);
extern double randomWeight();
extern NNET *create_NN(int numberOfLayers, int *neuronsPerLayer);
extern void forward_prop_sigmoid(NNET *, int, double *);
extern double calc_error(NNET *net, double *Y);
extern void back_prop(NNET *, double *errors);

//************************** prepare Q-net ***********************//
NNET *Vnet;

int VnumLayers = 5;
int VneuronsPerLayer[] = {9, 40, 30, 20, 1};		// success

void init_Vnet()
	{
	//the first layer -- input layer
	//the last layer -- output layer
	// int neuronsPerLayer[5] = {2, 3, 4, 4, 4};
	// int neuronsPerLayer[5] = {18, 18, 15, 10, 1};
	Vnet = (NNET*) malloc(sizeof (NNET));
	//create neural network for backpropagation
	Vnet = create_NN(VnumLayers, VneuronsPerLayer);

	// return Vnet;
	}

void load_Vnet()
	{
	int numLayers2;
	int *neuronsPerLayer2;
	extern NNET * loadNet(int *, int *p[], char *);
	Vnet = loadNet(&numLayers2, &neuronsPerLayer2, "v.net");
	// LAYER lastLayer = Vnet->layers[numLayers - 1];

	return;
	}

void save_Vnet(char *fname)
	{
	extern void saveNet(NNET *, int, int *, char *, char *);

	saveNet(Vnet, VnumLayers, VneuronsPerLayer, "", fname);
	}

// **** Learn a simple V-value map given specific V values

void train_V(int s[9], double V)
	{
	double S[9];

	for (int j = 0; j < 3; ++j)
		{
		for (int k = 0; k < 9; ++k)
			S[k] = (double) s[k];

		forward_prop_sigmoid(Vnet, 9, S);

		int numLayers = 5;
		LAYER LastLayer = (Vnet->layers[numLayers - 1]);
		// The last layer has only 1 neuron, which outputs the Q value:
		double V2 = LastLayer.neurons[0].output;

		double error[1];
		*error = V - V2; // desired - actual

		back_prop(Vnet, error);
		}
	}

// **** Learn a simple V-value map via backprop and Bellman update

void learn_V(int s2[9], int s[9])
	{
	double S2[9], S[9];

	for (int j = 0; j < 4; ++j)
		{

		for (int k = 0; k < 9; ++k)
			{
			S2[k] = (double) s2[k];
			S[k] = (double) s[k];
			}

		forward_prop_sigmoid(Vnet, 9, S2);

		int numLayers = 5;
		LAYER LastLayer = (Vnet->layers[numLayers - 1]);
		// The last layer has only 1 neuron, which outputs the Q value:
		double V2 = LastLayer.neurons[0].output;

		forward_prop_sigmoid(Vnet, 9, S);
		double V = LastLayer.neurons[0].output;

		double error[1];
		*error = V2 - V;

		back_prop(Vnet, error);
		}
	}

// Get V-value by forward propagation

double get_V(int x[9])
	{
	double X[9];
	for (int k = 0; k < 9; ++k)
		X[k] = (double) x[k];

	forward_prop_sigmoid(Vnet, 9, X);

	int numLayers = 5;
	LAYER LastLayer = (Vnet->layers[numLayers - 1]);
	// The last layer has only 1 neuron, which outputs the Q value:
	return LastLayer.neurons[0].output;
	}
