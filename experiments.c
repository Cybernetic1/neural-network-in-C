// **************** Newer Experiments *****************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "RNN.h"
#include "feedforward-NN.h"

extern void create_NN(NNET *, int, int *);
extern void create_RNN(RNN *, int, int *);
extern void free_RNN(RNN *, int *);
extern void free_NN(NNET *, int *);
extern void forward_prop(NNET *, int, double *);
extern void forward_prop_ReLU(NNET *, int, double *);
extern void forward_RNN(RNN *, int, double *);
extern void back_prop(NNET *);
extern void back_prop_ReLU(NNET *, double *);
extern void RTRL(RNN *, double *);
extern NNET *loadNet(int, int *);
extern void pause_graphics();
extern void quit_graphics();
extern void start_NN_plot(void);
extern void start_NN2_plot(void);
extern void start_W_plot(void);
extern void start_K_plot(void);
extern void start_output_plot(void);
extern void plot_NN(NNET *net);
extern void plot_NN2(NNET *net);
extern void plot_W(NNET *net);
extern void plot_output(NNET *net, void ());
extern void flush_output();
extern void plot_tester(double, double);
extern void plot_K();
extern int delay_vis(int);
extern void plot_trainer(double);
extern void plot_ideal(void);
extern void beep(void);
extern double sigmoid(double);
extern void start_timer(), end_timer();

extern double K[];

// Try to learn input-output pairs with flexible iteration

// Strategy: let unfold N times until convergence, then back-prop through all unfoldings
// Require: back-prop with ReLU's
// forward-propagate and perhaps no need to record until convergence

// From now on we adopt a simple architecture:  The RNN is a multi-layer feed-forward
// network, with its output layer connected to its input layer.  For learning, we simply
// use traditional back-prop, the key is by allowing the network to iterate as long as
// it needs to converge to an equilibrium point, and then we use the difference between
// the equilibrium point and the target as error.

void RTRL_equilibrium_test()
	{
	// create RNN
	RNN *Net = (RNN *) malloc(sizeof (RNN));
	int neuronsPerLayer[4] = {3, 4, 4, 3}; // first = input layer, last = output layer
	int numLayers = sizeof(neuronsPerLayer) / sizeof(int);
	create_RTRL_NN(Net, numLayers, neuronsPerLayer);
	rLAYER lastLayer = Net->layers[numLayers - 1];

	int dimK = 3;
	double K2[dimK];
	double errors[dimK];
	int quit;
	double sum_error2;

	// Create random input-output pairs as target training set
	#define DataSize 5
	double K_star[DataSize][2][dim_K]; // second index: 0 = input, 1 = output
	for (int i = 0; i < DataSize; ++i)
		for (int k = 0; k < dim_K; ++k)
			{
			K_star[i][0][k] = (rand() / (float) RAND_MAX); // random in [0,1]
			K_star[i][1][k] = (rand() / (float) RAND_MAX); // random in [0,1]
			}

	printf("RTRL equilibrium learning test\n");
	printf("Press 'Q' to quit\n\n");
	start_NN_plot();
	start_W_plot();
	start_K_plot();

	// For each (outer) iteration, allow network to converge to equilibrium
	// Then apply back-prop once to train network

	for (int i = 0; 1; ++i)
		{
		for (int k = 0; k < dimK; ++k) // initialize K
			K[k] = K_star[i % DataSize][0][k];

		#define MaxIterations 100
		for (int j = 0; j < MaxIterations; j++) // allow network to converge
			{
			forward_RTRL(Net, dimK, K);

			// Check if convergence has reached
			// Difference between last output and current output:
			double diff = 0.0f;
			for (int k = 0; k < dimK; ++k)
				diff += fabs(lastLayer.neurons[k].output - K[k]);
			if (diff < 0.001)
				break;

			// If not, copy output to input, and re-iterate
			for (int k = 0; k < dimK; ++k)
				K[k] = lastLayer.neurons[k].output;
			}

		// When we have reached here, network has either converged or is chaotic
		// We apply to back-prop to train the network
		errors[0] = 0.0f;

		// The rest of the errors are zero:
		for (int k = 1; k < dimK; ++k)
			errors[k] = 0.0f;

		RTRL(Net, errors);

		// copy output to input
		for (int k = 0; k < dimK; ++k)
			K[k] = lastLayer.neurons[k].output;

		// sum_error2 += (error * error); // record sum of squared errors

		// plot_W(Net);
		// plot_NN(Net);
		// plot_trainer(K_star);
		plot_K();
		if (quit = delay_vis(0))
			break;

		printf("iteration: %05d, error: %lf\n", i, sum_error2);
		if (isnan(sum_error2))
			break;
		if (sum_error2 < 0.01)
			break;
		if (quit)
			break;
		}

	if (!quit)
		pause_graphics();
	free_RTRL_NN(Net, neuronsPerLayer);
	}
