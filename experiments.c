// **************** Newer Experiments *****************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "RNN.h"
#include "feedforwardNN.h"

extern void create_NN(NNET *, int, int *);
extern void create_RNN(RNN *, int, int *);
extern void free_RNN(RNN *, int *);
extern void free_NN(NNET *, int *);
extern void forward_prop(NNET *, int, double *);
extern void forward_prop_ReLU(NNET *, int, double *);
extern void forward_RNN(RNN *, int, double *);
extern void back_prop(NNET *);
extern void back_prop_ReLU(NNET *);
extern void RTRL(RNN *);
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

void RNN_equilibrium_test()
	{
	// create RNN
	RNN *Net = (RNN *) malloc(sizeof (RNN));
	int neuronsOfLayer[4] = {3, 4, 4, 3}; // first = input layer, last = output layer
	int numLayers = sizeof(neuronsOfLayer) / sizeof(int);
	create_RNN(Net, numLayers, neuronsOfLayer);
	rLAYER lastLayer = Net->layers[numLayers - 1];

	int dimK = 3;
	double K2[dimK];
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

	printf("RNN equilibrium learning test\n");
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
			forward_RNN(Net, dimK, K);

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
		lastLayer.neurons[0].error = 0.0f;

		// The rest of the errors are zero:
		for (int k = 1; k < dimK; ++k)
			lastLayer.neurons[k].error = 0.0f;

		RTRL(Net);

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
	free_RNN(Net, neuronsOfLayer);
	}


void classic_BP_test_ReLU2()
	{
	int neuronsOfLayer[] = {2, 10, 10, 1}; // first = input layer, last = output layer
	int numLayers = sizeof (neuronsOfLayer) / sizeof (int);
	NNET *Net = (NNET *) malloc(sizeof (NNET));
	create_NN(Net, numLayers, neuronsOfLayer);
	LAYER lastLayer = Net->layers[numLayers - 1];

	int quit = 0;
	#define M	50			// how many errors to record for averaging
	double errors1[M], errors2[M]; // two arrays for recording errors
	double sum_err1 = 0.0, sum_err2 = 0.0; // sums of errors
	int tail = 0; // index for cyclic arrays (last-in, first-out)

	for (int i = 0; i < M; ++i) // clear errors to 0.0
		errors1[i] = errors2[i] = 0.0;

	start_NN_plot();
	start_W_plot();
	// start_K_plot();
	start_output_plot();
	// plot_ideal();
	printf("BP ReLU test.\nPress 'Q' to quit\n\n");

	for (int i = 1; 1; ++i)
		{
		// Create random K vector
		for (int k = 0; k < 2; ++k)
			K[k] = (rand() / (float) RAND_MAX);
		// printf("*** K = <%lf, %lf>\n", K[0], K[1]);

		forward_prop_ReLU(Net, 2, K); // dim K = 2

		// Desired value = K_star
		double training_err = 0.0;
		for (int k = 0; k < 1; ++k) // output has only 1 component
			{
			// double ideal = K[k];				/* identity function */
			#define f2b(x) (x > 0.5f ? 1 : 0)	// convert float to binary
			// ^ = binary XOR
			double ideal = ((double) (f2b(K[0]) ^ f2b(K[1]))); // ^ f2b(K[2]) ^ f2b(K[3])))

			// Difference between actual outcome and desired value:
			double error = ideal - lastLayer.neurons[k].output;
			lastLayer.neurons[k].error = error; // record this for back-prop

			training_err += fabs(error); // record sum of errors
			// printf("training error = %lf \n", training_err);
			}
		// printf("sum of squared error = %lf  ", training_err);

		// update error arrays cyclically
		// (This is easier to understand by referring to the next block of code)
		sum_err2 -= errors2[tail];
		sum_err2 += errors1[tail];
		sum_err1 -= errors1[tail];
		sum_err1 += training_err;
		// printf("sum1, sum2 = %lf %lf\n", sum_err1, sum_err2);

		double mean_err = (i < M) ? (sum_err1 / i) : (sum_err1 / M);

		// record new error in cyclic arrays
		errors2[tail] = errors1[tail];
		errors1[tail] = training_err;
		++tail;
		if (tail == M) // loop back in cycle
			tail = 0;

		back_prop_ReLU(Net);

		double ratio = (sum_err2 - sum_err1) / sum_err1;

		if ((i % 500) == 0) // display status periodically
			{
			printf("iteration: %05d: ", i);
			printf("mean error = %.03lf  ", mean_err);
			if (ratio > 0)
				printf("error ratio = %.03f \t", ratio);
			else
				printf("error ratio = \x1b[31m%.03f\x1b[39;49m\t", ratio);

			plot_NN(Net);
			plot_W(Net);
			plot_output(Net, forward_prop_ReLU); // note: this function calls forward_prop!
			flush_output();
			// plot_trainer(0);		// required to clear the window
			// plot_K();
			if (quit = delay_vis(0))
				break;
			}

		if ((i % 500) == 0)
			{
			// Testing set
			double test_err = 0.0;
			#define numTests 100
			for (int j = 0; j < numTests; ++j)
				{
				// Create random K vector
				for (int k = 0; k < 2; ++k)
					K[k] = ((double) rand() / (double) RAND_MAX);
				// plot_tester(K[0], K[1]);

				forward_prop_ReLU(Net, 2, K);

				// Desired value = K_star
				double single_err = 0.0;
				for (int k = 0; k < 1; ++k)
					{
					// double ideal = 1.0f - (0.5f - K[0]) * (0.5f - K[1]);
					double ideal = (double) (f2b(K[0]) ^ f2b(K[1]));
					// double ideal = K[k];				/* identity function */

					// Difference between actual outcome and desired value:
					double error = ideal - lastLayer.neurons[k].output;

					single_err += fabs(error); // record sum of errors
					}
				test_err += single_err;
				}
			test_err /= ((double) numTests);
			printf("random test error = %.03lf \n", test_err);

			if (test_err < 0.05)
				break;
			}

		if (isnan(ratio) && i > 10)
			break;
		// if (ratio - 0.5f < 0.0000001)	// ratio == 0.5 means stationary
		// if (test_err < 0.01)
		}

	beep();
	plot_output(Net, forward_prop_ReLU);
	flush_output();
	plot_W(Net);

	if (!quit)
		pause_graphics();
	else
		quit_graphics();
	free_NN(Net, neuronsOfLayer);
	}
