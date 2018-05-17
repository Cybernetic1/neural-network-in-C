// YKY's idea of stochastic forward-backward search, for training a recurrent network

// Stragegy: stochastic forward-backward and record noise, then back-prop to bridge gaps
// Require: simple back-prop

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <time.h>		// time as random seed in create_NN()
#include "BPTT-RNN.h"

#define Eta 0.001			// learning rate
#define BIASOUTPUT 1.0		// output for bias. It's always 1.

// At this point there is evidence that rectified back-prop can be an "okay" function
// approximator.

// Algorithm:
// given: input, desired output
// 1) forward-propagate input a number of iterations
// 2) backward-propagate output a number of iterations
// 3) for each pair of frontiers, check if they are close enough (within a fixed threshold)
// 4) if yes, use back-prop thru-time to train
// 5) there could be multiple matches

// Testing Algorithm:
// 1) present input-output pairs (?)
// 2) let learn
// 3) let test

// Cf: BPTT_arithmetic_test()

void relations_test()
	{
	extern void forward_BPTT(RNN *, int, double [], int);
	extern void backprop_through_time(RNN *, double *, int);
	#define ForwardPropMethod	forward_BPTT
	#define BackPropMethod		backprop_through_time

	int dimK = 2;
	double K[dimK];
	int neuronsPerLayer[] = {dimK, 5, dimK}; // first = input layer, last = output layer
	RNN *Net = (RNN *) malloc(sizeof (RNN));
	int numLayers = sizeof (neuronsPerLayer) / sizeof (int);
	create_BPTT_NN(Net, numLayers, neuronsPerLayer);
	rLAYER lastLayer = Net->layers[numLayers - 1];
	double errors[dimK];

	int quit = 0;
	#define M	50			// how many errors to record for averaging
	double errors1[M], errors2[M]; // two arrays for recording errors
	double sum_err1 = 0.0, sum_err2 = 0.0; // sums of errors
	int tail = 0; // index for cyclic arrays (last-in, first-out)

	for (int i = 0; i < M; ++i) // clear errors to 0.0
		errors1[i] = errors2[i] = 0.0;

	// start_NN_plot();
	start_W_plot();
	// start_K_plot();
	start_output_plot();
	start_LogErr_plot();
	// plot_ideal();
	printf("Stochastic forward-backward test.\nPress 'Q' to quit\n\n");
	start_timer();

	#define ErrorThreshold 0.01

	// Perhaps create random boolean function as target
	bool signiture1[dimK * dimK - dimK];
	bool signiture2[dimK * dimK];
	for (int k = 0; k < dimK * dimK - dimK; ++k)
		signiture1[k] = (rand() / (float) RAND_MAX) > 0.5 ? true : false;
	for (int k = 0; k < dimK * dimK; ++k)
		signiture2[k] = (rand() / (float) RAND_MAX) > 0.5 ? true : false;

	// target function
	void target(bool in_vec[], bool out_vec[], bool sig1[], bool sig2[])
		{
		for (int k1 = 0; k1 < dimK; ++k1)
			{
			bool result = sig2[k1 * dimK] ? in_vec[0] : !in_vec[0];
			for (int k2 = 1; k2 < dimK; ++k2)
				{
				bool input = sig2[k1 * dimK + k2] ? in_vec[k2] : !in_vec[k2];
				result = sig1[k1 * dimK + k2 - 1] ?
					result & input :
					result | input;
				}
			out_vec[k1] = result;
			}
		}

	char str[200], *s;
	for (int i = 0; true; ++i)
		{
		s = str + sprintf(str, "iteration: %05d: ", i);

		// Create random K vector
		bool K_in[dimK], K_out[dimK];
		for (int k = 0; k < dimK; ++k)
			{
			K[k] = (rand() / (float) RAND_MAX);
			#define f2b(x) (x > 0.5f ? 1 : 0)	// convert float to binary
			K_in[k] = f2b(K[k]);
			}

		ForwardPropMethod(Net, dimK, K, 2); // iterations = 2-fold

		// **** Calculate error
		double training_err = 0.0;
		target(K_in, K_out, signiture1, signiture2);
		for (int k = 0; k < dimK; ++k)
			{
			// Desired value
			double ideal = K_out[k];
			// #define Ideal ((double) (f2b(K[k]) ^ f2b(K[2]) ^ f2b(K[3])))
			// double ideal = 1.0f - (0.5f - K[0]) * (0.5f - K[1]);
			// printf("*** ideal = %lf\n", ideal);

			// Difference between actual outcome and desired value:
			int t = 0;
			double error = ideal - lastLayer.neurons[k].output[t];
			errors[k] = error; // record this for back-prop

			training_err += fabs(error); // record sum of errors
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
		s += sprintf(s, "mean error = %lf  ", mean_err);

		// record new error in cyclic arrays
		errors2[tail] = errors1[tail];
		errors1[tail] = training_err;
		++tail;
		if (tail == M) // loop back in cycle
			tail = 0;

		// plot_W(Net);
		BackPropMethod(Net, errors, 2);		// 2-fold
		// plot_W(Net);
		// pause_graphics();
		
		if ((i % 200) == 0)
			{
			// Testing set
			double test_err = 0.0;
			#define numTests 50
			for (int j = 0; j < numTests; ++j)
				{
				// Create random K vector
				for (int k = 0; k < 2; ++k)
					K[k] = ((double) rand() / (double) RAND_MAX);
				// plot_tester(K[0], K[1]);

				ForwardPropMethod(Net, 2, K, 2);

				// Desired value = K_star
				double single_err = 0.0;
				for (int k = 0; k < 1; ++k)
					{
					// double ideal = 1.0f - (0.5f - K[0]) * (0.5f - K[1]);
					double ideal = (double) (f2b(K[0]) ^ f2b(K[1]));
					// double ideal = K[k];				/* identity function */

					// Difference between actual outcome and desired value:
					int t = 0;
					double error = ideal - lastLayer.neurons[k].output[t];

					single_err += fabs(error); // record sum of errors
					}
				test_err += single_err;
				}
			test_err /= ((double) numTests);
			s += sprintf(s, "random test error = %1.06lf  ", test_err);

			plot_LogErr(test_err, ErrorThreshold);

			if (test_err < ErrorThreshold)
				break;
			}

		if ((i % 200) == 0)
			{
			double ratio = (sum_err2 - sum_err1) / sum_err1;
			if (ratio > 0)
				s += sprintf(s, "error ratio = %f\r", ratio);
			else
				s += sprintf(s, "error ratio = \x1b[31m%f\x1b[39;49m\r", ratio);
			printf(str);
			if (isnan(ratio))
				break;
			}

		if ((i % 200) == 0) // display status periodically
			{
			// plot_NN(Net);
			plot_W(Net);
			plot_output(Net, ForwardPropMethod);
			flush_output();
			// plot_trainer(0);		// required to clear the window
			// plot_K();
			if (quit = delay_vis(0))
				break;
			}
		}

	end_timer(NULL);
	beep();
	plot_output(Net, ForwardPropMethod);
	flush_output();
	plot_W(Net);

	if (!quit)
		pause_graphics();
	else
		quit_graphics();
	free_NN(Net, neuronsPerLayer);
	}
