#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "RNN.h"

extern void create_NN(NNET *, int, int *);
extern void forward_prop(NNET *, int, double *);
extern void back_prop(NNET *);

#define LastLayer (Net->layers[numLayers - 1])

extern NNET *Net;

// Randomly generate an RNN, watch it operate on K and see how K moves
void K_wandering_test()
	{
	Net = (NNET *) malloc(sizeof (NNET));
	int numLayers = 3;
	int neuronsOfLayer[3] = {10, 13, 10}; // first = input layer, last = output layer
	create_NN(Net, numLayers, neuronsOfLayer);
	double K2[dim_K];

	for (int j = 0; j < 10000; j++) // max number of iterations
		{
		forward_prop(Net, dim_K, K);

		printf("%02d", j);
		double d = 0.0;

		// copy output to input
		for (int k = 0; k < dim_K; ++k)
			{
			K2[k] = K[k];
			K[k] = LastLayer.neurons[k].output;
			printf(", %0.4lf", K[k]);
			double diff = (K2[k] - K[k]);
			d += (diff * diff);
			}
		printf("\n");
		if (d < 0.000001)
			{
			fprintf(stderr, "terminated after %d cycles,\t delta = %lf\n", j, d);
			break;
			}
		}
	free(Net);
	}

// Train RNN to reproduce a sine wave time-series
// Train the 0-th component of K to move as sine wave
void sine_wave_test()
	{
	Net = (NNET *) malloc(sizeof (NNET));
	int numLayers = 3;
	int neuronsOfLayer[3] = {10, 13, 10}; // first = input layer, last = output layer
	create_NN(Net, numLayers, neuronsOfLayer);
	double K2[dim_K];

	#define Pi 3.141592654f
	
	double sum_error2;
	
	// Initialize K vector
	for (int k = 0; k < dim_K; ++k)
		K[k] = (rand() / (float) RAND_MAX) * 2.0 - 1.0;
	
	for (int i = 0; i < 50; ++i)
		{
		sum_error2 = 0.0f;

		#define N 30		// loop from 0 to 2π in N divisions
		for (int j = 0; j < N; j++) 
			{
			forward_prop(Net, dim_K, K);

			// The difference between K[0] and K'[0] should be equal to [sin(θ+dθ) - sinθ]
			// where θ = 2π j/60.
			double dK_star = ( sin(2*Pi * (j+1) / N) - sin(2*Pi * j / N) );

			// Calculate actual difference between K[0] and K'[0]:
			double dK = LastLayer.neurons[0].output - K[0];

			// The error is the difference between the above two values:
			double error = dK_star - dK;

			// Error in the back-prop NN is recorded as [ideal - actual]:
			//		K* - K = dK*
			//		K' - K = dK
			// thus, K* - k' = dK* - dK 
			LastLayer.neurons[0].error = error;

			// The rest of the errors are zero:
			for (int k = 1; k < dim_K; ++k)
				LastLayer.neurons[k].error = 0.0f;

			back_prop(Net);
			
			// copy output to input
			for (int k = 0; k < dim_K; ++k)
				K[k] = LastLayer.neurons[k].output;

			sum_error2 += (error * error);		// record sum of squared errors
			}

		printf("iteration: %05d, error: %lf\n", i, sum_error2);

		if (sum_error2 < 0.01)
			break;
		}
	free(Net);
	}
