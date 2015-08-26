#include "RNN.h"

#define LastLayer (Net->layers[numLayers - 1])

// Randomly generate an RNN, watch it operate on K and see how K moves

void test_K_wandering() {
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
		for (int k = 0; k < dim_K; ++k) {
			K2[k] = K[k];
			K[k] = LastLayer.neurons[k].output;
			printf(", %0.4lf", K[k]);
			double diff = (K2[k] - K[k]);
			d += (diff * diff);
		}
		printf("\n");
		if (d < 0.000001) {
			fprintf(stderr, "terminated after %d cycles,\t delta = %lf\n", j, d);
			break;
		}
	}
}

