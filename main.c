#include <stdio.h>
#include <stdlib.h>
#include <SDL2/SDL.h>

#include "RNN.h"

#define dim_K 10			// dimension of cognitive state vector K
double K[dim_K];

extern void create_NN(NNET *, int, int *);
extern void Q_learn(double *, double *, double, double);
extern void Q_act(double *, double *);
extern void forward_prop(NNET *, int, double *);
extern double calc_error(NNET *, double []);
extern void back_prop(NNET *);
extern void drawNetwork(NNET *);
extern void pause_graphics();
extern void init_graphics();

//************************** training data ***********************//
// Each entry of training data consists of a K input value and a desired K
// output value.

#define DATASIZE 100	// number of training / testing examples

double trainingIN[DATASIZE][dim_K];
double trainingOUT[DATASIZE][dim_K];

double testingIN[DATASIZE][dim_K];
double testingOUT[DATASIZE][dim_K];

void read_trainers()
	{
	//open training file
	FILE *fp1, *fp2, *fp3, *fp4;

	if ((fp1 = fopen("training-set-in.txt", "r")) == NULL)
		{
		fprintf(stderr, "Cannot open training-set-in.\n");
		exit(1);
		}
	if ((fp2 = fopen("training-set-out.txt", "r")) == NULL)
		{
		fprintf(stderr, "Cannot open training-set-out.\n");
		exit(1);
		}
	if ((fp3 = fopen("testing-set-in.txt", "r")) == NULL)
		{
		fprintf(stderr, "Cannot open testing-set-in.\n");
		exit(1);
		}
	if ((fp4 = fopen("testing-set-out.txt", "r")) == NULL)
		{
		fprintf(stderr, "Cannot open testing-set-out.\n");
		exit(1);
		}

	for (int i = 0; i < DATASIZE; ++i)
		for (int j = 0; j < dim_K; ++j)
			{
			fscanf(fp1, "%lf", &trainingIN[i][j]);
			fscanf(fp2, "%lf", &trainingOUT[i][j]);
			fscanf(fp3, "%lf", &testingIN[i][j]);
			fscanf(fp4, "%lf", &testingOUT[i][j]);
			}

	fclose(fp1);
	fclose(fp2);
	fclose(fp3);
	fclose(fp4);
	}

//************************** get rewards ***************************//

// Check if "output" bit of K is set;  If yes, the "output word" inside K is printed.
// Users may reward Genifer's response but there is the problem of *delays*.  At this 
// stage, to simplify things, we pause processing for users to give *immediate* response.
// If no output word is printed, reward would be 0 (which defaults to a discount cost).
double get_reward(double K[])
	{
	double R = -0.01f;				// default value
	
	if (0)	// Genifer outputs word?
		// Wait for user / supervisor response
		;

	return R;
	}
			
//************************** main algorithm ***********************//
// Main loop:
// 	----- RNN part -----
//	Input is copied into K.
//	Desired output is K*.
//	Do forward propagation (recurrently) a few times.
//	Output is K'.  Error is K'-K*.
//	Use back-prop to reduce this error.
//	----- RL part -----
//	Use Q value to choose an optimal action, taking K' to K''.
//	Invoke Q-learning, using the reward to update Q
// Repeat

NNET *Net;

#define LastLayer (Net->layers[numLayers - 1])

void main_loop()
	{
	Net = (NNET *) malloc(sizeof (NNET));
	int numLayers = 4;
	//the first layer -- input layer
	//the last layer -- output layer
	// int neuronsOfLayer[5] = {2, 3, 4, 4, 4};
	int neuronsOfLayer[4] = {10, 14, 13, 10};

	//read training data and testing data from file
	read_trainers();

	//create neural network for backpropagation
	create_NN(Net, numLayers, neuronsOfLayer);

	//error array to keep track of errors
	#define MAX_EPOCHS 30
	double error[MAX_EPOCHS];
	int maxlen = 0;
	int epoch = 1;

	init_graphics();

	//output data to a file
	FILE *fout;
	if ((fout = fopen("randomtest-1.txt", "w")) == NULL)
		{
		fprintf(stderr, "file open failed.\n");
		exit(1);
		}

	double *K2 = (double *) malloc(sizeof (double) * dim_K);

	do // Loop over all epochs
		{
		// double squareErrorSum = 0;

		// ----- RNN part -----

		// Loop over all training data
		for (int i = 0; i < DATASIZE; ++i)
			{
			// Write input value to K
			for (int k = 0; k < dim_K; ++k)
				K[k] = trainingIN[i][k];

			// Let RNN act on K n times (TO-DO: is this really meaningful?)
			#define Recurrence 10
			for (int j = 0; j < Recurrence; j++)
				{
				forward_prop(Net, dim_K, K);

				calc_error(Net, trainingOUT[i]);
				back_prop(Net);

				// copy output to input
				for (int k = 0; k < dim_K; ++k)
					K[k] = LastLayer.neurons[k].output;
				}
			}

		// ----- RL part -----

		// Use Q value to choose an optimal action, taking K to K2.
		Q_act(K, K2); // this changes K2

		// Invoke Q-learning, using the reward to update Q
		double R = get_reward(K2);	// reward is gotten from the state transition
		double oldQ = 0.0;			// ? TO-DO
		Q_learn(K, K2, R, oldQ);

		// ------ calculate error -------

		// error[maxlen] = sqrt(squareErrorSum / DATASIZE);
		printf("%03d", epoch);
		for (int i = 0; i < dim_K; ++i)
			printf(", %lf", K[i]);
		printf("\n");
		// fprintf(fout, "%d", epoch);
		maxlen++;
		epoch++;

		drawNetwork(Net);
		SDL_Delay(1000 /* milliseconds */);

		}
	while (maxlen < MAX_EPOCHS);

	fclose(fout);
	free(Net);
	extern NNET *Qnet;
	free(Qnet);
	free(K2);

	pause_graphics();	//keep the window open
	}


//************************** Genifer main function ***********************//

int main(int argc, char** argv)
	{
	printf("*** Welcome to Genifer 5.3 ***\n\n");

	// extern void K_wandering_test();
	// K_wandering_test();
	
	extern void sine_wave_test();
	sine_wave_test();

	return 0;
	}
