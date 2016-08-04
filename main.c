#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <SDL2/SDL.h>

#include "feedforward-NN.h"

#define dim_K 10			// dimension of cognitive state vector K
double K[dim_K];

extern void create_NN(NNET *, int, int *);
extern void Q_learn(double *, double *, double, double);
extern void Q_act(double *, double *);
extern void forward_prop_sigmoid(NNET *, int, double *);
extern double calc_error(NNET *, double []);
extern void back_prop(NNET *);
extern void plot_NN(NNET *net);
extern void pause_graphics();
extern void start_NN_plot();
extern void start_K_plot();
extern void beep();

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
	double R = -0.01f; // default value

	if (0) // Genifer outputs word?
		// Wait for user / supervisor response
		;

	return R;
	}

double state[9] = {		// state of Tic-Tac-Toe
	0.0, 0.0, 0.0,
	0.0, 0.0, 0.0,
	0.0, 0.0, 0.0
	};

bool won(double who)		// check if player has won
	{
	// total of 8 cases:
	// 0 1 2
	// 3 4 5
	// 6 7 8

	#define TAKEN(x)		((x - who) < 0.01)

	if (TAKEN(state[0]) && TAKEN(state[1]) && TAKEN(state[2]))
		return true;
	if (TAKEN(state[3]) && TAKEN(state[4]) && TAKEN(state[5]))
		return true;
	if (TAKEN(state[6]) && TAKEN(state[7]) && TAKEN(state[8]))
		return true;

	if (TAKEN(state[0]) && TAKEN(state[3]) && TAKEN(state[6]))
		return true;
	if (TAKEN(state[1]) && TAKEN(state[4]) && TAKEN(state[7]))
		return true;
	if (TAKEN(state[2]) && TAKEN(state[5]) && TAKEN(state[8]))
		return true;

	if (TAKEN(state[0]) && TAKEN(state[4]) && TAKEN(state[8]))
		return true;
	if (TAKEN(state[2]) && TAKEN(state[4]) && TAKEN(state[6]))
		return true;

	return false;
	}

void aliceMove()			// Alice = 1.0, Bob = -1.0
	{
	int choice;

	do
		choice = rand() % 9;
	while (state[choice] != 0);

	state[choice] = 1.0;
	}

//************************** main algorithm (old) ***********************//
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

//************************** main algorithm (new) ***********************//
// Main loop:
//	1. Use Q value to choose an optimal action, taking K to K'.
//	2. Play the game until win / lose
//	3. Invoke Q-learning, using the reward to update Q
// Repeat

void main_loop()
	{
	//create neural network for Q learning
	extern NNET *Qnet;
	extern void init_Qnet(void);
	extern NNET *Qnet;
	init_Qnet();

	//error array to keep track of errors
	#define MAX_EPOCHS 30
	double error[MAX_EPOCHS];
	int maxlen = 0;
	int epoch = 1;

	start_NN_plot();
	start_K_plot();

	/* output data to a file
	FILE *fout;
	if ((fout = fopen("randomtest-1.txt", "w")) == NULL)
		{
		fprintf(stderr, "file open failed.\n");
		exit(1);
		}
	*/

	#define dim_K 8
	double *K2 = (double *) malloc(sizeof (double) * dim_K);

	do // Loop over all epochs
		{
		// double squareErrorSum = 0;
		double oldQ = 0.0; // ? TO-DO

		aliceMove();
		if (won(1.0))			// Alice won
			Q_learn(K, K2, -100.0, oldQ);

		// copy current state to K
		for (int k = 0; k < dim_K; ++k)
			K[k] = state[k];

		// Use Q value to choose an optimal action, taking K to K2.
		Q_act(K, K2); // this changes K2

		// If K2 is an invalid move... we should give -reward.
		// There's also the problem of wandering / exploration.
		//

		// Copy K2 to current state
		for (int k = 0; k < dim_K; ++k)
			state[k] = K2[k];

		if (won(-1.0))			// Bob won
			;

		// Invoke Q-learning, using the reward to update Q
		double R = get_reward(K2); // reward is gotten from the state transition
		// double
		oldQ = 0.0; // ? TO-DO
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

		plot_NN(Qnet);
		SDL_Delay(1000 /* milliseconds */);

		}
	while (maxlen < MAX_EPOCHS);

	// fclose(fout);
	free(Qnet);
	free(K2);

	pause_graphics(); //keep the window open
	}

//************************** Genifer main function ***********************//

int main(int argc, char** argv)
	{
	extern void K_wandering_test();
	extern void sine_wave_test();
	extern void sine_wave_test2();
	extern void classic_BP_test();
	extern void classic_BP_test_ReLU();
	extern void forward_test();
	extern void loop_dance_test();
	extern void arithmetic_testA();
	extern void arithmetic_testB();
	extern void arithmetic_testC();
	extern void arithmetic_testD();
	extern void RNN_sine_test();
	extern void BPTT_arithmetic_test();
	extern void BPTT_arithmetic_testB();
	extern void evolve();
	extern void main2();
	extern void jacobian_test();
	extern void Q_test();
	extern void tic_tac_toe_test2();
	extern void tic_tac_toe_test3();

	bool quit = false;
	char whichTest = '\n';
	while (!quit)
		{
		printf("\n\n*** Welcome to Genifer 5.3 ***\n\n");

		printf("[1] forward test\n");
		printf("[2] classic BP test (XOR)\n");
		printf("[3] K wandering test\n");
		printf("[4] sine wave test (differential)\n");
		printf("[5] sine wave test (absolute)\n");
		printf("[6] K dance test\n");
		printf("[7] arthmetic test: test operator\n");
		printf("[8] arithmetic test: learn operator\n");
		printf("[9] arithmetic test: test learned operator\n");
		printf("[a] arithmetic test: learn 1-step operator\n");
		printf("[b] arithmetic test: test learned 1-step operator\n");
		printf("[c] BPTT arithmetic test\n");
		printf("[d] BPTT test learned operator\n");
		printf("[e] rectifier BP test (XOR)\n");
		printf("[f] RNN sine-wave test\n");
		printf("[g] genetic NN test\n");
		printf("[h] run maze\n");
		printf("[i] ???? \n");
		printf("[j] Jacobian NN\n");
		printf("[q] Q-learning test\n");
		printf("[t] Tic-Tac-Toe test #3\n");
		printf("[u] Tic-Tac-Toe test #2\n");
		printf("[x] exit\n");

		do
			whichTest = getchar();
		while (whichTest == '\n');

		switch (whichTest)
			{
			case '1':
				forward_test();
				break;
			case '2':
				classic_BP_test(); // learn XOR function
				break;
			case '3':
				K_wandering_test();
				break;
			case '4':
				sine_wave_test(); // train with differential values of sine
				break;
			case '5':
				sine_wave_test2(); // train with absolute values of sine
				break;
			case '6':
				loop_dance_test(); // make K vector dance in a loop
				break;
			case '7':
				arithmetic_testA(); // primary-school subtraction arithmetic
				break; // test the reference transition function
			case '8':
				arithmetic_testB(); // primary-school subtraction arithmetic
				break; // learn transition operator via back-prop
			case '9':
				arithmetic_testC(); // primary-school subtraction arithmetic
				break; // test transition operator that was learned
			case 'a':
				arithmetic_testD(); // primary-school subtraction arithmetic
				break; // learn 1-step transition operator
			case 'b':
				arithmetic_testE(); // primary-school subtraction arithmetic
				break; // test 1-step transition operator that was learned
			case 'c':
				BPTT_arithmetic_test(); // learn arithmetic operator using BPTT
				break;
			case 'd':
				BPTT_arithmetic_testB(); // learn arithmetic operator using BPTT
				break; // test BPTT learned operator
			case 'e':
				// classic_BP_test_ReLU(); // learn XOR function
				break;
			case 'f':
				RNN_sine_test(); // train RNN to produce sine wave
				break;
			case 'g':
				evolve(); // learn arithmetic operator using BPTT
				break;
			case 'h':
				main2(); // run maze
				break;
			case 'j':
				// jacobian_test(); // test Jacobian neural network
				break;
			case 'q':
				// Q_test(); // test Q learning
				break;
			case 't':
				tic_tac_toe_test3();
				break;
			case 'u':
				tic_tac_toe_test2();
				break;
			case 'x':
				quit = true;
				break;
			}
		}
	}
