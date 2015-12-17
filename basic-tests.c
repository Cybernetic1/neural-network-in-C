#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_matrix.h>		// GNU scientific library
#include <gsl/gsl_eigen.h>		// ...for finding matrix eigen values
#include <gsl/gsl_complex_math.h>	// ...for complex abs value
#include <stdbool.h>
#include "RNN.h"
#include "feedforward-NN.h"

extern void create_NN(NNET *, int, int *);
extern void create_RTRL_NN(RNN *, int, int *);
extern void free_NN(NNET *, int *);
extern void free_RTRL_NN(RNN *, int *);
extern void forward_prop(NNET *, int, double *);
extern void forward_prop_ReLU(NNET *, int, double *);
extern void forward_prop_SP(NNET *, int, double *);
extern void forward_RTRL(RNN *, int, double *);
extern void back_prop(NNET *, double *);
extern void back_prop_ReLU(NNET *, double *);
extern void RTRL(RNN *, double *);
extern void pause_graphics();
extern void quit_graphics();
extern void start_NN_plot(void);
extern void start_NN2_plot(void);
extern void start_W_plot(void);
extern void start_K_plot(void);
extern void start_output_plot(void);
extern void start_LogErr_plot(void);
extern void plot_NN(NNET *net);
extern void plot_NN2(NNET *net);
extern void plot_W(NNET *net);
extern void plot_output(NNET *net, void ());
extern void plot_LogErr(double, double);
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

// **** Randomly generate an RNN, watch it operate on K and see how K moves
// Observation: chaotic behavior seems to be observed only when the spectral radii of
// weight matrices are sufficiently > 1 (on average).
// The RNN operator is NOT contractive because if K1 ↦ K1', K2 ↦ K2',
// it is not necessary that d(K1',K2') is closer than d(K1,K2).

void K_wandering_test()
	{
	NNET *Net = (NNET *) malloc(sizeof (NNET));
	int neuronsOfLayer[] = {10, 10, 10}; // first = input layer, last = output layer
	int numLayers = sizeof (neuronsOfLayer) / sizeof (int);
	create_NN(Net, numLayers, neuronsOfLayer);
	LAYER lastLayer = Net->layers[numLayers - 1];

	// **** Calculate spectral radius of weight matrices
	printf("Eigen values = \n");
	for (int l = 1; l < numLayers; ++l) // except first layer which has no weights
		{
		int N = 10;
		// assume weight matrix is square, if not, fill with zero rows perhaps (TO-DO)
		gsl_matrix *A = gsl_matrix_alloc(N, N);
		for (int n = 0; n < N; ++n)
			for (int i = 0; i < N; ++i)
				gsl_matrix_set(A, n, i, Net->layers[l].neurons[n].weights[i]);

		gsl_eigen_nonsymmv_workspace *wrk = gsl_eigen_nonsymmv_alloc(N);
		gsl_vector_complex *Aval = gsl_vector_complex_alloc(N);
		gsl_matrix_complex *Avec = gsl_matrix_complex_alloc(N, N);

		gsl_eigen_nonsymmv(A, Aval, Avec, wrk);
		gsl_eigen_nonsymmv_free(wrk);

		gsl_eigen_nonsymmv_sort(Aval, Avec, GSL_EIGEN_SORT_ABS_DESC);

		printf("[ ");
		for (int i = 0; i < N; i++)
			{
			gsl_complex v = gsl_vector_complex_get(Aval, i);
			// printf("%.02f %.02f, ", GSL_REAL(v), GSL_IMAG(v));
			printf("%.02f ", gsl_complex_abs(v));
			}
		printf(" ]\n");

		gsl_matrix_free(A);
		gsl_matrix_complex_free(Avec);
		gsl_vector_complex_free(Aval);
		}

	start_K_plot();
	printf("\nPress 'Q' to quit\n\n");

	// **** Initialize K vector
	for (int k = 0; k < dim_K; ++k)
		K[k] = (rand() / (float) RAND_MAX) - 0.5f;

	double K2[dim_K];
	int quit = 0;
	for (int j = 0; j < 10000; j++) // max number of iterations
		{
		forward_prop(Net, dim_K, K);

		// printf("%02d", j);
		double d = 0.0;

		// copy output to input
		for (int k = 0; k < dim_K; ++k)
			{
			K2[k] = K[k];
			K[k] = lastLayer.neurons[k].output;
			// printf(", %0.4lf", K[k]);
			double diff = (K2[k] - K[k]);
			d += (diff * diff);
			}

		plot_trainer(0); // required to clear window
		plot_K();
		if (quit = delay_vis(60)) // delay in milliseconds
			break;

		// printf("\n");
		if (d < 0.000001)
			{
			fprintf(stderr, "terminated after %d cycles,\t delta = %lf\n", j, d);
			break;
			}
		}

	beep();

	if (!quit)
		pause_graphics();
	else
		quit_graphics();
	free_NN(Net, neuronsOfLayer);
	}

// Train RNN to reproduce a sine wave time-series
// Train the 0-th component of K to move as sine wave
// This version uses the time-step *differences* to train K
// In other words, K moves like the sine wave, but K's magnitude is free to vary and
// will be different every time this test is called.

void sine_wave_test()
	{
	NNET *Net = (NNET *) malloc(sizeof (NNET));
	int neuronsOfLayer[3] = {10, 12, 10}; // first = input layer, last = output layer
	int numLayers = sizeof (neuronsOfLayer) / sizeof (int);
	create_NN(Net, numLayers, neuronsOfLayer);
	LAYER lastLayer = Net->layers[numLayers - 1];
	double K2[dim_K];
	double errors[dim_K];
	int quit;
	double sum_error2;

	start_NN_plot();
	start_W_plot();
	start_K_plot();
	printf("Press 'Q' to quit\n\n");

	// Initialize K vector
	for (int k = 0; k < dim_K; ++k)
		K[k] = (rand() / (float) RAND_MAX) * 2.0 - 1.0;

	for (int i = 0; 1; ++i)
		{
		sum_error2 = 0.0f;

		#define N 20		// loop from 0 to 2π in N divisions
		for (int j = 0; j < N; j++)
			{
			#define Pi 3.141592654
			K[1] = cos(2 * Pi * j / N) + 1.0f; // Phase information to aid learning

			// Allow multiple forward propagations
			forward_prop(Net, dim_K, K);

			// The difference between K[0] and K'[0] should be equal to [sin(θ+dθ) - sinθ]
			// where θ = 2π j/60.
			#define Pi 3.141592654
			#define Amplitude 0.5f
			double dK_star = Amplitude * (sin(2 * Pi * (j + 1) / N) - sin(2 * Pi * j / N));

			// Calculate actual difference between K[0] and K'[0]:
			double dK = lastLayer.neurons[0].output - K[0];

			// The error is the difference between the above two values:
			double error = dK_star - dK;

			// Error in the back-prop NN is recorded as [ideal - actual]:
			//		K* - K = dK*
			//		K' - K = dK
			// thus, K* - k' = dK* - dK 
			errors[0] = error;

			// The rest of the errors are zero:
			for (int k = 1; k < dim_K; ++k)
				errors[k] = 0.0f;

			back_prop(Net, errors);

			// copy output to input
			for (int k = 0; k < dim_K; ++k)
				K[k] = lastLayer.neurons[k].output;

			sum_error2 += (error * error); // record sum of squared errors

			plot_W(Net);
			plot_NN(Net);
			plot_trainer(dK_star / 5.0 * N);
			plot_K();
			if (quit = delay_vis(0))
				break;
			}

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
	else
		quit_graphics();
	free_NN(Net, neuronsOfLayer);
	}

// Train RNN to reproduce a sine wave time-series
// Train the 0-th component of K to move as sine wave
// This version uses the actual value of sine to train K
// New idea: allow RNN to act *multiple* times within each step of the sine wave.
// This will stretch the time scale arbitrarily so the "sine" shape will be lost, but
// I think this kind of learning is more suitable for this RNN model's capability.

// Currently this test fails miserably because the training error is jumping all around
// the place and so BP fails to converge.

void sine_wave_test2()
	{
	NNET *Net = (NNET *) malloc(sizeof (NNET));
	int neuronsOfLayer[3] = {10, 7, 10}; // first = input layer, last = output layer
	int numLayers = sizeof (neuronsOfLayer) / sizeof (int);
	create_NN(Net, numLayers, neuronsOfLayer);
	LAYER lastLayer = Net->layers[numLayers - 1];
	double K2[dim_K];
	double errors[dim_K];
	int quit;
	double sum_error2;

	start_NN_plot();
	start_W_plot();
	start_K_plot();
	printf("Press 'Q' to quit\n\n");

	// Initialize K vector
	for (int k = 0; k < dim_K; ++k)
		K[k] = (rand() / (float) RAND_MAX) * 1.0f;

	for (int i = 0; 1; ++i)
		{
		sum_error2 = 0.0f;

		#define N2 10		// loop from 0 to 2π in N divisions
		for (int j = 0; j < N2; j++)
			{
			// K[1] = cos(2 * Pi * j / N2);		// Phase information to aid learning

			forward_prop(Net, dim_K, K);

			// Desired value
			#define Pi 3.141592654
			#define Amplitude2 1.0f
			double K_star = Amplitude2 * (sin(2 * Pi * j / N2)) + 1.0f;

			// Difference between actual outcome and desired value:
			double error = lastLayer.neurons[0].output - K_star;
			errors[0] = error;

			// The rest of the errors are zero:
			for (int k = 1; k < dim_K; ++k)
				errors[k] = 0.0f;

			back_prop(Net, errors);

			// copy output to input
			for (int k = 0; k < dim_K; ++k)
				K[k] = lastLayer.neurons[k].output;

			sum_error2 += (error * error); // record sum of squared errors

			plot_W(Net);
			plot_NN(Net);
			plot_trainer(K_star);
			plot_K();
			if (quit = delay_vis(0))
				break;
			}

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
	else
		quit_graphics();
	free_NN(Net, neuronsOfLayer);
	}

// Test classical back-prop
// To test convergence, we record the sum of squared errors for the last M and last M..2M
// trials, then compare their ratio.

void classic_BP_test()
	{
	#define ForwardPropMethod	forward_prop_ReLU
	#define BackPropMethod		back_prop_ReLU

	int neuronsOfLayer[] = {2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1}; // first = input layer, last = output layer
	NNET *Net = (NNET *) malloc(sizeof (NNET));
	int numLayers = sizeof (neuronsOfLayer) / sizeof (int);
	create_NN(Net, numLayers, neuronsOfLayer);
	LAYER lastLayer = Net->layers[numLayers - 1];
	double errors[dim_K];

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
	printf("Press 'Q' to quit\n\n");
	start_timer();

	#define ErrorThreshold 0.005

	char str[200], *s;
	for (int i = 0; 1; ++i)
		{
		s = str + sprintf(str, "iteration: %05d: ", i);

		// Create random K vector
		for (int k = 0; k < 2; ++k)
			K[k] = (rand() / (float) RAND_MAX);
		// printf("*** K = <%lf, %lf>\n", K[0], K[1]);

		//		if ((i % 4) == 0)
		//			K[0] = 1.0, K[1] = 0.0;
		//		if ((i % 4) == 1)
		//			K[0] = 0.0, K[1] = 0.0;
		//		if ((i % 4) == 2)
		//			K[0] = 0.0, K[1] = 1.0;
		//		if ((i % 4) == 3)
		//			K[0] = 1.0, K[1] = 1.0;

		ForwardPropMethod(Net, 2, K); // dim K = 2

		// Desired value = K_star
		double training_err = 0.0;
		for (int k = 0; k < 1; ++k) // output has only 1 component
			{
			// double ideal = K[k];				/* identity function */
			#define f2b(x) (x > 0.5f ? 1 : 0)	// convert float to binary
			// ^ = binary XOR
			double ideal = ((double) (f2b(K[0]) ^ f2b(K[1]))); // ^ f2b(K[2]) ^ f2b(K[3])))
			// #define Ideal ((double) (f2b(K[k]) ^ f2b(K[2]) ^ f2b(K[3])))
			// double ideal = 1.0f - (0.5f - K[0]) * (0.5f - K[1]);
			// printf("*** ideal = %lf\n", ideal);

			// Difference between actual outcome and desired value:
			double error = ideal - lastLayer.neurons[k].output;
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
		BackPropMethod(Net, errors);
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

				ForwardPropMethod(Net, 2, K);

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

		// if (ratio - 0.5f < 0.0000001)	// ratio == 0.5 means stationary
		// if (test_err < 0.01)
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
	free_NN(Net, neuronsOfLayer);
	}

// Test forward propagation

void forward_test()
	{
	NNET *Net = (NNET *) malloc(sizeof (NNET));
	int neuronsOfLayer[4] = {4, 3, 3, 2}; // first = input layer, last = output layer
	int numLayers = sizeof (neuronsOfLayer) / sizeof (int);
	create_NN(Net, numLayers, neuronsOfLayer);
	LAYER lastLayer = Net->layers[numLayers - 1];
	double sum_error2;

	start_NN_plot();
	start_W_plot();
	start_K_plot();

	printf("This test sets all the weights to 1, then compares the output with\n");
	printf("the test's own calculation, with 100 randomized inputs.\n\n");

	// Set all weights to 1
	for (int l = 1; l < numLayers; l++) // for each layer
		for (int n = 0; n < neuronsOfLayer[l]; n++) // for each neuron
			for (int k = 0; k <= neuronsOfLayer[l - 1]; k++) // for each weight
				Net->layers[l].neurons[n].weights[k] = 1.0f;

	for (int i = 0; i < 100; ++i)
		{
		// Randomize K
		double sum = 1.0f;
		for (int k = 0; k < 4; ++k)
			{
			K[k] = (rand() / (float) RAND_MAX) * 2.0 - 1.0;
			sum += K[k];
			}

		forward_prop(Net, 4, K);

		// Expected output value:
		double K_star = sigmoid(3.0f * sigmoid(3.0f * sigmoid(sum) + 1.0f) + 1.0f);

		// Calculate error
		sum_error2 = 0.0f;
		for (int k = 0; k < 2; ++k)
			{
			// Difference between actual outcome and desired value:
			double error = lastLayer.neurons[k].output - K_star;
			sum_error2 += (error * error); // record sum of squared errors
			}

		plot_W(Net);
		plot_NN(Net);
		plot_trainer(0);
		plot_K();
		delay_vis(50);

		printf("iteration: %05d, error: %lf\n", i, sum_error2);
		}

	pause_graphics();
	free_NN(Net, neuronsOfLayer);
	}

// Randomly generate a loop of K vectors;  make the RNN learn to traverse this loop.

void loop_dance_test()
	{
	NNET *Net = (NNET *) malloc(sizeof (NNET));
	int neuronsOfLayer[4] = {dim_K, 10, 10, dim_K}; // first = input layer, last = output layer
	int numLayers = sizeof (neuronsOfLayer) / sizeof (int);
	create_NN(Net, numLayers, neuronsOfLayer);
	LAYER lastLayer = Net->layers[numLayers - 1];
	double sum_error2;
	double errors[dim_K];
	int quit;

	// start_NN_plot();
	start_W_plot();
	start_K_plot();

	printf("Randomly generate a loop of K vectors;\n");
	printf("Make the RNN learn to traverse this loop.\n\n");

	#define LoopLength 3
	double Kn[LoopLength][dim_K];
	for (int i = 0; i < LoopLength; ++i)
		for (int k = 0; k < dim_K; ++k)
			Kn[i][k] = (rand() / (float) RAND_MAX); // random in [0,1]

	for (int j = 0; true; ++j) // iterations
		{
		sum_error2 = 0.0f;

		for (int i = 0; i < LoopLength; ++i) // do one loop
			{
			forward_prop(Net, dim_K, K);

			// Expected output value = Kn[i][k].
			// Calculate error:
			for (int k = 0; k < dim_K; ++k)
				{
				// Difference between actual outcome and desired value:
				double error = lastLayer.neurons[k].output - Kn[i][k];
				errors[k] = error; // record this for back-prop
				sum_error2 += (error * error); // record sum of squared errors

				// copy output to input
				K[k] = lastLayer.neurons[k].output;
				}

			back_prop(Net, errors);

			plot_W(Net);
			// plot_NN(Net);
			plot_trainer(0);
			plot_K();
			if (quit = delay_vis(0))
				break;
			}

		printf("iteration: %05d, error: %lf\n", j, sum_error2);
		if (quit)
			break;
		}

	pause_graphics();
	free_NN(Net, neuronsOfLayer);
	}

// **************** 2-Digit Primary-school Subtraction Arithmetic test *****************

// The goal is to perform subtraction like a human child would.
// Input: 2-digit numbers A and B, for example "12", "07"
// Output: A - B, eg:  "12" - "07" = "05"

// State vector = [ A1, A0, B1, B0, C1, C0, carry-flag, current-digit, result-ready-flag,
//		underflow-error-flag ]

// Algorithm:

// If current-digit = 0:
//		if A0 >= B0 then C0 = A0 - B0
//		else C0 = 10 + (A0 - B0) , carry-flag = 1
//		current-digit = 1

// If current-digit = 1:
//		if A1 >= B1 then
//			C1 = A1 - B1
//		else Underflow Error
//		if carry-flag = 0:
//			result-ready = 1
//		else	// carry-flag = 1
//			if C1 >= 1
//				--C1
//			else Underflow error
//			result-ready = 1

// This defines the transition operator acting on vector space K1 (of dimension 10)

void RNN_sine_test()
	{
	// create RNN
	RNN *Net = (RNN *) malloc(sizeof (RNN));
	int neuronsOfLayer[4] = {2, 10, 10, 1}; // first = input layer, last = output layer
	int numLayers = sizeof (neuronsOfLayer) / sizeof (int);
	create_RTRL_NN(Net, numLayers, neuronsOfLayer);
	rLAYER lastLayer = Net->layers[numLayers - 1];

	int dimK = 2;
	double K2[dimK];
	double errors[dim_K];
	double sum_error2;
	int quit;

	start_NN_plot();
	start_W_plot();
	start_K_plot();
	printf("RNN sine test\n");
	printf("Press 'Q' to quit\n\n");

	// Initialize K vector
	K[0] = (rand() / (float) RAND_MAX) * 1.0f;

	// new sequence item, new error
	// weight change = given by new gradient (for current time-step)
	// new gradient = given by recursive formula (old gradient)

	for (int i = 0; true; ++i)
		{
		sum_error2 = 0.0f;

		#define N3 10		// loop from 0 to 2π in N divisions
		for (int j = 0; j < N3; j++)
			{
			K[1] = cos(2 * Pi * j / N2); // Phase information to aid learning

			forward_RTRL(Net, dimK, K);

			// create test sequence (sine wave?)
			// Desired value
			#define Amplitude2 1.0f
			double K_star = Amplitude2 * (sin(2.0 * Pi * j / N2)) + 1.0f;

			// Difference between actual outcome and desired value:
			double error = lastLayer.neurons[0].output - K_star;
			errors[0] = error;

			// The rest of the errors are zero:
			for (int k = 1; k < dimK; ++k)
				errors[k] = 0.0f;

			RTRL(Net, errors);

			// copy output to input
			for (int k = 0; k < dimK; ++k)
				K[k] = lastLayer.neurons[k].output;

			sum_error2 += (error * error); // record sum of squared errors

			// plot_W(Net);
			// plot_NN(Net);
			plot_trainer(K_star);
			plot_K();
			if (quit = delay_vis(0))
				break;
			}

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
	else
		quit_graphics();
	free_RTRL_NN(Net, neuronsOfLayer);
	}
