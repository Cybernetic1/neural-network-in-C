#include <cstdlib>
#include <iostream>
#include <math.h>
#include <stdbool.h>
#include <random>
#include "QNET.h"

using namespace std;

extern "C" QNET *create_QNN(int);
extern "C" void free_QNN(QNET *);
extern "C" void forward_prop_quadratic(QNET *, double*);
extern "C" void back_prop_quadratic(QNET *, double*);
extern "C" void re_randomize(QNET *);

// extern "C" void pause_graphics();
// extern "C" void quit_graphics();
// extern "C" void start_NN_plot(void);
// extern "C" void start_NN2_plot(void);
// extern "C" void start_W_plot(void);
// extern "C" void start_K_plot(void);
// extern "C" void start_output_plot(void);
// extern "C" void start_LogErr_plot(void);
// extern "C" void restart_LogErr_plot(void);
// extern "C" void plot_NN(NNET *net);
// extern "C" void plot_NN2(NNET *net);
// extern "C" void plot_W(NNET *net);
// extern "C" void plot_output(NNET *net, void ());
// extern "C" void plot_LogErr(double, double);
// extern "C" void flush_output();
// extern "C" void plot_tester(double, double);
// extern "C" void plot_K();
// extern "C" int delay_vis(int);
// extern "C" void plot_trainer(double);
// extern "C" void plot_ideal(void);
// extern "C" void beep(void);
// extern "C" void start_timer(), end_timer(char *);

/* BASIC IDEA
   ==========
1. Weights are constrained by a number of equations.
2. Then the NN would be symmetric;  This has already been proven in Python.
3. What we want to demo here is to train such an NN to approximate some symmetric function.
4. So we need to test against some random symmetric function.  This can be generated from
	a bunch of random data points and permuting them.
5. Then we train the NN with the constraints to see if it can learn the target.
6. The "colorful" sym-NN seems to have only equality constraints for 1 layer, but we don't
	know for 2 layers.  This has to be solved by Python algebraically.
7. "Colorless" / "unordered" version sym-NN contain additive constraints even for 1 layer.
8. Multiple layers seem to involve *polynomial* constraints, but this has yet to be confirmed.

	How to implement constraints
	============================
1. Good news: all constraints are equality constraints,
	and there are only 4 distinct weights per layer.
	Obviously the weights should be represented as a sparse matrix.
2. Forward propagation:
*/


// ****** The following is an old idea, now abandoned ******
// 1. create an FFNN, h(x)
//		F(x₁, x₂, ..., xₙ) = g(h(x₁), h(x₂), ..., h(xₙ)) would be symmetric if g() is.
// 2. Test whether F can learn a symmetric target function.  For example, let input size N = 3,
//		train F to detect (0.3, 0.1, 0.4) in any order
//		But each input also has M dimensions.
// 3. The test set should be:  random vectors of size M, forming a list of size N
//		In other words, each datum is of size M × N
//		The objective function measures how close it is to:
//		(0.3, 0.4, 0.3), (0.1, 0.2, 0.1), (0.4, 0.5, 0.4)
// 4. Problem: there is a sub-pattern which is INSIDE each vector.
//		Perhaps we start with 1-dim vectors first?
// 5. Another problem is that when generating the test set, we should make the appearance of
//		(.3 .1 .4) more frequent.  

#define ForwardPropMethod	forward_prop_quadratic
#define ErrorThreshold		0.02

extern "C" void symmetric_test()
	{
	// std::default_random_engine generator;
	// std::normal_distribution<double> distribution(0.0,0.2);

	int numLayers = 3;						// must be at least 3
	QNET *Net = create_QNN(numLayers);		// our NN for learning
	LAYER lastLayer = Net->layers[numLayers - 1];
	double errors[dim_V];

	printf("test 撚佢个 forward prop...\n");
	double K[dim_V];
	printf("K ={ ");
	for (int k = 0; k < 4; ++k)
		{
		K[k] = (rand() / (float) RAND_MAX);
		printf("%f ", K[k]);
		}
	printf("}\n");
	// Permute K
	int perm[4] = {0, 1, 2, 3};
	// for (int i = 0; i < 4; i++)
	//	perm[i] = i;
	// Random permutation the order
	for (int i = 0; i < 4; i++)
		{
		int j, t;
		j = rand() % (4-i) + i;
		t = perm[j]; perm[j] = perm[i]; perm[i] = t; // Swap i and j
		}
	for (int i = 0; i < 4; ++i)
		printf("%d ", perm[i]);
	double sigma_K[4];
	printf("\nσK={ ");
	for (int i = 0; i < 4; ++i)
		{
		sigma_K[i] = K[perm[i]];
		printf("%f ", sigma_K[i]);
		}
	printf("}\n");
	ForwardPropMethod(Net, sigma_K); // dim K = 4
	double f_sigma_K[4];
	printf("\nf(σK)={ ");
	for (int i = 0; i < 4; ++i)
		{
		f_sigma_K[i] = lastLayer.neurons[i].output;
		printf("%f ", f_sigma_K[i]);
		}
	printf("}\n");
	// Compare f(σK) == σf(K)?
	ForwardPropMethod(Net, K); // dim K = 4
	double f_K[4];
	for (int i = 0; i < 4; ++i)
		f_K[i] = lastLayer.neurons[i].output;
	printf("σf(K)={ ");
	for (int i = 0; i < 4; ++i)
		printf("%f ", f_K[perm[i]]);
	printf("}\n");

	int userKey = 0;
	#define M	50			// how many errors to record for averaging
	double errors1[M], errors2[M]; // two arrays for recording errors
	double sum_err1 = 0.0, sum_err2 = 0.0; // sums of errors
	int tail = 0; // index for cyclic arrays (last-in, first-out)

	for (int i = 0; i < M; ++i) // clear errors to 0.0
		errors1[i] = errors2[i] = 0.0;

	// start_NN_plot();
	// start_W_plot();
	// start_K_plot();
	// start_output_plot();
	// start_LogErr_plot();
	// plot_ideal();
	printf("Press 'Q' to quit\n\n");
	// start_timer();

	char status[1024], *s;
	for (int i = 1; 1; ++i)
		{
		s = status + sprintf(status, "[%05d] ", i);

		// Create random K vector
		for (int k = 0; k < dim_V; ++k)
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

		ForwardPropMethod(Net, K); // dim K = 4

		// Desired value = K_star
		double training_err = 0.0;
		for (int k = 0; k < dim_V; ++k) // output has 4 components
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
		if (mean_err < 2.0)
			s += sprintf(s, "mean |e|=%1.06lf, ", mean_err);
		else
			s += sprintf(s, "mean |e|=%e, ", mean_err);

		// record new error in cyclic arrays
		errors2[tail] = errors1[tail];
		errors1[tail] = training_err;
		++tail;
		if (tail == M) // loop back in cycle
			tail = 0;

		// plot_W(Net);
		back_prop_quadratic(Net, errors);
		// plot_W(Net);
		// pause_graphics();

		if ((i % 2000) == 0)
			{
			// Testing set
			double test_err = 0.0;
			#define numTests 50
			for (int j = 0; j < numTests; ++j)
				{
				// Create random K vector
				for (int k = 0; k < dim_V; ++k)
					K[k] = ((double) rand() / (double) RAND_MAX);
				// plot_tester(K[0], K[1]);

				ForwardPropMethod(Net, K);	// (?) expect dim = 2

				// Desired value = K_star
				double single_err = 0.0;
				for (int k = 0; k < dim_V; ++k)
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
			if (test_err < 2.0)
				s += sprintf(s, "random test |e|=%1.06lf, ", test_err);
			else
				s += sprintf(s, "random test |e|=%e, ", test_err);
			if (test_err < ErrorThreshold)
				break;
			}

		if (i > 50 && (isnan(mean_err) || mean_err > 10.0))
			{
			re_randomize(Net);
			sum_err1 = 0.0; sum_err2 = 0.0;
			tail = 0;
			for (int j = 0; j < M; ++j) // clear errors to 0.0
				errors1[j] = errors2[j] = 0.0;
			i = 1;

			// restart_LogErr_plot();
			// start_timer();
			printf("\n****** Network re-randomized.\n");
			}

		if ((i % 5000) == 0)
			{
			double ratio = (sum_err2 - sum_err1) / sum_err1;
			if (ratio > 0)
				s += sprintf(s, "|e| ratio=%e", ratio);
			else
				s += sprintf(s, "|e| ratio=\x1b[31m%e\x1b[39;49m", ratio);
			//if (isnan(ratio))
			//	break;
			}

		if ((i % 5000) == 0) // display status periodically
			{
			printf("%s\n", status);
			// plot_NN(Net);
			// plot_W(Net);
			// plot_LogErr(mean_err, ErrorThreshold);
			// plot_output(Net, ForwardPropMethod);
			// flush_output();
			// plot_trainer(0);		// required to clear the window
			// plot_K();
			// userKey = delay_vis(0);
			}

		// if (ratio - 0.5f < 0.0000001)	// ratio == 0.5 means stationary
		// if (test_err < 0.01)

		if (userKey == 1)
			break;
		else if (userKey == 3)			// Re-start with new random weights
			{
			re_randomize(Net);
			sum_err1 = 0.0; sum_err2 = 0.0;
			tail = 0;
			for (int j = 0; j < M; ++j) // clear errors to 0.0
				errors1[j] = errors2[j] = 0.0;
			i = 1;

			// restart_LogErr_plot();
			// start_timer();
			printf("\n****** Network re-randomized.\n");
			userKey = 0;
			// beep();
			// pause_key();
			}
		}

	// end_timer(NULL);
	// beep();
	// plot_output(Net, ForwardPropMethod);
	// flush_output();
	// plot_W(Net);

	// if (userKey == 0)
		// pause_graphics();
	// else
		// quit_graphics();
	free_QNN(Net);
	}

int main() {
	printf("\n\x1b[32m——`—,—{\x1b[31;1m@\x1b[0m\n");	// Genifer logo ——`—,—{@
	symmetric_test();
}
