// Q-learner
// Basically it is a feed-forward neural net that approximates the Q-value function
// used in reinforcement learning.  Call this neural network the Q-net.
// Q-net accepts K and K' as input.  K is the current state of the Reasoner,
// K' is the "next" state.  Q-net's output is the Q-value, ie, the utility at state
// K making the transition to K'.
// In other words, Q-net approximates the function K x K' → ℝ.
// Given K, K', and Q, Q-net learns via traditional back-prop.
// The special thing about Q-net is that there is another algorithm that computes,
// when given K, the K' that achieves maximum Q value.  This is the optimization part.

// TO-DO:
// * Learn the Q: X,Y → ℝ map, where X = input, Y = output
//   Though only the difference is important, we need to learn specific Q values.
//   Would it perform better?  But this is Q learning versus previously was V learning.
//   The advantage of Q is model-free, but what is a model?  Perhaps it is the internal
//	 structure of Q?

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
extern void plot_W(NNET *);
extern void start_W_plot(void);

//************************** prepare Q-net ***********************//
NNET *Qnet;

#define dimK 9
int QnumLayers = 4;
int QneuronsPerLayer[] = {dimK * 2, 10, 7, 1};

void init_Qnet()
	{
	// int numLayers2 = 5;
	//the first layer -- input layer
	//the last layer -- output layer
	// int neuronsPerLayer[5] = {2, 3, 4, 4, 4};
	// int neuronsPerLayer[5] = {18, 18, 15, 10, 1};
	// int neuronsPerLayer2[5] = {18, 40, 30, 20, 1};

	Qnet = (NNET*) malloc(sizeof (NNET));
	//create neural network for backpropagation
	Qnet = create_NN(QnumLayers, QneuronsPerLayer);

	start_W_plot();
	// return Qnet;
	}

void load_Qnet(char *fname)
	{
	int numLayers2;
	int *neuronsPerLayer2;
	extern NNET * loadNet(int *, int *p[], char *);
	Qnet = loadNet(&numLayers2, &neuronsPerLayer2, fname);
	// LAYER lastLayer = Vnet->layers[numLayers - 1];

	return;
	}

void save_Qnet(char *fname)
	{
	extern void saveNet(NNET *, int, int *, char *, char *);

	saveNet(Qnet, QnumLayers, QneuronsPerLayer, "", fname);
	}

//************************** Q-learning ***********************//
// Algorithm:
// ---- (Part 1) Acting ----
// At some point in the main algorithm, control is passed here.
// At current state K, We pick an optimal action according to Q.
// We make the state transition from K to K', getting reward R.
// ---- (Part 2) Learning ----
// Then we can use R to update the Q value via:
//			Q(K,K') += η { R + γ max_a Q(K',a) }.
// So the above is a ΔQ value that needs to be added to Q(K,K').
// The Q-net computes Q(K,K'); its output should be adjusted by ΔQ.
// Thus we use back-prop to adjust the weights in Q-net to achieve this.
// ==============================================================

// Finds Q value by forward-propagation

double getQ(double K[], double K2[])
	{
	// For input, we need to combine K1 and K2 into a single vector
	double K12[dimK * 2];
	for (int k = 0; k < dimK; ++k)
		{
		K12[k] = (double) K[k];
		K12[k + dimK] = (double) K2[k];
		}

	forward_prop_sigmoid(Qnet, dimK * 2, K12);

	LAYER LastLayer = (Qnet->layers[QnumLayers - 1]);
	// The last layer has only 1 neuron, which outputs the Q value:
	return LastLayer.neurons[0].output;
	}

// returns the Euclidean norm (absolute value, or size) of the gradient vector

double norm(double grad[dimK])
	{
	double r = 0.0;
	for (int k = 0; k < dimK; ++k)
		r += (grad[k] * grad[k]);
	return sqrt(r);
	}

// **** Learn a simple Q-value map given specific Q values
// **** Used in network initialization
void train_Q(int s[dimK], double Q)
	{
	double S[dimK * 2];
	static int count = 0;

	for (int j = 0; j < 1; ++j)		// iterate a few times
		{
		for (int k = 0; k < dimK; ++k)
			{
			S[k] = (double) s[k];

			// 2nd argument should be random
			S[k + dimK] = (rand() / (float) RAND_MAX) * 2.0 - 1.0; // in [+1,-1]
			}

		forward_prop_sigmoid(Qnet, dimK * 2, S);

		LAYER LastLayer = (Qnet->layers[QnumLayers - 1]);
		// The last layer has only 1 neuron, which outputs the Q value:
		double Q2 = LastLayer.neurons[0].output;

		double error[1];
		*error = Q - Q2; // desired - actual

		back_prop(Qnet, error);
		}

	if (++count == 1000)
		{
		plot_W(Qnet);
		count = 0;
		}
	}


// (Part 2) Q-learning:
// Invoke ordinary back-prop to learn Q.
// On entry, we have just made a transition K1 -> K2 with maximal Q(K1, a: K1->K2)
// and gotten a reward R(K1->K2).
// We need to calculate the max value of Q(K2,a) which, beware, is from the NEXT state K2.
// We know old Q(K1,K2), but it is now adjusted to Q += ΔQ, thus the "error" for back-prop
// is ΔQ.

// Why is Bellman update needed here?  We made a transition.

// K2 needs to be re-interpreted in Sayaka-2 architecture.
// K2 is now an "action", not a state.
// We need to calculate the next state X2, but which is now unavailable.
// If the next state is invalid, then of course it should have -max value.
// Else we can get maxQ(X2).

void Q_learn(int K1[dimK], int K2[dimK], double R)
	{
	double maxQ(int [dimK], double [dimK]);
	double K_out[dimK];

	#define Gamma	0.95
	#define Eta		0.2

	// Calculate ΔQ = η { R + γ max_a Q(K2,a) }
	double dQ[1];
	dQ[0] = Eta * (R + Gamma * maxQ(K2, K_out));

	// Adjust old Q value
	// oldQ += dQ;

	double K[dimK * 2];
	for (int k = 0; k < dimK; ++k)
		{
		K[k] = (double) K1[k];
		K[k + dimK] = (double) K2[k];
		}

	// Invoke back-prop a few times (perhaps this would make the learning effect stronger?)
	for (int i = 0; i < 2; ++i)
		{
		// We need to forward_prop Qnet with input (K1,K2)
		forward_prop_sigmoid(Qnet, dimK * 2, K);

		back_prop(Qnet, dQ);
		}
	}

// **** Learn a simple V-value map via backprop
/*
void learn_V(int s2[dimK], int s[dimK])
	{
	double S2[dimK], S[dimK];

	for (int j = 0; j < 4; ++j)
		{

		for (int k = 0; k < dimK; ++k)
			{
			S2[k] = (double) s2[k];
			S[k] = (double) s[k];
			}

		forward_prop_sigmoid(Qnet, dimK, S2);

		LAYER LastLayer = (Qnet->layers[QnumLayers - 1]);
		// The last layer has only 1 neuron, which outputs the Q value:
		double Q2 = LastLayer.neurons[0].output;

		forward_prop_sigmoid(Qnet, dimK, S);
		double Q = LastLayer.neurons[0].output;

		double error[1];
		*error = Q2 - Q;

		back_prop(Qnet, error);
		}
	}
*/

// (Part 1) Q-acting:
// Find K2 that maximizes Q(K,K2).  Q is a real number.
// Method: numerical differentiation to find the gradient ∇Q = [∂Q/∂K2] which is a vector.
//		The approximation formula for each component of ∂Q/∂K2 is:  (subscripts omitted)
//			∂Q/∂K2 ≈ { Q(K2 + δ) - Q(K2 - δ) } /2δ
// TO-DO: Perhaps with multiple random restarts
// Note: function changes the components of K2.

void Q_act(double K[dimK], double K2[dimK])
	{
	double gradQ[dimK]; // the gradient vector ∇Q = [∂Q/∂K2]

	do // While change is smaller than threshold
		{
		// Start with a random K2
		for (int k = 0; k < dimK; ++k)
			K2[k] = (rand() / (float) RAND_MAX) * 2.0 - 1.0; // in [+1,-1]

		// Find the steepest direction [∂Q/∂K2], using numerical differentiation.
		#define delta	0.1
		for (int k = 0; k < dimK; ++k)
			{
			// Create 2 copies of K2, whose k-th component is added / subtracted with δ
			double K2plus[dimK], K2minus[dimK];
			for (int k2 = 0; k2 < dimK; ++k2)
				K2plus[k2] = K2minus[k2] = K2[k2];
			K2plus[k] += delta;
			K2minus[k] -= delta;

			gradQ[k] = (getQ(K, K2plus) - getQ(K, K2minus)) / 2 / delta;
			}

		// Move a little along the gradient direction: K2 += -λ ∇Q
		// (There seems to be a negative sign in the above formula)
		#define Lambda	0.1
		for (int k = 0; k < dimK; ++k)
			K2[k] += (-Lambda * gradQ[k]);
		}
		#define Epsilon 0.005
		while (norm(gradQ) > Epsilon);

	// Return with optimal K2 value
	}

// Find maximum Q(K,K') value at state K, by varying K'.
// Method: gradient descent, using numerical differentiation to find the gradient [∂Q/∂K'].
// Algorithm is similar to above.
// 2nd argument is a place-holder.
double maxQ(int K[dimK], double K2[dimK])
	{
	double gradQ[dimK]; // the gradient vector ∇Q = [∂Q/∂K2]
	double K1[dimK];
	double gradSize;

	for (int k = 0; k < dimK; ++k)
		K1[k] = (double) K[k];

	int tries = 0;
	do // While change is smaller than threshold
		{
		// Start with a random K2
		for (int k = 0; k < dimK; ++k)
			K2[k] = (rand() / (float) RAND_MAX) * 2.0 - 1.0; // in [+1,-1]

		// Find the steepest direction [∂Q/∂K2], using numerical differentiation.
		#define delta	0.001
		for (int k = 0; k < dimK; ++k)
			{
			// Create 2 copies of K2, whose k-th component is added / subtracted with δ
			double K2plus[dimK], K2minus[dimK];
			for (int k2 = 0; k2 < dimK; ++k2)
				K2plus[k2] = K2minus[k2] = K2[k2];
			K2plus[k] += delta;
			K2minus[k] -= delta;

			gradQ[k] = (getQ(K1, K2plus) - getQ(K1, K2minus)) / (2 * delta);
			}

		// Move a little along the gradient direction: K2 += -λ ∇Q
		// (There seems to be a negative sign in the above formula)
		#define Lambda	0.1
		for (int k = 0; k < dimK; ++k)
			K2[k] += (-Lambda * gradQ[k]);

		gradSize = norm(gradQ);
		// printf("gradient norm = %f\r", gradSize);
		++tries;
		}
	#define Epsilon 0.1
	#define MaxTries 20000
	while (gradSize > Epsilon && tries < MaxTries);

	if (tries >= MaxTries)						// need to handle exception here
		{
		printf("fail: ");
		// The result in K2 may still be usable?
		}

	double result = getQ(K1, K2);
	printf("%2.3f ", result);
	plot_W(Qnet);
	return result; // return Q value
	}
