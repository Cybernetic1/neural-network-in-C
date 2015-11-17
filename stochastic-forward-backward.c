// YKY's idea of stochastic forward-backward search, for training a recurrent network

// Stragegy: stochastic forward-backward and record noise, then back-prop to bridge gaps
// Require: simple back-prop

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <time.h>		// time as random seed in create_NN()
#include "RNN.h"

#define Eta 0.001			// learning rate
#define BIASOUTPUT 1.0		// output for bias. It's always 1.

