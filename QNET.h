
//********************** struct for NEURON **********************************//
typedef struct NEURON
	{
    double output;
    double grad;			// "local gradient"
	} NEURON;

//********************** struct for LAYER ***********************************//
typedef struct LAYER
	{
    NEURON *neurons;
    double alpha, beta, gamma, delta;		// The 4 distinct weights
	} LAYER;

//********************* struct for QNET ************************************//
typedef struct QNET
	{
    int numLayers;
    LAYER *layers;
	} QNET;					// neural network

#define dim_V 4				// dimension of input, output, and hidden layers, all the same
