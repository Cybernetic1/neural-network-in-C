#define dimX	10

//**********************struct for NEURON**********************************//
typedef struct JNEURON
	{
    double output;
    double *weights;
	} JNEURON;

//**********************struct for LAYER***********************************//
typedef struct JLAYER
	{
    int numNeurons;
    JNEURON *neurons;
	double J1[dimX * dimX];			// forward Jacobian (propagating at this layer)
	double J[dimX * dimX];			// inverse Jacobian (propagating at this layer)
	double grad[dimX * dimX];		// local gradient for calculating dJ/dw
	} JLAYER;

//*********************struct for NNET************************************//
typedef struct JNET
	{
    int numLayers;
    JLAYER *layers;
	} JNET; //neural network
