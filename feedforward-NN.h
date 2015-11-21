
//**********************struct for NEURON**********************************//
typedef struct NEURON
	{
    double output;
    double *weights;
    double grad;		// "local gradient"
	} NEURON;

//**********************struct for LAYER***********************************//
typedef struct LAYER
	{
    int numNeurons;
    NEURON *neurons;
	} LAYER;

//*********************struct for NNET************************************//
typedef struct NNET
	{
    int numLayers;
    LAYER *layers;
	} NNET; //neural network

#define dim_K	10
