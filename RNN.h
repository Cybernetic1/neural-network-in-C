
//**********************struct for NEURON**********************************//
typedef struct NEURON
	{
    double input;
    double output;
    double *weights;
    double delta;
    double error;
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
    double *inputs;
    int numLayers;
    LAYER *layers;
	} NNET; //neural network

#define dim_K	10
