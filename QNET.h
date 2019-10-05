
//**********************struct for NEURON**********************************//
typedef struct NEURON
	{
    double output;
    // double grad;			// "local gradient"
	} NEURON;

//**********************struct for LAYER***********************************//
typedef struct LAYER
	{
    int numNeurons;
    NEURON *neurons;
    double α, β, γ, δ;		// The 4 distinct weights
	} LAYER;

//*********************struct for NNET************************************//
typedef struct QNET
	{
    int numLayers;
    LAYER *layers;
	} QNET; //neural network
