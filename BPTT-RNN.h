#define Nfold 2					// network will be unfolded for N time steps

//**********************struct for NEURON**********************************//
typedef struct rNEURON
	{
    double output[Nfold];
    double *weights;
    double grad[Nfold];			// "local gradient", allow for 2 time steps
	} rNEURON;

//**********************struct for LAYER***********************************//
typedef struct rLAYER
	{
    int numNeurons;
    rNEURON *neurons;
	} rLAYER;

//*********************struct for RNN************************************//
typedef struct RNN
	{
    int numLayers;
    rLAYER *layers;
	} RNN;

#define dim_K	10
