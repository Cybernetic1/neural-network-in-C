#include <stdbool.h>
#include <SDL2/SDL.h>

#include "RNN.h"

SDL_Renderer *gfx_NNW;					// For YKY's NN visualizer
SDL_Window *win_NNW;

SDL_Renderer *gfx_NN;					// For YKY's NN visualizer
SDL_Window *win_NN;

SDL_Renderer *gfx_NN2;					// For Seh's NN visualizer
SDL_Window *win_NN2;

SDL_Renderer *gfx_K;					// For K-vector visualizer
SDL_Window *win_K;

#define f2i(v) ((int)(256.0f * v))		// for converting color values

// *************************** YKY's NN weights visualizer ********************************

#define NNW_box_width 900
#define NNW_box_height 400

void start_NNW_plot(void)
	{

	if (SDL_Init(SDL_INIT_VIDEO) != 0)
		{
		printf("SDL_Init Error: %s \n", SDL_GetError());
		return;
		}

	win_NNW = SDL_CreateWindow("NN weights", 80, 1200, NNW_box_width, NNW_box_height, SDL_WINDOW_SHOWN);
	if (win_NN == NULL)
		{
		printf("SDL_CreateWindow Error: %s \n", SDL_GetError());
		SDL_Quit();
		return;
		}

	gfx_NNW = SDL_CreateRenderer(win_NNW, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (gfx_NNW == NULL)
		{
		SDL_DestroyWindow(win_NNW);
		printf("SDL_CreateRenderer Error: %s \n", SDL_GetError());
		SDL_Quit();
		return;
		}
	}

void plot_NNW(NNET *net)
	{
	SDL_SetRenderDrawColor(gfx_NNW, 0, 0, 0, 0xFF);
	SDL_RenderClear(gfx_NNW);		//Clear screen
	
	SDL_SetRenderDrawBlendMode(gfx_NNW, SDL_BLENDMODE_BLEND);

	#define Volume 20.0f

	#define numLayers (net->numLayers)
	for (int l = 1; l < numLayers; l++)		// Note: layer 0 has no weights
		{
		double gain = 1.0f;
		// increase amplitude for hidden layers
		if (l > 0 && l < numLayers - 1)
			gain = 5.0f;
		else
			gain = 1.0f;

		// set color
		float r = ((float) l ) / numLayers;
		float b = 1.0f - ((float) l ) / numLayers;
		SDL_SetRenderDrawColor(gfx_NNW, f2i(r), 0x50, f2i(b), 0xFF);

		int nn = net->layers[l].numNeurons;

		int neuronWidth = (NNW_box_width - 20 - 50) / (nn - 1);

		// draw baseline
		#define Y_step ((NNW_box_height - (int) Volume * 10) / (numLayers - 2))
		int baseline_y = (int) Volume * 5 + (l - 1) * Y_step;
		SDL_RenderDrawLine(gfx_NNW, 10, baseline_y, \
			neuronWidth * (nn - 1) + 10 + 50, baseline_y);

		SDL_SetRenderDrawColor(gfx_NNW, f2i(r), 0x50, f2i(b), 0x80);
		
		for (int n = 0; n < nn; n++)		// for each neuron on layer l
			{
			for (int m = 0; m < net->layers[l - 1].numNeurons; ++m)		// for each weight
				{
				double weight = Volume * net->layers[l].neurons[n].weights[m];
			
				int basepoint_x = 10 + neuronWidth * n + m * 2;
				SDL_RenderDrawLine(gfx_NNW, basepoint_x, baseline_y, \
					basepoint_x, baseline_y - weight);
				}
			}
		}

	SDL_RenderPresent(gfx_NNW);
	}

// *************************** YKY's NN visualizer *************************************

#define NN_box_width 600
#define NN_box_height 400

void start_NN_plot(void)
	{

	if (SDL_Init(SDL_INIT_VIDEO) != 0)
		{
		printf("SDL_Init Error: %s \n", SDL_GetError());
		return;
		}

	win_NN = SDL_CreateWindow("NN activity", 100, 600, NN_box_width, NN_box_height, SDL_WINDOW_SHOWN);
	if (win_NN == NULL)
		{
		printf("SDL_CreateWindow Error: %s \n", SDL_GetError());
		SDL_Quit();
		return;
		}

	gfx_NN = SDL_CreateRenderer(win_NN, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (gfx_NN == NULL)
		{
		SDL_DestroyWindow(win_NN);
		printf("SDL_CreateRenderer Error: %s \n", SDL_GetError());
		SDL_Quit();
		return;
		}
	}

void plot_NN(NNET *net)
	{
	SDL_SetRenderDrawColor(gfx_NN, 0, 0, 0, 0xFF);
	SDL_RenderClear(gfx_NN);		//Clear screen

	#define Volume 20.0f

	#define numLayers (net->numLayers)
	for (int l = 0; l < numLayers - 1; l++)
		{
		double gain = 1.0f;
		// increase amplitude for hidden layers
		if (l > 0 && l < numLayers - 1)
			gain = 5.0f;
		else
			gain = 1.0f;

		// set color
		float r = ((float) l ) / numLayers;
		float b = 1.0f - ((float) l ) / numLayers;
		SDL_SetRenderDrawColor(gfx_NN, f2i(r), 0x40, f2i(b), 0xFF);

		int nn = net->layers[l].numNeurons;

		int neuronWidth = (NN_box_width - 20) / (nn - 1);
		
		// draw baseline
		#define Y_step ((NN_box_height - (int) Volume * 10) / (numLayers - 2))
		int baseline_y = (int) Volume * 5 + l * Y_step;
		SDL_RenderDrawLine(gfx_NN, 10, baseline_y, \
			neuronWidth * (nn - 1) + 10, baseline_y);

		SDL_SetRenderDrawColor(gfx_NN, f2i(r), 0x70, f2i(b), 0xFF);
		
		for (int n = 1; n < nn; n++)
			{
			double output0 = Volume * gain * net->layers[l].neurons[n - 1].output;
			double output1 = Volume * gain * net->layers[l].neurons[n].output;
			
			int basepoint_x = 10 + neuronWidth * n;
			SDL_RenderDrawLine(gfx_NN, basepoint_x - neuronWidth, baseline_y - output0, \
				basepoint_x, baseline_y - output1);
			}
		}

	SDL_RenderPresent(gfx_NN);
	}

// Older version with vertical lines, suitable for many layers
void plot_NN_old(NNET *net)
	{
	SDL_SetRenderDrawColor(gfx_NN, 0, 0, 0, 0xFF);
	SDL_RenderClear(gfx_NN);		//Clear screen

	#define Volume 20.0f
	#define NeuronWidth 20

	#define numLayers (net->numLayers)
	for (int l = 0; l < numLayers; l++)
		{
		double gain = 1.0f;
		// increase amplitude for hidden layers
		if (l > 0 && l < numLayers - 1)
			gain = 5.0f;
		else
			gain = 1.0f;

		// set color
		float r = ((float) l ) / numLayers;
		float b = 1.0f - ((float) l ) / numLayers;
		SDL_SetRenderDrawColor(gfx_NN, f2i(r), 0x60, f2i(b), 0xFF);

		int nn = net->layers[l].numNeurons;

		// draw baseline
		#define X_step ((NN_box_width - 20 - nn * NeuronWidth) / (numLayers - 1))
		int baseline_x = 10 + l * X_step;
		#define Y_step ((NN_box_height - (int) Volume * 14) / (numLayers - 1))
		int baseline_y = (int) Volume * 7 + l * Y_step;
		SDL_RenderDrawLine(gfx_NN, baseline_x, baseline_y, \
			baseline_x + nn * NeuronWidth, baseline_y);

		SDL_SetRenderDrawColor(gfx_NN, f2i(r), 0xB0, f2i(b), 0xFF);
		
		for (int n = 0; n < nn; n++)
			{
			double output = gain * net->layers[l].neurons[n].output;
			
			int basepoint_x = baseline_x + NeuronWidth * n;
			SDL_RenderDrawLine(gfx_NN, basepoint_x, baseline_y, \
				basepoint_x, baseline_y - output * Volume);
			}
		}

	SDL_RenderPresent(gfx_NN);
	}

// *************************** Seh's NN visualizer *************************************

#define NN2_box_width 150
#define NN2_box_height 400

void start_NN2_plot(void)
	{

	if (SDL_Init(SDL_INIT_VIDEO) != 0)
		{
		printf("SDL_Init Error: %s \n", SDL_GetError());
		return;
		}

	win_NN2 = SDL_CreateWindow("NN activity", 800, 650, NN2_box_width, NN2_box_height, SDL_WINDOW_SHOWN);
	if (win_NN2 == NULL)
		{
		printf("SDL_CreateWindow Error: %s \n", SDL_GetError());
		SDL_Quit();
		return;
		}

	gfx_NN2 = SDL_CreateRenderer(win_NN2, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (gfx_NN2 == NULL)
		{
		SDL_DestroyWindow(win_NN2);
		printf("SDL_CreateRenderer Error: %s \n", SDL_GetError());
		SDL_Quit();
		return;
		}
	}

void rectI(int x, int y, int w, int h, int r, int g, int b)
	{
	SDL_Rect fillRect = {x, y, w, h};
	SDL_SetRenderDrawColor(gfx_NN2, r, g, b, 0xFF);
	SDL_RenderFillRect(gfx_NN2, &fillRect);
	}

void rect(int x, int y, int w, int h, float r, float g, float b)
	{
	rectI(x, y, w, h, f2i(r), f2i(g), f2i(b));
	}

void plot_NN2(NNET *net)
	{
	SDL_SetRenderDrawColor(gfx_NN2, 0, 0, 0, 0xFF);
	SDL_RenderClear(gfx_NN2);		//Clear screen

	int bwh = 20; /* neuron block width,height*/
	#define numLayers (net->numLayers)
	#define L_margin ((NN2_box_width - (numLayers - 1) * bwh) / 2)
	#define T_margin ((NN2_box_height - nn * bwh) / 2)
	
	for (int l = 0; l < numLayers - 1; l++)
		{
		int nn = net->layers[l].numNeurons;
		for (int n = 0; n < nn; n++)
			{
			NEURON neuron = net->layers[l].neurons[n];
			double output = neuron.output;

			float r = output < 0 ? -output : 0;
			if (r < -1) r = -1;

			float g = output > 0 ? output : 0;
			if (g < +1) g = +1;

			float b = neuron.input;

			rect(L_margin + l*bwh, T_margin + n*bwh, bwh, bwh, r, g, b);
			}
		}

	SDL_RenderPresent(gfx_NN2);
	}

//******************************* K vector visualizer ******************************

#define K_box_width 600
#define K_box_height 300

void start_K_plot(void)
	{

	if (SDL_Init(SDL_INIT_VIDEO) != 0)
		{
		printf("SDL_Init Error: %s \n", SDL_GetError());
		return;
		}

	win_K = SDL_CreateWindow("K vector", 400, 200, K_box_width, K_box_height, SDL_WINDOW_SHOWN);
	if (win_K == NULL)
		{
		printf("SDL_CreateWindow Error: %s \n", SDL_GetError());
		SDL_Quit();
		return;
		}

	gfx_K = SDL_CreateRenderer(win_K, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (gfx_K == NULL)
		{
		SDL_DestroyWindow(win_K);
		printf("SDL_CreateRenderer Error: %s \n", SDL_GetError());
		SDL_Quit();
		return;
		}
	}

void line(int x1, double y1, int x2, double y2)
	{
	#define TopX 20
	#define TopY (K_box_height / 2)
	SDL_RenderDrawLine(gfx_K, x1 + TopX, (int) y1 + TopY, x2 + TopX, (int) y2 + TopY);
	}

// Show components of K vector as a line graph
extern double K[];
int plot_K(int delay)
	{
	const Uint8 *keys = SDL_GetKeyboardState(NULL);		// keyboard states

	//Clear screen
	SDL_SetRenderDrawColor(gfx_K, 0, 0, 0, 0xFF);
	SDL_RenderClear(gfx_K);

	// Draw base line
	#define K_Width ((K_box_width - TopX * 2) / dim_K)
	SDL_SetRenderDrawColor(gfx_K, 0xFF, 0x00, 0x00, 0xFF);
	SDL_RenderDrawLine(gfx_K, 0, TopY, K_box_width, TopY);

	SDL_SetRenderDrawColor(gfx_K, 0x1E, 0xD3, 0xEB, 0xFF);
	#define Amplitude 20.0f
	for (int k = 1; k < dim_K; ++k)
		line(k * K_Width,		 Amplitude * K[k - 1],
			(k + 1) * K_Width, Amplitude * K[k]);

	SDL_RenderPresent(gfx_K);
	SDL_Delay(delay);

	// Read keyboard state, if "Q" is pressed, return 1
	SDL_PumpEvents();
    if (keys[SDL_SCANCODE_Q])
		return 1;
	else
		return 0;
	}

void plot_trainer(double val)
	{
	int y = (int) (Amplitude * val);

	SDL_SetRenderDrawColor(gfx_K, 0xEB, 0xCC, 0x1E, 0xFF);
	SDL_Rect fillRect = {TopX, y + TopY, 5, -y};
	SDL_RenderFillRect(gfx_K, &fillRect);
	SDL_RenderPresent(gfx_K);
	SDL_Delay(70 /* milliseconds */);
	}

void pause_graphics()
	{
	const Uint8 *keys = SDL_GetKeyboardState(NULL);		// keyboard states
	bool quit = NULL;
	SDL_Event e;

	//Update screen
	SDL_RenderPresent(gfx_K);

	//While application is running
	while (!quit)
		{
		//Handle events on queue
		while (SDL_PollEvent(&e) != 0)
			{
			//User requests quit
			if (e.type == SDL_QUIT)			// This seems to be close-window event
				quit = true;
			if (keys[SDL_SCANCODE_Q])		// press 'Q' to quit
				quit = true;
			}
		}
	SDL_DestroyRenderer(gfx_NN);
	SDL_DestroyRenderer(gfx_NN2);
	SDL_DestroyRenderer(gfx_K);

	SDL_DestroyWindow(win_NN);
	SDL_DestroyWindow(win_NN2);
	SDL_DestroyWindow(win_K);

	SDL_Quit();
	}

void quit_graphics()
	{
	SDL_DestroyRenderer(gfx_NN);
	SDL_DestroyRenderer(gfx_NN2);
	SDL_DestroyRenderer(gfx_K);

	SDL_DestroyWindow(win_NN);
	SDL_DestroyWindow(win_NN2);
	SDL_DestroyWindow(win_K);

	SDL_Quit();
	}

/* ************************ old code below, just for testing ************************

void test_rectangles()	// old code
	{
	bool quit = NULL;
	SDL_Event e;

	//While application is running
	while (!quit)
		{
		//Handle events on queue
		while (SDL_PollEvent(&e) != 0)
			{
			//User requests quit
			if (e.type == SDL_QUIT)
				{
				quit = true;
				}
			//Render red filled quad
			#define SCREEN_WIDTH 800
			#define SCREEN_HEIGHT 900
			SDL_Rect fillRect = {SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2};
			SDL_SetRenderDrawColor(gfx, 0xFF, 0x00, 0x00, 0xFF);
			SDL_RenderFillRect(gfx, &fillRect);
			}

		//Clear screen
		SDL_SetRenderDrawColor(gfx, 0xFF, 0xFF, 0xFF, 0xFF);
		SDL_RenderClear(gfx);

		//Render red filled quad
		SDL_Rect fillRect = {SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2};
		SDL_SetRenderDrawColor(gfx, 0xFF, 0x00, 0x00, 0xFF);
		SDL_RenderFillRect(gfx, &fillRect);

		//Render green outlined quad
		SDL_Rect outlineRect = {SCREEN_WIDTH / 6, SCREEN_HEIGHT / 6, SCREEN_WIDTH * 2 / 3, SCREEN_HEIGHT * 2 / 3};
		SDL_SetRenderDrawColor(gfx, 0x00, 0xFF, 0x00, 0xFF);
		SDL_RenderDrawRect(gfx, &outlineRect);

		//Draw blue horizontal line
		SDL_SetRenderDrawColor(gfx, 0x00, 0x00, 0xFF, 0xFF);
		SDL_RenderDrawLine(gfx, 0, SCREEN_HEIGHT / 2, SCREEN_WIDTH, SCREEN_HEIGHT / 2);

		//Draw vertical line of purple dots
		SDL_SetRenderDrawColor(gfx, 0x44, 0x00, 0x44, 0xFF);
		for (int i = 0; i < SCREEN_HEIGHT; i += 4)
			{
			SDL_RenderDrawPoint(gfx, SCREEN_WIDTH / 2, i);
			}

		//Update screen
		SDL_RenderPresent(gfx);
		}
	}

int test_SDL()		// old code, just to test if it works
	{
	if (SDL_Init(SDL_INIT_VIDEO) != 0)
		{
		printf("SDL_Init Error: %s \n", SDL_GetError());
		return 1;
		}

	win = SDL_CreateWindow("Hello World!", 100, 100, 640, 480, SDL_WINDOW_SHOWN);
	if (win == NULL)
		{
		printf("SDL_CreateWindow Error: %s \n", SDL_GetError());
		SDL_Quit();
		return 1;
		}

	gfx = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (gfx == NULL)
		{
		SDL_DestroyWindow(win);
		printf("SDL_CreateRenderer Error: %s \n", SDL_GetError());
		SDL_Quit();
		return 1;
		}

	SDL_Surface *bmp = SDL_LoadBMP("hello.bmp");
	if (bmp == NULL)
		{
		SDL_DestroyRenderer(gfx);
		SDL_DestroyWindow(win);
		printf("SDL_LoadBMP Error: %s \n", SDL_GetError());
		SDL_Quit();
		return 1;
		}

	SDL_Texture *tex = SDL_CreateTextureFromSurface(gfx, bmp);
	SDL_FreeSurface(bmp);
	if (tex == NULL)
		{
		SDL_DestroyRenderer(gfx);
		SDL_DestroyWindow(win);
		printf("SDL_CreateTextureFromSurface Error: %s \n", SDL_GetError());
		SDL_Quit();
		return 1;
		}

	//A sleepy rendering loop, wait for 3 seconds and render and present the screen each time
	for (int i = 0; i < 3; ++i)
		{
		//First clear the renderer
		SDL_RenderClear(gfx);
		//Draw the texture
		SDL_RenderCopy(gfx, tex, NULL, NULL);
		//Update the screen
		SDL_RenderPresent(gfx);
		//Take a quick break after all that hard work
		SDL_Delay(1000);
		}

	SDL_DestroyTexture(tex);
	SDL_DestroyRenderer(gfx);
	SDL_DestroyWindow(win);
	SDL_Quit();

	return 0;
	}

*/