#include <stdbool.h>
#include <SDL2/SDL.h>

#include "RNN.h"

SDL_Renderer *gfx;
SDL_Window *win;

void init_graphics(void)
	{

	if (SDL_Init(SDL_INIT_VIDEO) != 0)
		{
		printf("SDL_Init Error: %s \n", SDL_GetError());
		return;
		}

	win = SDL_CreateWindow("?", 10, 300, 600, 400, SDL_WINDOW_SHOWN);
	if (win == NULL)
		{
		printf("SDL_CreateWindow Error: %s \n", SDL_GetError());
		SDL_Quit();
		return;
		}

	gfx = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (gfx == NULL)
		{
		SDL_DestroyWindow(win);
		printf("SDL_CreateRenderer Error: %s \n", SDL_GetError());
		SDL_Quit();
		return;
		}
	}

void pause_graphics()
	{
	bool quit = NULL;
	SDL_Event e;

	//Update screen
	SDL_RenderPresent(gfx);

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
			}
		}
	SDL_DestroyRenderer(gfx);
	SDL_DestroyWindow(win);
	SDL_Quit();
	}

void rectI(int x, int y, int w, int h, int r, int g, int b)
	{
	SDL_Rect fillRect = {x, y, w, h};
	SDL_SetRenderDrawColor(gfx, r, g, b, 0xFF);
	SDL_RenderFillRect(gfx, &fillRect);
	}

void rect(int x, int y, int w, int h, float r, float g, float b)
	{
	#define f2i(v) ((int)(256.0f * v))
	rectI(x, y, w, h, f2i(r), f2i(g), f2i(b));
	}

void drawNetwork(NNET *net)
	{
	SDL_SetRenderDrawColor(gfx, 0, 0, 0, 0xFF);
	SDL_RenderClear(gfx);		//Clear screen

	int bwh = 20; /* neuron block width,height*/

	#define numLayers (net->numLayers)
	for (int l = 0; l < numLayers; l++)
		{
		int nn = net->layers[l].numNeurons;
		for (int n = 0; n < nn; n++)
			{
			NEURON neuron = net->layers[l].neurons[n];
			double output = neuron.output;

			float r = output < 0 ? -output : 0;
			if (r < -1) r = -1;

			float g = output > 0 ? output : 0;
			if (r < +1) r = +1;

			float b = neuron.input;

			rect(l*bwh, n*bwh, bwh, bwh, r, g, b);
			}
		}

	SDL_RenderPresent(gfx);
	}

void line(int x1, double y1, int x2, double y2)
	{
	#define TopX 50
	#define TopY 200
	SDL_RenderDrawLine(gfx, x1 + TopX, (int) y1 + TopY, x2 + TopX, (int) y2 + TopY);
	}

// Show components of K vector as a line graph
extern double K[];
int plot_K()
	{
	const Uint8 *keys = SDL_GetKeyboardState(NULL);		// keyboard states

	SDL_SetRenderDrawColor(gfx, 0, 0, 0, 0xFF);
	SDL_RenderClear(gfx);		//Clear screen

	// Draw base line
	#define Width 30
	SDL_SetRenderDrawColor(gfx, 0xFF, 0, 0, 0xFF);
	line(0, 0, 12 * Width, 0);

	SDL_SetRenderDrawColor(gfx, 0, 0xFF, 0, 0xFF);
	#define Amplitude 20.0f
	for (int k = 1; k < dim_K; ++k)
		line(k * Width,		 Amplitude * K[k - 1],
			(k + 1) * Width, Amplitude * K[k]);

	SDL_RenderPresent(gfx);
	SDL_Delay(70 /* milliseconds */);

	// Read keyboard state, if "Q" is pressed, return 1
	SDL_PumpEvents();
    if (keys[SDL_SCANCODE_Q])
		return 1;
	else
		return 0;
	}

void test_rectangles()	// old code, just for testing
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

