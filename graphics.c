#include <stdbool.h>
#include <SDL2/SDL.h>

#include "RNN.h"

SDL_Renderer *newWindow(void) {

    bool loadMedia()
    {
        //Loading success flag
        bool success = true;

        //Nothing to load
        return success;
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        printf("SDL_Init Error: %s \n", SDL_GetError());
        return NULL;
    }

    SDL_Window *win = SDL_CreateWindow("?", 10, 10, 800, 900, SDL_WINDOW_SHOWN);
    if (win == NULL)
    {
        printf("SDL_CreateWindow Error: %s \n", SDL_GetError());
        SDL_Quit();
        return NULL;
    }

    SDL_Renderer *gRenderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (gRenderer == NULL)
    {
        SDL_DestroyWindow(win);
        printf("SDL_CreateRenderer Error: %s \n", SDL_GetError());
        SDL_Quit();
        return NULL;
    }

    return gRenderer;

}

#define f2i(v) ((int)(256.0*v))

void rectI(SDL_Renderer* gfx, int x, int y, int w, int h, int r, int g, int b) {
    SDL_Rect fillRect = { x, y, w, h };
    SDL_SetRenderDrawColor( gfx, r, g, b, 0xFF );
    SDL_RenderFillRect( gfx, &fillRect );
}

void rect(SDL_Renderer* gfx, int x, int y, int w, int h, float r, float g, float b) {
    rectI(gfx, x, y, w, h, f2i(r), f2i(g), f2i(b) );
}

void drawNetwork(NNET *net, SDL_Renderer* gRenderer) {
   //Clear screen
    SDL_SetRenderDrawColor( gRenderer, 0, 0, 0, 0xFF );
    SDL_RenderClear( gRenderer );

    int bwh = 20; /* neuron block width,height*/

    int numLayers = net->numLayers;
    for (int l = 0; l < numLayers; l++) {
        int nn = net->layers[l].numNeurons;
        for (int n = 0; n < nn; n++) {

            NEURON neuron = net->layers[l].neurons[n];
            double output = neuron.output;

                float r = output < 0 ? -output : 0;
                if (r < -1) r = -1;

                float g = output > 0 ? output : 0;
                if (r < +1) r = +1;

                float b = neuron.input;

            rect(gRenderer, l*bwh, n*bwh, bwh, bwh, r, g, b);
        }
    }

    SDL_RenderPresent( gRenderer );


}

void plot_rectangles(SDL_Renderer* gRenderer)
{
    bool quit;
    SDL_Event e;

    //While application is running
    while ( !quit )
    {
        //Handle events on queue
        while ( SDL_PollEvent( &e ) != 0 )
        {
            //User requests quit
            if ( e.type == SDL_QUIT )
            {
                quit = true;
            }//Render red filled quad
			#define SCREEN_WIDTH 800
			#define SCREEN_HEIGHT 900
            SDL_Rect fillRect = { SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 };
            SDL_SetRenderDrawColor( gRenderer, 0xFF, 0x00, 0x00, 0xFF );
            SDL_RenderFillRect( gRenderer, &fillRect );
        }

        //Clear screen
        SDL_SetRenderDrawColor( gRenderer, 0xFF, 0xFF, 0xFF, 0xFF );
        SDL_RenderClear( gRenderer );

        //Render red filled quad
        SDL_Rect fillRect = { SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 };
        SDL_SetRenderDrawColor( gRenderer, 0xFF, 0x00, 0x00, 0xFF );
        SDL_RenderFillRect( gRenderer, &fillRect );

        //Render green outlined quad
        SDL_Rect outlineRect = { SCREEN_WIDTH / 6, SCREEN_HEIGHT / 6, SCREEN_WIDTH * 2 / 3, SCREEN_HEIGHT * 2 / 3 };
        SDL_SetRenderDrawColor( gRenderer, 0x00, 0xFF, 0x00, 0xFF );
        SDL_RenderDrawRect( gRenderer, &outlineRect );

        //Draw blue horizontal line
        SDL_SetRenderDrawColor( gRenderer, 0x00, 0x00, 0xFF, 0xFF );
        SDL_RenderDrawLine( gRenderer, 0, SCREEN_HEIGHT / 2, SCREEN_WIDTH, SCREEN_HEIGHT / 2 );

        //Draw vertical line of purple dots
        SDL_SetRenderDrawColor( gRenderer, 0x44, 0x00, 0x44, 0xFF );
        for( int i = 0; i < SCREEN_HEIGHT; i += 4 )
        {
            SDL_RenderDrawPoint( gRenderer, SCREEN_WIDTH / 2, i );
        }

        //Update screen
        SDL_RenderPresent( gRenderer );
    }
}

int test_SDL()
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        printf("SDL_Init Error: %s \n", SDL_GetError());
        return 1;
    }

    SDL_Window *win = SDL_CreateWindow("Hello World!", 100, 100, 640, 480, SDL_WINDOW_SHOWN);
    if (win == NULL)
    {
        printf("SDL_CreateWindow Error: %s \n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (ren == NULL)
    {
        SDL_DestroyWindow(win);
        printf("SDL_CreateRenderer Error: %s \n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Surface *bmp = SDL_LoadBMP("hello.bmp");
    if (bmp == NULL)
    {
        SDL_DestroyRenderer(ren);
        SDL_DestroyWindow(win);
        printf("SDL_LoadBMP Error: %s \n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Texture *tex = SDL_CreateTextureFromSurface(ren, bmp);
    SDL_FreeSurface(bmp);
    if (tex == NULL)
    {
        SDL_DestroyRenderer(ren);
        SDL_DestroyWindow(win);
        printf("SDL_CreateTextureFromSurface Error: %s \n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    //A sleepy rendering loop, wait for 3 seconds and render and present the screen each time
    for (int i = 0; i < 3; ++i)
    {
        //First clear the renderer
        SDL_RenderClear(ren);
        //Draw the texture
        SDL_RenderCopy(ren, tex, NULL, NULL);
        //Update the screen
        SDL_RenderPresent(ren);
        //Take a quick break after all that hard work
        SDL_Delay(1000);
    }

    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();

    return 0;
}

