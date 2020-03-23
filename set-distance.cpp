#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <random>
#include <algorithm>		// random_shuffle

extern void pause_graphics();
extern void quit_graphics();
extern void start_NN_plot(void);
extern void start_NN2_plot(void);
extern void start_W_plot(void);
extern void start_K_plot(void);
extern void start_output_plot(void);
extern void start_LogErr_plot(void);
extern void restart_LogErr_plot(void);
extern void plot_LogErr(double, double);
extern void flush_output();
extern void plot_tester(double, double);
extern void plot_K();
extern int delay_vis(int);
extern void plot_trainer(double);
extern void plot_ideal(void);
extern void beep(void);
extern double sigmoid(double);
extern void start_timer(), end_timer(char *);

#define N	2

double random01()
	{
	return rand() / (float) RAND_MAX;
	}

double set_distance(double x1[], double x2[])
	{
	double sum = 0.0;

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
			sum += abs(x1[i] - x2[j]) * 2 \
				- abs(x1[i] - x1[j]) \
				- abs(x2[i] - x2[j]);
	double dx = sum / (N * N);
	return dx;
	}

double set_distance2(double x1[], double x2[])
	{
	double sum = 0.0;

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
			sum += 2 * pow(x1[i] - x2[j], 2) \
				- pow(x1[i] - x1[j], 2) \
				- pow(x2[i] - x2[j], 2);
	double dx = sqrt(sum / N) / 2.0;
	return dx;
	}

// ****** The "standard" Euclidean distance between 2 points
// d(x,y) = sqrt((x1 - y1)^2 + (x2 - y2)^2)
double distance(double y1[], double y2[])
	{
	double sum = 0.0;

	for (int i = 0; i < N; ++i)
		sum += (y1[i] - y2[i]) * (y1[i] - y2[i]);
	double dy = sqrt(sum);
	return dy;
	}

int main()
	{
	double x1[N], x2[N], dx, dy, dx_dy;
	double max_dx, max_dy, max_dx_dy = 0.0;
	bool remarkable;

	srand(time(NULL));				// random seed

	// Generate and test random pairs of points
	for (int i = 0; i < 10000; ++i)
		{
		for (int j = 0; j < N; ++j)
			{
			x1[j] = random01();
			x2[j] = random01();
			}

		std::random_shuffle(x1, x1 + N);

		dx = set_distance2(x1, x2);
		dy = distance(x1, x2);
		dx_dy = dx / dy;

		remarkable = false;

		if (dx > max_dx)
			{
			remarkable = true;
			max_dx = dx;
			}
		if (dy > max_dy)
			{
			remarkable = true;
			max_dy = dy;
			}
		if (dx_dy > max_dx_dy)
			{
			max_dx_dy = dx_dy;
			}

		// if (dx > 0.6)
		//	remarkable = true;
		if (dx > dy)
			remarkable = true;
		if (dy > 0.99)
			remarkable = true;
		// if (dx_dy > 0.99999)
		//	remarkable = true;

		if (remarkable)
			{
			printf("\ndx=%f%c\t", dx, dx > 1.0 ? 'x' : ' ');
			printf("dy=%f\t", dy);
			printf("dx:dy=%f%c\n", dx / dy, dy > dx ? '+' : ' ');
			}
		else
			printf(".");
		}
	printf("max dx = %f\n", max_dx);
	printf("max dy = %f\n", max_dy);
	printf("max dx:dy = %f\n", max_dx_dy);
	}
