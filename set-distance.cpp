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

int N = 3;

double random01()
	{
	return rand() / (float) RAND_MAX;
	}

// In the current interpretation, each point is a set
// We want to measure the distance between 2 points as sets and also as lists
// The distance between 2 lists, where each "coordinate" belongs to one dimension, is the
// "standard" Euclidean distance:
//      d(x,y) = sqrt((x1 - y1)^2 + (x2 - y2)^2)
double distance_Eu(double x[], double y[])
	{
	double sum = 0.0;

	for (int i = 0; i < N; ++i)
		sum += pow(x[i] - y[i], 2);

	return sqrt(sum);
	}

double distance_abs(double x[], double y[])
	{
	double sum = 0.0;

	for (int i = 0; i < N; ++i)
		sum += abs(x[i] - y[i]);

	return sum / N;
	}

double set_distance_Eu(double x[], double y[])
	{
	double sum, sum1, sum2 = 0.0;

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
			sum += pow(x[i] - y[j], 2);

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
			sum1 += pow(x[i] - x[j], 2);

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
			sum2 += pow(y[i] - y[j], 2);

	return (sqrt(sum / N) - sqrt(sum1 / N) - sqrt(sum2 / N));
	}

double set_distance_Eu2(double x[], double y[])
	{
	double sum = 0.0;

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
			sum += 2 * pow(x[i] - y[j], 2) \
				- pow(x[i] - x[j], 2) \
				- pow(y[i] - y[j], 2);

	return sqrt(sum / N);
	}

double set_distance_abs(double x[], double y[])
	{
	double sum = 0.0;

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
			sum += abs(x[i] - y[j]) * 2 \
				- abs(x[i] - x[j]) \
				- abs(y[i] - y[j]);

	return sum / (N * N);
	}

void print_x(double x[])
	{
	printf("[");
	for (int k = 0; k < N; ++k)
		if (k > 0)
			printf(",%f", x[k]);
		else
			printf("%f", x[k]);
	printf("]");
	}

int main(int argc, char **argv)
	{
	void test_1(), test_2(), test_3();

	if (argc == 2)
		N = std::stoi(argv[1]);

	srand(time(NULL));				// random seed

	test_3();
	}

// Test that the set distance is always less than or equal to the Euclidean distance.
// This ratio approaches the maximum value of 1 as more and more pairs are tested.
void test_3()
	{
	double x[N], y[N], d1, d2, r;
	double max_d1, max_d2, max_r = 0.0;
	bool remarkable;

	// Generate and test random pairs of points
	while (true)
		{
		for (int j = 0; j < N; ++j)
			{
			x[j] = random01();
			y[j] = random01();
			}

		d1 = set_distance_Eu(x, y);
		d2 = distance_Eu(x, y);
		r = d1 / d2;

		remarkable = false;

		if (d1 > max_d1)
			{
			// remarkable = true;
			max_d1 = d1;
			}
		if (d2 > max_d2)
			{
			// remarkable = true;
			max_d2 = d2;
			}
		if (r > max_r)
			{
			max_r = r;
			printf("d1:d2 = %f\t", max_r);
			print_x(x); printf("\t");
			print_x(y); printf("\n");
			}

		// if (d1 > 0.6)
		//	remarkable = true;
		// if (d1 > d2)
		//	remarkable = true;
		// if (d2 > 0.99)
		//	remarkable = true;
		// if (d1_d2 > 0.99999)
		//	remarkable = true;

		if (remarkable)
			{
			printf("\nd1=%f%c\t", d1, d1 > 1.0 ? 'x' : ' ');
			printf("d2=%f\t", d2);
			printf("d1:d2=%f%c\n", r, d2 > d1 ? '+' : ' ');
			}
		// else
		//	printf(".");
		}
	}

// Randomly permute (x, ..., xn) and check if their set distances are invariant
void test_2()
	{
	double x[N], y[N], d1, d2, delta;
	double max_delta = 0.0;

	// Generate and test random pairs of points
	while (true)
		{
		for (int j = 0; j < N; ++j)
			{
			x[j] = random01();
			y[j] = random01();
			}
		// print_x(x);  printf("\t");
		// print_x(y);

		d1 = set_distance_Eu(x, y);

		std::random_shuffle(x, x + N);

		// print_x(x);  printf("\t");
		// print_x(y);

		d2 = set_distance_Eu(x, y);

		delta = d1 - d2;
		if (delta > max_delta)
			{
			max_delta = delta;
			printf("max ∆ = %f\n", max_delta);
			}
		}
	}

// Test that the maximal Euclidean distance between 2 points in the unit hypercube is √n
void test_1()
	{
	double x[N], y[N], d2;
	double max_d2 = 0.0;

	while (true)
		{
		for (int j = 0; j < N; ++j)
			{
			x[j] = random01();
			y[j] = random01();
			}
		d2 = distance_Eu(x, y);
		if (d2 > max_d2)
			{
			max_d2 = d2;
			printf("max d2 = %f\n", max_d2);	
			}
		}
	}
