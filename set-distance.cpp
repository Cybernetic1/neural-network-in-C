#include <iostream>
#include <cstdio>
#include <random>
#include <algorithm>		// random_shuffle

using namespace std;

int N = 3;

double random01()
	{
	return rand() * 2.0 / (float) RAND_MAX - 1.0;
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

// The set distance must satisfy 2 requirements simultaneously:
// 1) The distance should be 0 under permutations
// 2) The distance attains its maximum when 2 points are most dissimilar, and would equal the
//		Euclidean distance between them.
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

	return (2 * sqrt(sum / N) - sqrt(sum1 / N) - sqrt(sum2 / N)) / 2;
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
	void test_1(), test_2(), test_3(), test_4();
	int test_num;

	if (argc != 3)
		{
		printf("usage: set_distance <test #> <N>\n");
		printf("where <test #> = 1, 2, or 3\n");
		printf("      <N> = dimension of set vectors\n");
		printf("1. Test that the maximal Euclidean distance between 2 points in the unit hypercube is √n\n");
		printf("2. Randomly permute (x, ..., xn) and check if the set distances between the original\n");
		printf("\tand permuted sets (points) are 0.\n");
		printf("3. Test that the set distance is always less than or equal to the Euclidean distance\n");
		printf("\tThis ratio approaches the maximum value of 1 as more and more pairs are tested\n");
		printf("4. Manually test set distances\n");
		exit(0);
		}
	else
		{
		test_num = std::stoi(argv[1]);
		N = std::stoi(argv[2]);
		}

	srand(time(NULL));				// random seed

	switch (test_num)
		{
		case 1:
			test_1();
			break;
		case 2:
			test_2();
			break;
		case 3:
			test_3();
			break;
		case 4:
			test_4();
			break;
		}
	}

void test_3()
	{
	printf("Test that the set distance is always less than or equal to the Euclidean distance\n");
	printf("This ratio approaches the maximum value of 1 as more and more pairs are tested\n");

	double x[N], y[N], d1, d2, r;
	double max_d1, min_d1, max_r = 0.0;
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
			printf("max d1 = %f\n", max_d1);
			max_d1 = d1;
			}
		if (d1 < min_d1)
			{
			// remarkable = true;
			printf("min d1 = %f\n", min_d1);
			min_d1 = d1;
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

void test_2()
	{
	printf("Randomly permute (x, ..., xn) and check if the set distances between the original\n");
	printf("and permuted sets (points) are 0.\n");

	double x[N], y[N], d;
	double max_d, min_d = 0.0;

	// Generate a point and duplicate it
	while (true)
		{
		/*
		cout << "x=? ";
		for (int j = 0; j < N; ++j)
			cin >> x[j];

		cout << "y=? ";
		for (int j = 0; j < N; ++j)
			cin >> y[j];

		print_x(x);  printf("\t");
		print_x(y);  printf("\n");

		d = set_distance_Eu(x, y);
		printf("distance = %f\n", d);
		continue;
		*/

		for (int j = 0; j < N; ++j)
			{
			x[j] = random01();
			y[j] = x[j];
			}
		std::random_shuffle(y, y + N);
		// print_x(x);  printf("\t");
		// print_x(y);  printf("\n");

		d = set_distance_Eu(x, y);

		if (d > max_d)
			{
			max_d = d;
			printf("max distance = %f\n", max_d);
			}
		if (d < min_d)
			{
			min_d = d;
			printf("min distance = %f\n", min_d);
			}
		}
	}

void test_4()
	{
	printf("Manually test set distances\n");

	double x[N], y[N], d;
	double max_d, min_d = 0.0;

	while (true)
		{
		cout << "x=? ";
		for (int j = 0; j < N; ++j)
			cin >> x[j];

		cout << "y=? ";
		for (int j = 0; j < N; ++j)
			cin >> y[j];

		print_x(x);  printf("\t");
		print_x(y);  printf("\n");

		d = set_distance_Eu(x, y);
		printf("distance = %f\n", d);
		}
	}

void test_1()
	{
	printf("Test that the maximal Euclidean distance between 2 points in the unit hypercube is √n\n");

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
