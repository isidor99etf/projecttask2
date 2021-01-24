#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void getDataFromFile(double* X, double* Y, size_t size) {

	FILE* fp;
	fp = fopen("./../data.csv", "r");
	if (fp) {
		for (int i = 0; i < size; ++i) {
			int t = fscanf(fp, "%lf,%lf\n", &X[i], &Y[i]);
		}
		fclose(fp);
	}
}

void startNormal(double* X, double* Y, size_t size) {

	printf("Start Normal\n");
	double t_start = omp_get_wtime() * 1000;

	double x_mean = 0, y_mean = 0;
	for (int i = 0; i < size; ++i) {
		x_mean += X[i];
		y_mean += Y[i];
	}
	x_mean /= size;
	y_mean /= size;

	double SS_xx = 0;
	double SS_xy = 0;
	for (int i = 0; i < size; ++i) {
		SS_xx += (X[i] - x_mean) * (X[i] - x_mean);
		SS_xy += (X[i] - x_mean) * (Y[i] - y_mean);
	}

	double a = SS_xy / SS_xx;
	double b = y_mean - a * x_mean;

	double t_end = omp_get_wtime() * 1000 - t_start;
	printf("Time: %.2lf ms\n", t_end);
	printf("Y = %.5lf + %.5lf * X\n", b, a);
}


void startOpenMp(double* X, double* Y, size_t size) {
	
	printf("Start OpenMP\n");
	
	double t_start = omp_get_wtime() * 1000;

	double x_mean = 0, y_mean = 0;
	#pragma omp parallel num_threads(4)
	{
		#pragma omp for reduction (+:x_mean,y_mean)
		for (int i = 0; i < size; ++i) {
			x_mean += X[i];
			y_mean += Y[i];
		}
	}
	x_mean /= size;
	y_mean /= size;

	double SS_xx = 0;
	double SS_xy = 0;
	#pragma omp parallel num_threads(4)
	{
		#pragma omp for reduction (+:SS_xx,SS_xy)
		for (int i = 0; i < size; ++i) {
			SS_xx += (X[i] - x_mean) * (X[i] - x_mean);
			SS_xy += (X[i] - x_mean) * (Y[i] - y_mean);
		}
	}
	
	double a = SS_xy / SS_xx;
	double b = y_mean - a * x_mean;

	double t_end = omp_get_wtime() * 1000 - t_start;

	printf("Time: %.2lf ms\n", t_end);
	printf("Y = %.5lf + %.5lf * X\n", b, a);
} 

int main() {

	size_t size = 100000000;

	double* X = (double *) calloc(size, sizeof(double));
	double* Y = (double *) calloc(size, sizeof(double));

	double t_start = omp_get_wtime() * 1000;
	getDataFromFile(X, Y, size);
	double t_end = omp_get_wtime() * 1000 - t_start;
	printf("Time: %.2lf ms\n", t_end);

	printf("Data loaded!\n");

	for (int i = 0; i < 10; ++i) {
		printf("-------------------------------  %d  ---------------------------------\n", i + 1);
		// startNormal(X, Y, size);
		// printf("--------------------------------------------------------------------\n");
		startOpenMp(X, Y, size);
		printf("--------------------------------------------------------------------\n");
	}
}