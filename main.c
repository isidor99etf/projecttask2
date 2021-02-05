#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void getMatrix(double** matrix, size_t size) {
	FILE *fp;
	fp = fopen("./../data.csv", "r");
	if (fp) {
		for (int i = 0; i < size; ++i) {
			matrix[i] = (double *) calloc(size, sizeof(double));
			for (int j = 0; j <= size - 2; j += 2) {
				int t = fscanf(fp, "%lf,%lf\n", &matrix[i][j], &matrix[i][j + 1]);
			}
		}
		fclose(fp);
	}
}

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


void startNormal(double** matrix, size_t size) {

	printf("Start Normal\n");
	double t_start = omp_get_wtime() * 1000;

	double x_mean = 0, y_mean = 0;
	for (int j = 0; j< size; ++j)
		for (int i = 0; i < size; ++i) 
			if (j % 2 == 0)
				x_mean += matrix[i][j];
			else 
				y_mean += matrix[i][j];
	
	x_mean /= (size * size / 2);
	y_mean /= (size * size / 2);

	double SS_xx = 0;
	double SS_xy = 0;
	/*for (int i = 0; i < size; ++i)
		for (int j = 0; j <= size - 2; j += 2) {
				SS_xx += (matrix[i][j] - x_mean) * (matrix[i][j] - x_mean); 
				SS_xy += (matrix[i][j] - x_mean) * (matrix[i][j + 1] - y_mean);
		}*/

	for (int j = 0; j <= size - 2; j += 2)
		for (int i = 0; i < size; ++i) {
				SS_xx += (matrix[i][j] - x_mean) * (matrix[i][j] - x_mean); 
				SS_xy += (matrix[i][j] - x_mean) * (matrix[i][j + 1] - y_mean);
		}

	double a = SS_xy / SS_xx;
	double b = y_mean - a * x_mean;

	double t_end = omp_get_wtime() * 1000 - t_start;
	printf("Time: %.2lf ms\n", t_end);
	printf("Y = %.5lf + %.5lf * X\n", b, a);
}

/*void startNormal(double* X, double* Y, size_t size) {

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
}*/


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

	// size_t size = 100000000;
	size_t size = 10000;
	size_t size2 = 50000000;

	double** matrix = (double **) calloc(size, sizeof(double *));
	double* X = (double *) calloc(size2, sizeof(double));
	double* Y = (double *) calloc(size2, sizeof(double));

	double t_start = omp_get_wtime() * 1000;
	getMatrix(matrix, size);
	getDataFromFile(X, Y, size2);
	double t_end = omp_get_wtime() * 1000 - t_start;
	printf("Time: %.2lf ms\n", t_end);

	printf("Data loaded!\n");

	startNormal(matrix, size);
	startOpenMp(X, Y, size2);

	/*for (int i = 0; i < 10; ++i) {
		printf("-------------------------------  %d  ---------------------------------\n", i + 1);
		// startNormal(X, Y, size);
		// printf("--------------------------------------------------------------------\n");
		startOpenMp(X, Y, size);
		printf("--------------------------------------------------------------------\n");
	}*/
}