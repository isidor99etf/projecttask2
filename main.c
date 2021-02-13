#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

/*
	* 
	* Read data from file
	*
*/
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

/*
	*
	* Without Cache Optimization
	*
*/
void startNormalWithoutCacheOptimization(double** matrix, size_t size, FILE* out) {

	fprintf(out, "Start Normal Without Cache Optimization\nNumber Of Vectors: %ld\n", (long) (size * size / 2));
	double t_start = omp_get_wtime() * 1000;

	double x_mean = 0, y_mean = 0;
	for (int j = 0; j < size; ++j)
		for (int i = 0; i < size; ++i) 
			if (j % 2 == 0)
				x_mean += matrix[i][j];
			else 
				y_mean += matrix[i][j];
	
	x_mean /= (size * size / 2);
	y_mean /= (size * size / 2);

	double SS_xx = 0;
	double SS_xy = 0;
	for (int j = 0; j <= size - 2; j += 2)
		for (int i = 0; i < size; ++i) {
				SS_xx += (matrix[i][j] - x_mean) * (matrix[i][j] - x_mean); 
				SS_xy += (matrix[i][j] - x_mean) * (matrix[i][j + 1] - y_mean);
		}

	double a = SS_xy / SS_xx;
	double b = y_mean - a * x_mean;

	double t_end = omp_get_wtime() * 1000 - t_start;
	fprintf(out, "Time: %.2lf ms\n", t_end);
	fprintf(out, "y = %.5lf * x + %.5lf\n", a, b);
}

/*
	*
	* With Cache Optimization
	*
*/
void startNormalWithCacheOptimization(double** matrix, size_t size, FILE* out) {

	fprintf(out, "Start Normal With Cache Optimization\nNumber Of Vectors: %ld\n", (long) (size * size / 2));
	double t_start = omp_get_wtime() * 1000;

	double x_mean = 0, y_mean = 0;
	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j) 
			if (j % 2 == 0)
				x_mean += matrix[i][j];
			else 
				y_mean += matrix[i][j];
	
	x_mean /= (size * size / 2);
	y_mean /= (size * size / 2);

	double SS_xx = 0;
	double SS_xy = 0;
	for (int i = 0; i < size; ++i)
		for (int j = 0; j <= size - 2; j += 2) {
				SS_xx += (matrix[i][j] - x_mean) * (matrix[i][j] - x_mean); 
				SS_xy += (matrix[i][j] - x_mean) * (matrix[i][j + 1] - y_mean);
		}

	double a = SS_xy / SS_xx;
	double b = y_mean - a * x_mean;

	double t_end = omp_get_wtime() * 1000 - t_start;
	fprintf(out, "Time: %.2lf ms\n", t_end);
	fprintf(out, "y = %.5lf * x + %.5lf\n", a, b);
}

/*
	*
	* OpenMP Without Cache Optimization
	*
*/
void startOpenMpWithoutCacheOptimization(double** matrix, size_t size, FILE* out) {
	
	fprintf(out, "Start OpenMP Without Cache Optimization\nNumber Of Vectors: %ld\n", (long) (size * size / 2));
	
	double t_start = omp_get_wtime() * 1000;

	double x_mean = 0, y_mean = 0;
	#pragma omp parallel num_threads(2)
	{
		#pragma omp for reduction (+:x_mean,y_mean)
		for (int j = 0; j < size; ++j) 
			for (int i = 0; i < size; ++i)
				if (j % 2 == 0)
					x_mean += matrix[i][j];
				else 
					y_mean += matrix[i][j];
		
	}

	x_mean /= (size * size / 2);
	y_mean /= (size * size / 2);

	double SS_xx = 0;
	double SS_xy = 0;
	#pragma omp parallel num_threads(2)
	{
		#pragma omp for reduction (+:SS_xx,SS_xy)
		for (int j = 0; j <= size - 2; j += 2) 
			for (int i = 0; i < size; ++i) {
				SS_xx += (matrix[i][j] - x_mean) * (matrix[i][j] - x_mean);
				SS_xy += (matrix[i][j] - x_mean) * (matrix[i][j + 1] - y_mean);
			}
	}
	
	double a = SS_xy / SS_xx;
	double b = y_mean - a * x_mean;

	double t_end = omp_get_wtime() * 1000 - t_start;

	fprintf(out, "Time: %.2lf ms\n", t_end);
	fprintf(out, "y = %.5lf * x + %.5lf\n", a, b);
} 

/*
	*
	*	OpenMP With Cache Optimization
	*
*/
void startOpenMpWithCacheOptimization(double** matrix, size_t size, FILE* out) {
	
	fprintf(out, "Start OpenMP With Cache Optimization\nNumber Of Vectors: %ld\n", (long) (size * size / 2));
	
	double t_start = omp_get_wtime() * 1000;

	double x_mean = 0, y_mean = 0;
	#pragma omp parallel num_threads(2)
	{
		#pragma omp for reduction (+:x_mean,y_mean)
		for (int i = 0; i < size; ++i) 
			for (int j = 0; j < size; ++j)
				if (j % 2 == 0)
					x_mean += matrix[i][j];
				else 
					y_mean += matrix[i][j];
		
	}

	x_mean /= (size * size / 2);
	y_mean /= (size * size / 2);

	double SS_xx = 0;
	double SS_xy = 0;
	#pragma omp parallel num_threads(2)
	{
		#pragma omp for reduction (+:SS_xx,SS_xy)
		for (int i = 0; i < size; ++i) 
			for (int j = 0; j <= size - 2; j += 2) {
				SS_xx += (matrix[i][j] - x_mean) * (matrix[i][j] - x_mean);
				SS_xy += (matrix[i][j] - x_mean) * (matrix[i][j + 1] - y_mean);
			}
		
	}
	
	double a = SS_xy / SS_xx;
	double b = y_mean - a * x_mean;

	double t_end = omp_get_wtime() * 1000 - t_start;

	fprintf(out, "Time: %.2lf ms\n", t_end);
	fprintf(out, "y = %.5lf * x + %.5lf\n", a, b);
}

int main() {

	// number of vectors 50'000'000
	size_t size_50 = 10000;

	// number of vectors 24'500'000
	size_t size_24_5 = 7000;

	// number of vectors 12'500'000
	size_t size_12_5 = 5000;
	
	// number of vectors 6'125'000
	size_t size_6_125 = 3500;

	// number of vectors 1'125'000
	size_t size_1_25 = 1500;

	double** matrix = (double **) calloc(size_50, sizeof(double *));

	double t_start = omp_get_wtime() * 1000;
	getMatrix(matrix, size_50);
	double t_end = omp_get_wtime() * 1000 - t_start;
	printf("Time: %.2lf ms\n", t_end);
	
	printf("Data loaded!\n");

	FILE* out;
	out = fopen("output", "a");

	// 1'125'000
	fprintf(out, "-------------------------------------------------------------------\n");
	startNormalWithoutCacheOptimization(matrix, size_1_25, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startNormalWithCacheOptimization(matrix, size_1_25, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startOpenMpWithoutCacheOptimization(matrix, size_1_25, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startOpenMpWithCacheOptimization(matrix, size_1_25, out);
	fprintf(out, "-------------------------------------------------------------------\n");

	fprintf(out, "\n\n");

	// 6'125'000
	fprintf(out, "-------------------------------------------------------------------\n");
	startNormalWithoutCacheOptimization(matrix, size_6_125, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startNormalWithCacheOptimization(matrix, size_6_125, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startOpenMpWithoutCacheOptimization(matrix, size_6_125, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startOpenMpWithCacheOptimization(matrix, size_6_125, out);
	fprintf(out, "-------------------------------------------------------------------\n");

	fprintf(out, "\n\n");

	// 12'500'000
	fprintf(out, "-------------------------------------------------------------------\n");
	startNormalWithoutCacheOptimization(matrix, size_12_5, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startNormalWithCacheOptimization(matrix, size_12_5, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startOpenMpWithoutCacheOptimization(matrix, size_12_5, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startOpenMpWithCacheOptimization(matrix, size_12_5, out);
	fprintf(out, "-------------------------------------------------------------------\n");

	fprintf(out, "\n\n");

	// 24'500'000
	fprintf(out, "-------------------------------------------------------------------\n");
	startNormalWithoutCacheOptimization(matrix, size_24_5, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startNormalWithCacheOptimization(matrix, size_24_5, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startOpenMpWithoutCacheOptimization(matrix, size_24_5, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startOpenMpWithCacheOptimization(matrix, size_24_5, out);
	fprintf(out, "-------------------------------------------------------------------\n");

	fprintf(out, "\n\n");

	// 50'000'000
	fprintf(out, "-------------------------------------------------------------------\n");
	startNormalWithoutCacheOptimization(matrix, size_50, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startNormalWithCacheOptimization(matrix, size_50, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startOpenMpWithoutCacheOptimization(matrix, size_50, out);
	fprintf(out, "-------------------------------------------------------------------\n");
	startOpenMpWithCacheOptimization(matrix, size_50, out);
	fprintf(out, "-------------------------------------------------------------------\n");

	fprintf(out, "\n###################################################################\n\n");
}