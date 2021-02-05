main: main.c
	# gcc -fopenmp -o main main.c
	gcc -fopenmp -O3 -msse2 -mfpmath=sse -ftree-vectorizer-verbose=5 -o main main.c