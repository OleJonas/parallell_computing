#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <getopt.h>
#include <pthread.h>
#include <cblas.h>
#include <omp.h>

#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)
#define min(x,y) (((x) < (y)) ? (x) : (y))


void options ( int argc, char **argv );

const char *usage =
"\t-k <int>\n"
"\t-m <int>\n"
"\t-n <int>\n"
"\t-t <int>\n";


//Global variables and definitions
int num_threads = 4;

double
    *A, *B,
    *C_serial, *C_openmp, *C_pthreads, *C_blas,
    alpha = 1.0, beta = 0.0;

int
    m = 1024, n = 1024, k = 1024;

void print_matrices()
{
	size_t i, j;
	printf (" Top left corner of matrix A: \n");
	for (i=0; i<min(m,6); i++) {
		for (j=0; j<min(k,6); j++) {
			printf ("%12.0f", A[j+i*k]);
		}
		printf ("\n");
	}

	printf ("\n Top left corner of matrix B: \n");
	for (i=0; i<min(k,6); i++) {
		for (j=0; j<min(n,6); j++) {
			printf ("%12.0f", B[j+i*n]);
		}
		printf ("\n");
	}

	printf ("\n Top left corner of matrix C: \n");
	for (i=0; i<min(m,6); i++) {
		for (j=0; j<min(n,6); j++) {
			printf ("%12.5G", C_serial[j+i*n]);
		}
		printf ("\n");
	}
}

//TODO: Pthreads - function
void* calc_matrix_part(void* thread_number){

    int* thread_num_p = (int*) thread_number;
    int rows_per_t = -1;
    int remainder = m % num_threads; // This will be the last threads' amount of extra rows to calculate
    if(remainder == 0){
        rows_per_t = m / num_threads;
    } else{
        rows_per_t = (m-remainder) / (num_threads);
    }

    int start = *thread_num_p * rows_per_t; // Find offset to start from for each thread
    int end = (*thread_num_p == num_threads - 1) ? start + rows_per_t + remainder : start + rows_per_t;
    for (int s = start; s < end; s++) {
        for (int t = 0; t < n; t++) {
            double sum = 0;
            for (int u = 0; u < k; u++) {
                sum = sum + A[s * k + u] * B[u * n + t];
            }
            C_pthreads[s * n + t] = sum;
        }
    }
    return 0;
}


//TODO end

int main ( int argc, char **argv )
{
    options ( argc, argv );
    A = (double *)malloc( m*k*sizeof( double ) );
    B = (double *)malloc( k*n*sizeof( double ) );
    C_serial = (double *)malloc( m*n*sizeof( double ) );
    C_openmp = (double *)malloc( m*n*sizeof( double ) );
    C_pthreads = (double *)malloc( m*n*sizeof( double ) );
    C_blas = (double *)malloc( m*n*sizeof( double ) );

    int i, j;

    /* Initialize with dummy data */
    for ( i = 0; i < (m*k); i++ )
        A[i] = (double)(i+1);
    for ( i = 0; i < (k*n); i++ )
        B[i] = (double)(-i-1);
    for ( i = 0; i < (m*n); i++ ){
        C_serial[i] = 0.0;
        C_openmp [i] = 0.0;
        C_pthreads[i] = 0.0;
        C_blas[i] = 0.0;
    }

    struct timeval start, end;

    double total_time_serial = 0;
    double total_time_openmp = 0;
    double total_time_pthreads = 0;
    double total_time_blas = 0;

    /*
     * DGEMM (Double-precision GEneral Matrix Multiply)
     * Example: A general multiplication between two matricies A and B, accumulating in C.
     */
    // A has m rows and k columns
    // B has k rows and n columns
    // C has m rows and n columns
    double sum = 0;
    int p;


    gettimeofday( &start, NULL );
    for (i = 0; i < m; i++) {
    	for (j = 0; j < n; j++) {
    	    sum = 0;
    	    for (p = 0; p < k; p++) {
                sum = sum + A[i * k + p] * B[p * n + j];
    	    }
    	    C_serial[i * n + j] = sum;
        }
    }
    gettimeofday ( &end, NULL );
    total_time_serial = (WALLTIME(end)-WALLTIME(start));
    //print_matrices();


//TODO: OpenMP

    omp_set_num_threads(num_threads);

    gettimeofday( &start, NULL );
    #pragma omp parallel for 
    for (i = 0; i < m; i++) {
    	for (j = 0; j < n; j++) {
    	    sum = 0;
    	    for (p = 0; p < k; p++) {
                sum = sum + A[i * k + p] * B[p * n + j];
    	    }
    	    C_openmp[i * n + j] = sum;
    	}
    }
    gettimeofday ( &end, NULL );
    total_time_openmp = (WALLTIME(end)-WALLTIME(start));
//TODO end


//TODO: Pthreads - spawn threads
    gettimeofday( &start, NULL );
    pthread_t* threads = malloc(sizeof(pthread_t)*num_threads);
    int* thread_nums = malloc(sizeof(int)*num_threads);

    for(i = 0; i < num_threads; ++i){
        thread_nums[i] = i;
        pthread_create(&threads[i], NULL, (void*)calc_matrix_part, &thread_nums[i]);
    }
    for(i = 0; i < num_threads; ++i){
        pthread_join(threads[i], NULL);
    }

    gettimeofday ( &end, NULL );
    total_time_pthreads = (WALLTIME(end)-WALLTIME(start));
//TODO end

    gettimeofday( &start, NULL );
    // Documentation for GEMM:
    // https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-3-routines/cblas-gemm.html
    cblas_dgemm(
    	CblasRowMajor, 	// Formating of the matrix, RowMajor or ColMajor
    	CblasNoTrans, 	// The form of Matrix A
    	CblasNoTrans, 	// The form of Matrix B
    	m,		// Number of rows in A and in C
    	n, 		// Number of columns in B and C
    	k, 		// Number of columns in A and rows of B
    	alpha, 	  	// Scalar alpha of multiplication with A
    	A, 		// The A matrix
    	k, 		// Leading dimension of A, k if CblasRowMajor and NoTrans
    	B, 		// The B matrix
    	n, 		// Leading dimension of B, n if CblasRowMajor and NoTrans
    	beta,         	// Scalar beta of multiplication with C
    	C_blas, 		// The C matrix, this is also the output matrix
    	n		// Leading dimension of C, n if CblasRowMajor
    );
    gettimeofday ( &end, NULL );
    //print_matrices();
    total_time_blas = (WALLTIME(end)-WALLTIME(start));


    bool openmp_correct = true;
    bool pthreads_correct = true;
    bool blas_correct = true;

    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            if(C_openmp[i*n+j] != C_serial[i*n+j])
                openmp_correct = false;
            if(C_pthreads[i*n+j] != C_serial[i*n+j])
                pthreads_correct = false;
            if(C_blas[i*n+j] != C_serial[i*n+j])
                blas_correct = false;
        }
    }

    printf ( "Manual\t\t\tTime:\t%es\n", total_time_serial );

    printf("OpenMP:\t  %s\t", openmp_correct ? "CORRECT" : "INCORRECT");
    printf ( "Time:\t%es\t", total_time_openmp );
    printf ( "Speedup: %.2lfx\n", total_time_serial/total_time_openmp );

    printf("Pthreads: %s\t", pthreads_correct ? "CORRECT" : "INCORRECT");
    printf ( "Time:\t%es\t", total_time_pthreads );
    printf ( "Speedup: %.2lfx\n", total_time_serial/total_time_pthreads );

    printf("BLAS:\t  %s\t", blas_correct ? "CORRECT" : "INCORRECT");
    printf ( "Time:\t%es\t", total_time_blas );
    printf ( "Speedup: %.2lfx\n", total_time_serial/total_time_blas );

    free ( A );
    free ( B );
    free ( C_serial );
    free ( C_openmp );
    free ( C_pthreads );
    free ( C_blas );

    exit ( EXIT_SUCCESS );
}

void
options ( int argc, char **argv )
{
    int o;
    while ( (o = getopt(argc,argv,"k:m:n:t:h")) != -1 )
    switch ( o )
    {
        case 'k': k = strtol ( optarg, NULL, 10 ); break;
        case 'm': m = strtol ( optarg, NULL, 10 ); break;
        case 'n': n = strtol ( optarg, NULL, 10 ); break;
        case 't': num_threads = strtol ( optarg, NULL, 10 ); break;
        case 'h':
            fprintf ( stderr, "%s", usage );
            exit ( EXIT_FAILURE );
            break;
    }
}
