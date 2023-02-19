#include <stdio.h>
#include <mpi.h>
#include "malloc.h"
#include "math.h"

#define N 6 // matrix size

double norm (const double * vector, int size){
    double norm = 0;
    for (int i = 0; i < size; ++i) {
        norm += vector[i] * vector[i];
    }
    return sqrt(norm);
}

int** separator(int comm_size, int capacity, int** previous_separation_parameters ){
    int** separation_parameters = malloc(sizeof (int*) * 2);

    if (capacity == N) {
        // Vector separation
        int* len_array = malloc(sizeof (int) * comm_size);
        for (int i = 0; i < comm_size; ++i) {
            len_array[i] = previous_separation_parameters[0][i] / N;
        }

        int* indent_array = malloc(sizeof(int) * comm_size);
        for (int i = 0; i < comm_size; ++i) {
            if (i == 0) {
                indent_array[i] = 0;
            }
            else{
                indent_array[i] = indent_array[i - 1] + len_array[i - 1];
            }
        }
        separation_parameters[0] = len_array;
        separation_parameters[1] = indent_array;
    }
    else {
        // Matrix separation
        int * len_array = malloc(sizeof(int) * comm_size);
        {
            int capacity_copy = capacity;
            int i = 0;
            while (i < comm_size - 1) {
                capacity_copy -= N;
                len_array[i] = N;
                ++i;
            }
            len_array[i] = capacity_copy;
        }

        int * indent_array = malloc(sizeof(int) * comm_size);

        for (int i = 0; i < comm_size; ++i) {
            if (i == 0) {
                indent_array[i] = 0;
            } else {
                indent_array[i] = indent_array[i - 1] + len_array[i - 1];
            }
        }
        separation_parameters[0] = len_array;
        separation_parameters[1] = indent_array;
    }

    return separation_parameters;
}

void separator_free(int** separator){
    for (int i = 0; i < 2; ++i) {
        free(separator[i]);
    }
    free(separator);
}

int accuracy (const double * x_next, const double * receiver_array, const double * b, double epsilon, int** matrix_separation_parameters, int** y_separation_parameters){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


    double* y_fragment = malloc(sizeof (double) * (matrix_separation_parameters[0][rank] / N));
    for (int i = 0; i < (matrix_separation_parameters[0][rank] / N) ; ++i) {
        y_fragment[i] = 0;
    }

    //  Ax^n
    for (int i = 0; i < matrix_separation_parameters[0][rank]; ++i) {
        y_fragment[i/N] += receiver_array[i] * x_next[i % N];
    }

    //  Ax^n - b
    for (int i = 0; i < (matrix_separation_parameters[0][rank] / N) ; ++i) {
        y_fragment[i] -= b[i+rank];
    }


    double * y_n = malloc(sizeof(double ) *N);
    MPI_Allgatherv(y_fragment, y_separation_parameters[0][rank], MPI_DOUBLE, y_n, y_separation_parameters[0], y_separation_parameters[1], MPI_DOUBLE, MPI_COMM_WORLD);

    double first_norm = norm(y_n,N);
    double second_norm = norm(b,N);

    free(y_fragment);
    free(y_n);
    if (first_norm/second_norm < epsilon){
        return 1;
    }
    else {
        return 0;
    }
}


double * iterative_algorithm(const double *x_previous, const double * matrix_fragment, const double * b, int ** matrix_separation_parameters, int** y_separation_parameters, int** x_separation_parameters, double* x_next_fragment) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    double* y_fragment = malloc(sizeof (double) * (matrix_separation_parameters[0][rank] / N));
    for (int i = 0; i < (matrix_separation_parameters[0][rank] / N) ; ++i) {
        y_fragment[i] = 0;
    }

    //  Ax^n
    for (int i = 0; i < matrix_separation_parameters[0][rank]; ++i) {
        y_fragment[i/N] += matrix_fragment[i] * x_previous[i % N];
    }

    //  Ax^n - b
    for (int i = 0; i < (matrix_separation_parameters[0][rank] / N) ; ++i) {
        y_fragment[i] -= b[i+rank];
    }

    // Cuts the vector into parts depending on the number of threads
    double * y_n =malloc(sizeof(double ) * N);
    MPI_Allgatherv(y_fragment, y_separation_parameters[0][rank], MPI_DOUBLE, y_n, y_separation_parameters[0], y_separation_parameters[1], MPI_DOUBLE, MPI_COMM_WORLD);

    double * matrix_y_fragment = malloc(sizeof(double ) * (matrix_separation_parameters[0][rank] / N));
    for (int i = 0; i < matrix_separation_parameters[0][rank]; ++i) {
        matrix_y_fragment[i / N] += matrix_fragment[i] * y_n[i % N];
    }

    double matrix_y_scalar_multi_fragment = 0;
    for (int i = 0; i < matrix_separation_parameters[0][rank] / N; ++i) {
        matrix_y_scalar_multi_fragment += matrix_y_fragment[i] * matrix_y_fragment[i];
    }

    double y_matrix_y_scalar_multi_fragment = 0;
    for (int i = 0; i < matrix_separation_parameters[0][rank] / N; ++i) {
        y_matrix_y_scalar_multi_fragment += y_n[i + rank] * matrix_y_fragment[i];
    }

    double * dual_matrix_y_scalar_multi = malloc(sizeof(double ) * comm_size);
    MPI_Gather(&matrix_y_scalar_multi_fragment, 1, MPI_DOUBLE, dual_matrix_y_scalar_multi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double * y_matrix_y_scalar_multi = malloc(sizeof(double ) * comm_size);
    MPI_Gather(&y_matrix_y_scalar_multi_fragment, 1, MPI_DOUBLE, y_matrix_y_scalar_multi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t;
    if ( rank == 0 ) {
        double y_matrix_y_sum = 0;
        double matrix_y_sum = 0;
        for (int i = 0; i < comm_size; ++i) {
            y_matrix_y_sum += y_matrix_y_scalar_multi[i];
        }
        for (int i = 0; i < comm_size; ++i) {
            matrix_y_sum += dual_matrix_y_scalar_multi[i];
        }
        t = (double)(y_matrix_y_sum / matrix_y_sum);
        MPI_Bcast(&t,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    if (rank != 0){
        MPI_Bcast(&t,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }


    for (int i = 0; i < matrix_separation_parameters[0][rank] / N; ++i) {
        x_next_fragment[i] = 0;
    }

    for (int i = 0; i < matrix_separation_parameters[0][rank] / N; ++i) {
        x_next_fragment[i] += x_previous[i+rank] - (t*y_n[i+rank]);
    }

    double * x_next = malloc(sizeof(double )*N);
    MPI_Allgatherv(x_next_fragment, matrix_separation_parameters[0][rank] / N, MPI_DOUBLE, x_next, x_separation_parameters[0], x_separation_parameters[1], MPI_DOUBLE, MPI_COMM_WORLD);

    int i;
    MPI_Allgather(&i, 1, MPI_INT, &i, 1,  MPI_INT, MPI_COMM_WORLD);

    {
        free(y_fragment);
        free(y_n);
        free(matrix_y_fragment);
        free(dual_matrix_y_scalar_multi);
        free(y_matrix_y_scalar_multi);
    }
    return x_next;
}



int main() {
    MPI_Init(NULL, NULL);
    int capacity = N*N;

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int** matrix_separation_parameters = separator(comm_size, capacity, NULL);
    int** x_separation_parameters = separator(comm_size, N, matrix_separation_parameters);
    int** y_separation_parameters = separator(comm_size, N, matrix_separation_parameters);
    double* x_next_fragment = malloc(sizeof(double ) * (matrix_separation_parameters[0][rank] / N));
    double *receiver_array = malloc(sizeof(double) * matrix_separation_parameters[0][rank]);

    double * x = malloc(sizeof(double) * N);
    double * tmp = x;
    double * b = malloc(sizeof(double) * N);
    double * matrix;
    if (rank == 0){
        matrix = malloc(sizeof(double) * capacity);
        for (int i = 0; i < capacity; ++i) {
            if (i % (N + 1) == 0) {
                matrix[i] = 2.0;
            } else {
                matrix[i] = 1.0;
            }
        }
        MPI_Scatterv(matrix, matrix_separation_parameters[0], matrix_separation_parameters[1], MPI_DOUBLE, receiver_array, matrix_separation_parameters[0][rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    for (int i = 0; i < N; ++i) {
        x[i] = 0.0;
    }

    for (int i = 0; i < N; ++i) {
        b[i] = (15 + i);
    }

    if ( rank != 0 ){
        MPI_Scatterv(matrix, matrix_separation_parameters[0], matrix_separation_parameters[1], MPI_DOUBLE, receiver_array, matrix_separation_parameters[0][rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Flag for checking the accuracy of calculated values
    int flag = 0;
    int count = 0;
    while ( flag == 0 ){
        double * x_next = iterative_algorithm(x,receiver_array,b, matrix_separation_parameters, y_separation_parameters, x_separation_parameters, x_next_fragment);
        MPI_Barrier(MPI_COMM_WORLD);
        flag = accuracy(x_next,receiver_array,b,0.00001,matrix_separation_parameters, y_separation_parameters);
        x = x_next;
        count++;
        if (flag == 1 && rank == 0){
            for (int i = 0; i < N; ++i) {
                printf("%f\n",x_next[i]);
            }
        }
    }

    if (rank == 0){
        printf("\nnum of iteration %d\n", count);
        free(matrix);
    }
    free(x);
    free(tmp);
    free(b);
    separator_free(matrix_separation_parameters);
    separator_free(x_separation_parameters);
    separator_free(y_separation_parameters);
    free(x_next_fragment);
    free(receiver_array);
    MPI_Finalize();
    return 0;
}
