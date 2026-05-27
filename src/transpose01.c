#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double real;
    double imag;
} Complex;

// Function to initialize the local part of the array
void initialize_local_array(Complex ***array, int local_nk, int nj, int ni, int k_start) {
    for (int k = 0; k < local_nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                array[k][j][i].real = 1000 * (k + k_start) + 100 * j + 10* i;
                array[k][j][i].imag = 0;
            }
        }
    }
}

// Function to print the local array for verification
void print_local_array(Complex ***array, int local_nk, int nj, int ni, int rank) {
    printf("Rank %d: ", rank);
    for (int k = 0; k < local_nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                printf("(%2.1f, %2.1f) ", array[k][j][i].real, array[k][j][i].imag);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

// Function to save the local array to a file for verification
void save_local_file(Complex ***array, int local_nk, int nj, int ni, int rank) {
    char filename[256];
    sprintf(filename, "output_rank_%d.txt", rank); // Unique filename for each rank

    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing\n");
        return;
    }

    fprintf(file, "Rank %d:\n", rank);
    for (int k = 0; k < local_nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                fprintf(file, "(%2.1f, %2.1f) ", array[k][j][i].real, array[k][j][i].imag);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n\n");
    }

    fclose(file);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int nk = 8, nj = 8, ni = 8;
    int local_nk = nk / size; // Assume nk is divisible by size
    int k_start = rank * local_nk; // Start index for k on this process

    Complex ***array;

    // Allocate memory for the local slice
    array = malloc(local_nk * sizeof(Complex**));
    for (int k = 0; k < local_nk; ++k) {
        array[k] = malloc(nj * sizeof(Complex*));
        for (int j = 0; j < nj; ++j) {
            array[k][j] = malloc(ni * sizeof(Complex));
        }
    }

    // Initialize local array slice
    initialize_local_array(array, local_nk, nj, ni, k_start);

    // Print local array slice
    save_local_file(array, local_nk, nj, ni, rank);

    // Free memory
    for (int k = 0; k < local_nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            free(array[k][j]);
        }
        free(array[k]);
    }
    free(array);

    MPI_Finalize();
    return 0;
}
