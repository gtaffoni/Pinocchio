#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double real;
    double imag;
} Complex;

void setup_mpi_datatype(MPI_Datatype *complex_type) {
    MPI_Type_contiguous(2, MPI_DOUBLE, complex_type);
    MPI_Type_commit(complex_type);
}

void initialize_local_array(Complex *array, int local_nk, int nj, int ni, int k_start) {
    for (int k = 0; k < local_nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                int index = k * nj * ni + j * ni + i;
                array[index].real = 1000 * (k + k_start) + 100 * j + 10 * i;
                array[index].imag = 0;
            }
        }
    }
}

// Simplified global transposition assuming even distribution of the nk dimension
void global_transpose(Complex *local_data, Complex *transposed_data, int nk, int nj, int ni, MPI_Comm comm, MPI_Datatype complex_type) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    int total_elements = nk * nj * ni;
    int local_elements = total_elements / size;
    
    Complex *all_data = NULL;
    if (rank == 0) {
        all_data = malloc(total_elements * sizeof(Complex));
    }

    MPI_Gather(local_data, local_elements, complex_type, all_data, local_elements, complex_type, 0, comm);

    if (rank == 0) {
        // Assuming simple transpose that might need further adjustments
        for (int k = 0; k < nk; k++) {
            for (int j = 0; j < nj; j++) {
                for (int i = 0; i < ni; i++) {
                    int src_idx = k * nj * ni + j * ni + i;
                    int dst_idx = i * nk * nj + j * nk + k; // Adjust as per actual transpose needs
                    transposed_data[dst_idx] = all_data[src_idx];
                }
            }
        }
        free(all_data);
    }

    MPI_Scatter(transposed_data, local_elements, complex_type, local_data, local_elements, complex_type, 0, comm);
}

void save_local_file(Complex *array, Complex *tarray, int nk, int nj, int ni, int rank, const char* description) {
    char filename[256];
    sprintf(filename, "output_rank_%d.txt", rank);
    FILE *file = fopen(filename, "a");  // Open in append mode to write both arrays in the same file

    fprintf(file, "%s Array for Rank %d:\n", description, rank);
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                int index = k * nj * ni + j * ni + i;
                fprintf(file, "(%.1f, %.1f) ", array[index].real, array[index].imag);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
    fclose(file);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int nk = 8, nj = 8, ni = 8;
    int local_nk = nk / size; // Assume nk is divisible by size
    int k_start = rank * local_nk;

    Complex *local_data = malloc(local_nk * nj * ni * sizeof(Complex));
    Complex *transposed_data = malloc(local_nk * nj * ni * sizeof(Complex));

    MPI_Datatype complex_type;
    setup_mpi_datatype(&complex_type);

    initialize_local_array(local_data, local_nk, nj, ni, k_start);

    global_transpose(local_data, transposed_data, nk, nj, ni, MPI_COMM_WORLD, complex_type);
    save_local_file(local_data, transposed_data, ni, nj, local_nk, rank, "Transposed");

    free(local_data);
    free(transposed_data);
    MPI_Type_free(&complex_type);
    MPI_Finalize();
    return 0;
}