#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define _NVIDIA_
#include "energy_pmt.h"

// Funzione per il prodotto di matrici
void matrixProduct(double* A, double* B, double* C, int M, int N, int P) {
  // Direttiva OpenMP per parallelizzare il loop su GPU
  #pragma omp target teams distribute parallel for collapse(2)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      double sum = 0.0;
      // Direttiva OpenMP per parallelizzare il loop interno con riduzione
      #pragma omp simd reduction(+:sum)
      for (int k = 0; k < N; k++) {
        sum += A[i*N + k] * B[k*P + j];
      }
      C[i*P + j] = sum;
    }
  }
}

int main() {
  // Inizializzazione MPI
  MPI_Init(NULL, NULL);
  int rank, size;
  // Ottieni il rank del processo corrente e il numero totale di processi
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Seleziona il dispositivo GPU da utilizzare in base al rank del processo
  int devID = rank % omp_get_num_devices();
  omp_set_default_device(devID);

  // Dimensioni delle matrici
  const int M = 8192;
  const int N = 8192;
  const int P = 8192;

  // Calcola la porzione di matrice da elaborare per ogni processo
  int rows_per_process = M / size;
  int remainder = M % size;
  if (rank < remainder) {
    rows_per_process++;
  }

  // Allocazione della memoria per le matrici
  double* A = malloc(rows_per_process*N*sizeof(double));
  double* B = malloc(N*P*sizeof(double));
  double* C_local = malloc(rows_per_process*P*sizeof(double));
  double* C = NULL;
  if (rank == 0) {
    C = malloc(M*P*sizeof(double));
  }

  // Inizializzazione delle matrici
  if (rank == 0) {
    // Processo 0 inizializza la matrice B e la distribuisce agli altri processi
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < P; j++) {
        B[i*P + j] = i + j;
      }
    }
    for (int i = 1; i < size; i++) {
      MPI_Send(B, N*P, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
  } else {
    // Altri processi ricevono la matrice B dal processo 0
    MPI_Recv(B, N*P, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Inizializzazione della porzione di matrice A per ogni processo
  int offset = 0;
  if (rank >= remainder) {
    offset = remainder * (N * (rows_per_process + 1));
  } else {
    offset = rank * N * rows_per_process;
  }
  if (rank == 0) {
    for (int i = 0; i < rows_per_process; i++) {
      for (int j = 0; j < N; j++) {
        A[i*N + j] = (i + offset / N) + j;
      }
    }
    for (int i = 1; i < size; i++) {
      int rows = M / size;
      int remainder_local = M % size;
      if (i < remainder_local) {
        rows++;
      }
      int offset_local = 0;
      if (i >= remainder_local) {
        offset_local = remainder_local * (N * (rows + 1));
      } else {
        offset_local = i * N * rows;
      }

      double* A_local = malloc(rows*N*sizeof(double));
      for (int k = 0; k < rows; k++) {
        for (int j = 0; j < N; j++) {
          A_local[k*N + j] = (k + offset_local / N) + j;
        }
      }
      MPI_Send(A_local, rows*N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
      free(A_local);
    }
  } else {
    MPI_Recv(A, rows_per_process*N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Compilazione del prodotto di matrici
  PMT_CREATE(&devID, 1);
  PMT_CPU_START("Prodotto");
  PMT_GPU_START("Prodotto", devID);

  #pragma omp target data map(to:A[0:rows_per_process*N], B[0:N*P]) map(from:C_local[0:rows_per_process*P])
  matrixProduct(A, B, C_local, rows_per_process, N, P);

  PMT_CPU_STOP("Prodotto");
  PMT_GPU_STOP("Prodotto", devID);

  PMT_CPU_SHOW("Prodotto");
  PMT_GPU_SHOW("Prodotto", devID);

  PMT_FREE();
  
  // Raccolta dei risultati
  if (rank == 0) {
    for (int i = 0; i < rows_per_process; i++) {
      for (int j = 0; j < P; j++) {
        C[i*P + j] = C_local[i*P + j];
      }
    }
    for (int i = 1; i < size; i++) {
      int rows = M / size;
      int remainder_local = M % size;
      if (i < remainder_local) {
        rows++;
      }
      MPI_Recv(C + (i * (M / size) + (i < remainder_local ? i : remainder_local)) * P, rows*P, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  } else {
    MPI_Send(C_local, rows_per_process*P, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  // Stampa del risultato
  if (rank == 0) {
    printf("Risultato:\n");
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        printf("%f ", C[i*P + j]);
      }
      printf("\n");
    }
  }

  // Liberazione della memoria
  free(A);
  free(B);
  free(C_local);
  if (rank == 0) {
    free(C);
  }

  // Terminazione MPI
  MPI_Finalize();

  return 0;
}
