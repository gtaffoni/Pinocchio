#include <stdio.h>
#include <unistd.h>

//#define _NVIDIA_

#include "energy_pmt.h"
#include <mpi.h>
#include <omp.h>



int main()
{
  MPI_Init(NULL,NULL);

  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int devID, numGPUs;
  numGPUs = omp_get_num_devices();
  devID = rank % numGPUs;

  omp_set_default_device(devID);

  double tmp = MPI_Wtime();

 #if defined(PMT)
  PMT_CREATE(&devID,1);

  PMT_CPU_START("Kernel");
  PMT_GPU_START("Kernel", devID);
 #endif //PMT
  
  int gpu_working = 0;
 #pragma omp target map(from: gpu_working)
  {
    gpu_working = (omp_is_initial_device() ? 0 : 1);
  }

  if (gpu_working == 1)
    printf("Rank %d --> GPU is working!\n", rank);
  else
    printf("Rank %d --> Host is working!\n", rank);
  
  unsigned int my_sum = 0;
 #pragma omp target teams distribute parallel for reduction(+: my_sum)
  for (unsigned int i=0; i<10000; i++)
  {
    my_sum += (rank + 1) * i;
  }

 #if defined(PMT)
  PMT_CPU_STOP("Kernel");
  PMT_GPU_STOP("Kernel", devID);
 #endif //PMT
  
  tmp = MPI_Wtime() - tmp;

 #if defined(PMT)
  PMT_CPU_SHOW("Kernel");
  PMT_GPU_SHOW("Kernel", devID);

  PMT_FREE();
 #endif //PMT

  printf("Rank %d --> Summation = %u, Timings = %g\n", rank, my_sum, tmp);
  
  MPI_Finalize();
  return 0;
}
