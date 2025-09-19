#include <stdio.h>
#include <unistd.h>
#include <iostream>

//#include <pmt.h>
#include <pmt/NVML.h>
#include <pmt/Rapl.h>
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

   
 #if defined(PROFILE)
  std::unique_ptr<pmt::PMT> sensor_rapl = pmt::rapl::Rapl::Create();
  std::unique_ptr<pmt::PMT> sensor = pmt::nvml::NVML::Create(devID);
 #endif //PROFILE
 
  double tmp = MPI_Wtime();

  
 #if defined(PROFILE)
  auto start = sensor->Read();
  auto start_rapl = sensor_rapl->Read();
 #endif //PROFILE
  
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

 #if defined(PROFILE)
  auto stop = sensor->Read();
  auto stop_rapl = sensor_rapl->Read();
 #endif //PROFILE
  
  tmp = MPI_Wtime() - tmp;

 #if defined(PROFILE)
  //CPU counters
  std::cout << "GPU Runtime: " << sensor->seconds(start, stop) << " s" << std::endl;
  std::cout << "GPU Joules: " << sensor->joules(start, stop) << " J" << std::endl;
  std::cout << "GPU Watt: " << sensor->watts(start, stop) << " W" << std::endl;

  //GPU counters
  std::cout << "CPU Runtime: " << sensor_rapl->seconds(start_rapl, stop_rapl) << " s" << std::endl;
  std::cout << "CPU Joules: " << sensor_rapl->joules(start_rapl, stop_rapl) << " J" << std::endl;
  std::cout << "CPU Watt: " << sensor_rapl->watts(start_rapl, stop_rapl) << " W" << std::endl;
 #endif //PROFILE

  printf("Rank %d --> Summation = %u, Timings = %g\n", rank, my_sum, tmp);
  
  MPI_Finalize();
  return 0;
}
