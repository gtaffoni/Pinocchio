#pragma once

#include "energy_pmt_methods.hpp"


#define PMT_CREATE(devID, numGPUs)          mpi::Create_PMT((devID), (numGPUs))

#define PMT_CPU_START(string) mpi::Start_PMT_CPU((string))
#define PMT_CPU_STOP(string)  mpi::Stop_PMT_CPU((string))
#define PMT_CPU_SHOW(string)  mpi::Show_PMT_CPU((string))

#if defined(_NVIDIA_) || defined(_AMD_)

#define PMT_GPU_START(string, devID) mpi::Start_PMT_GPU((string), (devID))
#define PMT_GPU_STOP(string, devID)  mpi::Stop_PMT_GPU((string), (devID))
#define PMT_GPU_SHOW(string, devID)  mpi::Show_PMT_GPU((string), (devID))

#else

#define PMT_GPU_START(string, devID) 
#define PMT_GPU_STOP(string, devID)  
#define PMT_GPU_SHOW(string, devID)

#endif // defined(_NVIDIA_) || defined(_AMD_)

