#pragma once
#include "functions.h"

#ifdef __cplusplus
extern "C" {
 #endif

 #define PMT_CREATE(devID, numGPUs)          Create_PMT((devID), (numGPUs))
  
 #define PMT_CPU_START(string) Start_PMT_CPU((string))
 #define PMT_CPU_STOP(string)  Stop_PMT_CPU((string))
 #define PMT_CPU_SHOW(string)  Show_PMT_CPU((string))

 #if defined(_NVIDIA_) || defined(_AMD_)

 #define PMT_GPU_START(string, devID) Start_PMT_GPU((string), (devID))
 #define PMT_GPU_STOP(string, devID)  Stop_PMT_GPU((string), (devID))
 #define PMT_GPU_SHOW(string, devID)  Show_PMT_GPU((string), (devID))

 #else

 #define PMT_GPU_START(string, devID) 
 #define PMT_GPU_STOP(string, devID)  
 #define PMT_GPU_SHOW(string, devID)

 #endif // defined(_NVIDIA_) || defined(_AMD_)

 #define PMT_FREE()     Free_PMT()
  
 #ifdef __cplusplus
}
#endif //__cplusplus

