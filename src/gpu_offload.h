/* ============================================================
   GPU Offloading Infrastructure for PINOCCHIO
   Uses OpenMP target directives for GPU acceleration.
   Model: 1 MPI task per GPU.

   Compile with -DUSE_GPU to enable GPU offloading.
   Without USE_GPU, all functions are no-ops (CPU fallback).
   ============================================================ */

#ifndef GPU_OFFLOAD_H
#define GPU_OFFLOAD_H

#include "def_splines.h"

/* Forward declarations of types defined in build_groups.c */

#define GFLUT_NPTS 10000

typedef struct {
  int npts;
  int initialized;
  double z_min, z_max;
  double dz_inv;
  double grow1  [NkBINS][GFLUT_NPTS];
  double grow2  [NkBINS][GFLUT_NPTS];
  double grow31 [NkBINS][GFLUT_NPTS];
  double grow32 [NkBINS][GFLUT_NPTS];
  double fomega1[NkBINS][GFLUT_NPTS];
  double fomega2[NkBINS][GFLUT_NPTS];
  double fomega31[NkBINS][GFLUT_NPTS];
  double fomega32[NkBINS][GFLUT_NPTS];
  double hubble [GFLUT_NPTS];
#ifdef SCALE_DEPENDENT
  double kmin, kmax;
#endif
} gflut_data;

typedef struct {
  PRODFLOAT euler[3];
  double sigmaD;
} particle_precomp;

/* Global instances (defined in build_groups.c) */
extern gflut_data gflut;
extern particle_precomp *precomp;
extern double k_dens_global;


#ifdef USE_GPU

#include <omp.h>

/* GPU lifecycle */
void gpu_init(void);
void gpu_map_gflut(void);
void gpu_unmap_gflut(void);

/* GPU kernel for particle pre-computation */
void precompute_particles_gpu(int N,
                               product_data *frag_arr,
                               int *frag_pos_arr,
                               particle_precomp *precomp_arr,
                               double k_point, double k_dens,
                               double sqrt_var,
                               int myseg,
                               double z_seg, double z_seg_prev,
                               int Lgwbl0, int Lgwbl1, int Lgwbl2,
                               int pbc0, int pbc1, int pbc2);

#else  /* !USE_GPU */

static inline void gpu_init(void) {}
static inline void gpu_map_gflut(void) {}
static inline void gpu_unmap_gflut(void) {}

#endif /* USE_GPU */

#endif /* GPU_OFFLOAD_H */
