/* ============================================================
   GPU Offloading Implementation for PINOCCHIO
   OpenMP target directives — 1 MPI task per GPU.

   Compile with -DUSE_GPU to enable.
   Requires: nvc (NVIDIA HPC SDK) with -mp=gpu, or
             clang with -fopenmp -fopenmp-targets=nvptx64
   ============================================================ */

#include "pinocchio.h"
#include "gpu_offload.h"

#ifdef USE_GPU

/* ---- GPU device management ---- */

static int gpu_device_id = -1;

void gpu_init(void)
{
  int ndevices = omp_get_num_devices();
  if (ndevices > 0)
    {
      gpu_device_id = ThisTask % ndevices;
      omp_set_default_device(gpu_device_id);
      printf("[GPU] Task %d: %d device(s) detected, using device %d\n",
             ThisTask, ndevices, gpu_device_id);
    }
  else
    {
      gpu_device_id = -1;
      if (!ThisTask)
        printf("[GPU] No GPU devices found, falling back to CPU\n");
    }
}


/* ---- Persistent GFLUT mapping ---- */

void gpu_map_gflut(void)
{
  if (gpu_device_id < 0) return;

  /* Map the entire gflut struct to GPU as read-only.
     gflut contains only scalars and fixed-size arrays (no pointers),
     so the compiler can map it directly.
     Size: ~720KB (NkBINS=1) or ~6.5MB (NkBINS=10). */
#pragma omp target enter data map(to: gflut)

  if (!ThisTask)
    printf("[GPU] GFLUT tables mapped to device %d (%.1f MB)\n",
           gpu_device_id,
           sizeof(gflut_data) / 1.0e6);
}


void gpu_unmap_gflut(void)
{
  if (gpu_device_id < 0) return;

#pragma omp target exit data map(delete: gflut)
}


/* ============================================================
   Device-callable interpolation functions.
   These replicate the logic from build_groups.c but are
   compiled for both host and device via declare target.
   ============================================================ */

#pragma omp declare target

static inline double catmull_rom_dev(const double *y, int npts, int i, double t)
{
  if (i <= 0)
    return y[0] * (1.0 - t) + y[1] * t;
  if (i >= npts - 2)
    return y[npts-2] * (1.0 - t) + y[npts-1] * t;

  double y0 = y[i-1], y1 = y[i], y2 = y[i+1], y3 = y[i+2];
  double a = -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3;
  double b =      y0 - 2.5*y1 + 2.0*y2 - 0.5*y3;
  double c = -0.5*y0           + 0.5*y2;
  double d =                y1;
  return ((a*t + b)*t + c)*t + d;
}


static inline double gflut_interp_1d_dev(const double *table,
                                          int npts, double z_min, double dz_inv,
                                          double z)
{
  double fz = (z - z_min) * dz_inv;
  int iz = (int)fz;
  if (iz < 0) iz = 0;
  if (iz >= npts - 1) iz = npts - 2;
  double tz = fz - iz;
  return catmull_rom_dev(table, npts, iz, tz);
}


static inline double gflut_interp_dev(const double table[][GFLUT_NPTS],
                                       int npts, double z_min, double z_max,
                                       double dz_inv,
                                       double kmin_g, double kmax_g,
                                       double z, double k)
{
  if (z < z_min) z = z_min;
  if (z > z_max) z = z_max;

  double fz = (z - z_min) * dz_inv;
  int iz = (int)fz;
  if (iz < 0) iz = 0;
  if (iz >= npts - 1) iz = npts - 2;
  double tz = fz - iz;

#ifdef SCALE_DEPENDENT
  if (k <= kmin_g)
    return catmull_rom_dev(table[0], npts, iz, tz);
  else if (k >= kmax_g)
    return catmull_rom_dev(table[NkBINS-1], npts, iz, tz);
  else
    {
      double dk = (log10(k) - LOGKMIN) / DELTALOGK;
      int ik = (int)dk;
      if (ik < 0) ik = 0;
      if (ik >= NkBINS - 1) ik = NkBINS - 2;
      double wk = dk - ik;
      double v0 = catmull_rom_dev(table[ik],   npts, iz, tz);
      double v1 = catmull_rom_dev(table[ik+1], npts, iz, tz);
      return (1.0 - wk) * v0 + wk * v1;
    }
#else
  (void)k; (void)kmin_g; (void)kmax_g;
  return catmull_rom_dev(table[0], npts, iz, tz);
#endif
}


/* Device-callable fast_GrowingMode wrapper.
   kmin_g/kmax_g are always passed for uniform function signature;
   ignored when SCALE_DEPENDENT is not defined. */

static inline double dev_GrowingMode(double z, double k,
                                      int npts, double z_min, double z_max,
                                      double dz_inv,
                                      double kmin_g, double kmax_g,
                                      const double table[][GFLUT_NPTS])
{
  return gflut_interp_dev(table, npts, z_min, z_max, dz_inv,
                          kmin_g, kmax_g, z, k);
}

#pragma omp end declare target


/* ============================================================
   GPU Kernel: precompute_particles_gpu
   Offloads the per-particle pre-computation to the GPU.
   Each GPU thread computes euler[3] and sigmaD for one particle.
   ============================================================ */

void precompute_particles_gpu(int N,
                               product_data *frag_arr,
                               int *frag_pos_arr,
                               particle_precomp *precomp_arr,
                               double k_point, double k_dens,
                               double sqrt_var,
                               int myseg,
                               double z_seg, double z_seg_prev,
                               int Lgwbl0, int Lgwbl1, int Lgwbl2,
                               int pbc0, int pbc1, int pbc2)
{
  if (gpu_device_id < 0)
    return;  /* caller handles CPU fallback */

  /* Extract GFLUT scalar parameters for firstprivate use in kernel.
     The GFLUT table arrays are already on the device via gpu_map_gflut(). */
  const int    g_npts  = gflut.npts;
  const double g_zmin  = gflut.z_min;
  const double g_zmax  = gflut.z_max;
  const double g_dzinv = gflut.dz_inv;
#ifdef SCALE_DEPENDENT
  const double g_kmin  = gflut.kmin;
  const double g_kmax  = gflut.kmax;
#else
  const double g_kmin  = 0.0;  /* unused, but needed for uniform pragma */
  const double g_kmax  = 0.0;
#endif

  /* Map particle data to device and launch kernel */
#pragma omp target teams distribute parallel for \
    map(to: frag_arr[0:N], frag_pos_arr[0:N]) \
    map(from: precomp_arr[0:N]) \
    firstprivate(N, k_point, k_dens, sqrt_var, myseg, z_seg, z_seg_prev, \
                 Lgwbl0, Lgwbl1, Lgwbl2, pbc0, pbc1, pbc2, \
                 g_npts, g_zmin, g_zmax, g_dzinv, g_kmin, g_kmax)
  for (int p = 0; p < N; p++)
    {
      /* INDEX_TO_COORD: convert linear index to 3D grid coordinates */
      int kk = frag_pos_arr[p] % Lgwbl2;
      int tmp_idx = frag_pos_arr[p] / Lgwbl2;
      int jj = tmp_idx % Lgwbl1;
      int ii = tmp_idx / Lgwbl1;

      double z = (double)frag_arr[p].Fmax - 1.0;

      /* ---- fast_set_weight logic: compute LPT weights ---- */
      /* Macro to reduce repetition in dev_GrowingMode calls */
      #define DGROW(ZZ, TABLE) \
        dev_GrowingMode((ZZ), k_point, g_npts, g_zmin, g_zmax, g_dzinv, \
                        g_kmin, g_kmax, (TABLE))

      double w, w2, w31, w32;
      w2 = w31 = w32 = 0.0;

      if (!myseg)
        {
          double g_seg = DGROW(z_seg, gflut.grow1);
          w = DGROW(z, gflut.grow1) / g_seg;
#ifdef TWO_LPT
          double g2_seg = DGROW(z_seg, gflut.grow2);
          w2 = DGROW(z, gflut.grow2) / g2_seg;
#ifdef THREE_LPT
          double g31_seg = DGROW(z_seg, gflut.grow31);
          w31 = DGROW(z, gflut.grow31) / g31_seg;
          double g32_seg = DGROW(z_seg, gflut.grow32);
          w32 = DGROW(z, gflut.grow32) / g32_seg;
#endif
#endif
        }
      else
        {
          double g_cur = DGROW(z_seg, gflut.grow1);
          double g_pre = DGROW(z_seg_prev, gflut.grow1);
          double g_z   = DGROW(z, gflut.grow1);
          w = (g_z - g_pre) / (g_cur - g_pre);
#ifdef TWO_LPT
          double g2_cur = DGROW(z_seg, gflut.grow2);
          double g2_pre = DGROW(z_seg_prev, gflut.grow2);
          double g2_z   = DGROW(z, gflut.grow2);
          w2 = (g2_z - g2_pre) / (g2_cur - g2_pre);
#ifdef THREE_LPT
          double g31_cur = DGROW(z_seg, gflut.grow31);
          double g31_pre = DGROW(z_seg_prev, gflut.grow31);
          double g31_z   = DGROW(z, gflut.grow31);
          w31 = (g31_z - g31_pre) / (g31_cur - g31_pre);
          double g32_cur = DGROW(z_seg, gflut.grow32);
          double g32_pre = DGROW(z_seg_prev, gflut.grow32);
          double g32_z   = DGROW(z, gflut.grow32);
          w32 = (g32_z - g32_pre) / (g32_cur - g32_pre);
#endif
#endif
        }

      #undef DGROW

      /* ---- q2x logic: Lagrangian → Eulerian position ---- */
      /* q = grid_coord + SHIFT,  pos = q + w * v [+ w2 * v2 + ...] */

      double q0 = (double)ii + SHIFT;
      double q1 = (double)jj + SHIFT;
      double q2 = (double)kk + SHIFT;

      PRODFLOAT pos0, pos1, pos2;

      if (!myseg)
        {
          pos0 = q0 + w * frag_arr[p].Vel[0];
          pos1 = q1 + w * frag_arr[p].Vel[1];
          pos2 = q2 + w * frag_arr[p].Vel[2];
#ifdef TWO_LPT
          pos0 += w2 * frag_arr[p].Vel_2LPT[0];
          pos1 += w2 * frag_arr[p].Vel_2LPT[1];
          pos2 += w2 * frag_arr[p].Vel_2LPT[2];
#ifdef THREE_LPT
          pos0 += w31 * frag_arr[p].Vel_3LPT_1[0] + w32 * frag_arr[p].Vel_3LPT_2[0];
          pos1 += w31 * frag_arr[p].Vel_3LPT_1[1] + w32 * frag_arr[p].Vel_3LPT_2[1];
          pos2 += w31 * frag_arr[p].Vel_3LPT_1[2] + w32 * frag_arr[p].Vel_3LPT_2[2];
#endif
#endif
        }
#ifdef RECOMPUTE_DISPLACEMENTS
      else
        {
          pos0 = q0 + (1.0 - w) * frag_arr[p].Vel_prev[0] + w * frag_arr[p].Vel[0];
          pos1 = q1 + (1.0 - w) * frag_arr[p].Vel_prev[1] + w * frag_arr[p].Vel[1];
          pos2 = q2 + (1.0 - w) * frag_arr[p].Vel_prev[2] + w * frag_arr[p].Vel[2];
#ifdef TWO_LPT
          pos0 += (1.0 - w2) * frag_arr[p].Vel_2LPT_prev[0] + w2 * frag_arr[p].Vel_2LPT[0];
          pos1 += (1.0 - w2) * frag_arr[p].Vel_2LPT_prev[1] + w2 * frag_arr[p].Vel_2LPT[1];
          pos2 += (1.0 - w2) * frag_arr[p].Vel_2LPT_prev[2] + w2 * frag_arr[p].Vel_2LPT[2];
#ifdef THREE_LPT
          pos0 += (1.0 - w31) * frag_arr[p].Vel_3LPT_1_prev[0] + w31 * frag_arr[p].Vel_3LPT_1[0]
                + (1.0 - w32) * frag_arr[p].Vel_3LPT_2_prev[0] + w32 * frag_arr[p].Vel_3LPT_2[0];
          pos1 += (1.0 - w31) * frag_arr[p].Vel_3LPT_1_prev[1] + w31 * frag_arr[p].Vel_3LPT_1[1]
                + (1.0 - w32) * frag_arr[p].Vel_3LPT_2_prev[1] + w32 * frag_arr[p].Vel_3LPT_2[1];
          pos2 += (1.0 - w31) * frag_arr[p].Vel_3LPT_1_prev[2] + w31 * frag_arr[p].Vel_3LPT_1[2]
                + (1.0 - w32) * frag_arr[p].Vel_3LPT_2_prev[2] + w32 * frag_arr[p].Vel_3LPT_2[2];
#endif
#endif
        }
#endif

      /* Periodic boundary conditions */
      double Box0 = (double)Lgwbl0;
      double Box1 = (double)Lgwbl1;
      double Box2 = (double)Lgwbl2;

      if (pbc0) { if (pos0 >= Box0) pos0 -= Box0; if (pos0 < 0.0) pos0 += Box0; }
      if (pbc1) { if (pos1 >= Box1) pos1 -= Box1; if (pos1 < 0.0) pos1 += Box1; }
      if (pbc2) { if (pos2 >= Box2) pos2 -= Box2; if (pos2 < 0.0) pos2 += Box2; }

      precomp_arr[p].euler[0] = pos0;
      precomp_arr[p].euler[1] = pos1;
      precomp_arr[p].euler[2] = pos2;

      /* sigmaD = sqrt(TrueVariance) * GrowingMode(z, k_dens) */
      precomp_arr[p].sigmaD = sqrt_var * dev_GrowingMode(z, k_dens,
                                                          g_npts, g_zmin, g_zmax, g_dzinv,
                                                          g_kmin, g_kmax,
                                                          gflut.grow1);
    }

  if (!ThisTask)
    printf("[GPU] precompute_particles: %d particles processed on device %d\n",
           N, gpu_device_id);
}

#endif /* USE_GPU */
