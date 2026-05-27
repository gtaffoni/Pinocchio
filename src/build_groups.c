/*****************************************************************
 *                        PINOCCHIO  V5.1                        *
 *  (PINpointing Orbit-Crossing Collapsed HIerarchical Objects)  *
 *****************************************************************
 
 This code was written by
 Pierluigi Monaco, Tom Theuns, Giuliano Taffoni, Marius Lepinzan, 
 Chiara Moretti, Luca Tornatore, David Goz, Tiago Castro
 Copyright (C) 2025
 
 github: https://github.com/pigimonaco/Pinocchio
 web page: http://adlibitum.oats.inaf.it/monaco/pinocchio.html
 
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

/* 
   This file contains code to construct groups (dark matter halos)
   from the list of collapsed particles.
   The main function is build_groups, that performs the group
   construction down to redshift zstop.
   quick_build_groups implements a quick version of build_groups, used
   to determine which part of the boundary region should be
   communicated to perform group construction.
*/

#include "pinocchio.h"
#include "def_splines.h"
#include "gpu_offload.h"
#include <gsl/gsl_sf.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NCOUNTERS 15

unsigned long long int particle_name;
int good_particle;
pos_data obj1,obj2;

void set_weight(pos_data *);

/* ============================================================
   GFLUT: Growth Factor Lookup Tables

   Thread-safe replacement for GSL spline calls in the hot path.
   The tables are pre-computed at init time using the GSL functions,
   then interpolated during fragmentation using Catmull-Rom splines.
   This avoids thread-safety issues with GSL's mutable ACCEL state.
   ============================================================ */

/* Global instances (declared extern in gpu_offload.h) */
gflut_data gflut = {0};
particle_precomp *precomp = NULL;
double k_dens_global;

/* ---- Catmull-Rom interpolation ---- */

static inline double catmull_rom(const double *y, int npts, int i, double t)
{
  /* Boundary: linear interpolation */
  if (i <= 0)
    return y[0] * (1.0 - t) + y[1] * t;
  if (i >= npts - 2)
    return y[npts-2] * (1.0 - t) + y[npts-1] * t;

  /* Interior: 4-point Catmull-Rom stencil */
  double y0 = y[i-1], y1 = y[i], y2 = y[i+1], y3 = y[i+2];
  double a = -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3;
  double b =      y0 - 2.5*y1 + 2.0*y2 - 0.5*y3;
  double c = -0.5*y0           + 0.5*y2;
  double d =                y1;
  return ((a*t + b)*t + c)*t + d;
}


/* ---- 1D interpolation (for hubble) ---- */

static inline double gflut_interp_1d(const double *table, double z)
{
  if (z < gflut.z_min) z = gflut.z_min;
  if (z > gflut.z_max) z = gflut.z_max;

  double fz = (z - gflut.z_min) * gflut.dz_inv;
  int iz = (int)fz;
  if (iz < 0) iz = 0;
  if (iz >= gflut.npts - 1) iz = gflut.npts - 2;
  double tz = fz - iz;
  return catmull_rom(table, gflut.npts, iz, tz);
}


/* ---- 2D interpolation (z, k) for growth factors ---- */

static inline double gflut_interp(const double table[][GFLUT_NPTS], double z, double k)
{
  if (z < gflut.z_min) z = gflut.z_min;
  if (z > gflut.z_max) z = gflut.z_max;

  double fz = (z - gflut.z_min) * gflut.dz_inv;
  int iz = (int)fz;
  if (iz < 0) iz = 0;
  if (iz >= gflut.npts - 1) iz = gflut.npts - 2;
  double tz = fz - iz;

#ifdef SCALE_DEPENDENT
  if (k <= gflut.kmin)
    return catmull_rom(table[0], gflut.npts, iz, tz);
  else if (k >= gflut.kmax)
    return catmull_rom(table[NkBINS-1], gflut.npts, iz, tz);
  else
    {
      double dk = (log10(k) - LOGKMIN) / DELTALOGK;
      int ik = (int)dk;
      if (ik < 0) ik = 0;
      if (ik >= NkBINS - 1) ik = NkBINS - 2;
      double wk = dk - ik;
      double v0 = catmull_rom(table[ik],   gflut.npts, iz, tz);
      double v1 = catmull_rom(table[ik+1], gflut.npts, iz, tz);
      return (1.0 - wk) * v0 + wk * v1;
    }
#else
  (void)k;
  return catmull_rom(table[0], gflut.npts, iz, tz);
#endif
}


/* ---- Drop-in replacements for GSL-backed growth functions ---- */

static inline double fast_GrowingMode(double z, double k)
{
  return gflut_interp(gflut.grow1, z, k);
}

static inline double fast_GrowingMode_2LPT(double z, double k)
{
  return gflut_interp(gflut.grow2, z, k);
}

static inline double fast_GrowingMode_3LPT_1(double z, double k)
{
  return gflut_interp(gflut.grow31, z, k);
}

static inline double fast_GrowingMode_3LPT_2(double z, double k)
{
  return gflut_interp(gflut.grow32, z, k);
}

static inline double fast_fomega(double z, double k)
{
  return gflut_interp(gflut.fomega1, z, k);
}

static inline double fast_fomega_2LPT(double z, double k)
{
  return gflut_interp(gflut.fomega2, z, k);
}

static inline double fast_fomega_3LPT_1(double z, double k)
{
  return gflut_interp(gflut.fomega31, z, k);
}

static inline double fast_fomega_3LPT_2(double z, double k)
{
  return gflut_interp(gflut.fomega32, z, k);
}

static inline double fast_Hubble(double z)
{
  return gflut_interp_1d(gflut.hubble, z);
}


/* ---- GFLUT initialization: sample GSL functions at uniform z points ---- */

void gflut_init(double z_min, double z_max)
{
  int ik, iz;

  gflut.npts = GFLUT_NPTS;
  gflut.z_min = z_min;
  gflut.z_max = z_max;
  gflut.dz_inv = (GFLUT_NPTS - 1) / (z_max - z_min);

#ifdef SCALE_DEPENDENT
  gflut.kmin = pow(10.0, LOGKMIN);
  gflut.kmax = pow(10.0, LOGKMIN + (NkBINS - 1) * DELTALOGK);
#endif

  for (iz = 0; iz < GFLUT_NPTS; iz++)
    {
      double z = z_min + (double)iz / gflut.dz_inv;

      /* Hubble is k-independent */
      gflut.hubble[iz] = Hubble(z);

      for (ik = 0; ik < NkBINS; ik++)
        {
#ifdef SCALE_DEPENDENT
          double k = pow(10.0, LOGKMIN + ik * DELTALOGK);
#else
          double k = params.k_for_GM;
#endif

          gflut.grow1[ik][iz]    = GrowingMode(z, k);
          gflut.grow2[ik][iz]    = GrowingMode_2LPT(z, k);
          gflut.grow31[ik][iz]   = GrowingMode_3LPT_1(z, k);
          gflut.grow32[ik][iz]   = GrowingMode_3LPT_2(z, k);
          gflut.fomega1[ik][iz]  = fomega(z, k);
          gflut.fomega2[ik][iz]  = fomega_2LPT(z, k);
          gflut.fomega31[ik][iz] = fomega_3LPT_1(z, k);
          gflut.fomega32[ik][iz] = fomega_3LPT_2(z, k);
        }
    }

  gflut.initialized = 1;

  if (!ThisTask)
    printf("[GFLUT] Initialized %d points in z=[%g, %g] with %d k-bins (%.1f MB)\n",
           GFLUT_NPTS, z_min, z_max, NkBINS, sizeof(gflut_data) / 1.0e6);
}


/* ---- GFLUT validation: compare against GSL at random z values ---- */

void gflut_validate(void)
{
  if (!gflut.initialized)
    {
      if (!ThisTask)
        printf("[GFLUT] WARNING: validate called before init\n");
      return;
    }

  double max_err_grow1 = 0.0, max_err_hub = 0.0;
  int ntest = 100;
  double dz = (gflut.z_max - gflut.z_min) / (ntest - 1);

  for (int it = 0; it < ntest; it++)
    {
      double z = gflut.z_min + it * dz;
#ifdef SCALE_DEPENDENT
      double k = pow(10.0, LOGKMIN);
#else
      double k = params.k_for_GM;
#endif

      double gsl_val = GrowingMode(z, k);
      double lut_val = fast_GrowingMode(z, k);
      double err = (gsl_val != 0.0) ? fabs((lut_val - gsl_val) / gsl_val) : 0.0;
      if (err > max_err_grow1) max_err_grow1 = err;

      double gsl_hub = Hubble(z);
      double lut_hub = fast_Hubble(z);
      err = (gsl_hub != 0.0) ? fabs((lut_hub - gsl_hub) / gsl_hub) : 0.0;
      if (err > max_err_hub) max_err_hub = err;
    }

  if (!ThisTask)
    printf("[GFLUT] Validation: max relative error GrowingMode=%e, Hubble=%e\n",
           max_err_grow1, max_err_hub);
}


/* ============================================================
   Particle pre-computation infrastructure

   Pre-computes Eulerian positions and sigmaD for all particles
   before entering the fragmentation loop. This allows the
   parallel path to avoid calling GSL (which is not thread-safe).
   ============================================================ */

void precomp_allocate(int nparticles)
{
  precomp = (particle_precomp *)malloc(nparticles * sizeof(particle_precomp));
  if (!precomp)
    {
      printf("ERROR on task %d: cannot allocate precomp array (%d particles)\n",
             ThisTask, nparticles);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void precomp_free(void)
{
  if (precomp)
    {
      free(precomp);
      precomp = NULL;
    }
}


/* Thread-safe virial computation using pre-computed sigmaD */

static inline PRODFLOAT virial_fast(int mass, double sigmaD, int flag)
{
  /* Same as virial() but uses pre-computed sigmaD instead of calling GrowingMode.
     flag=1 for accretion, flag=0 for merging. */

  PRODFLOAT r2, rlag;

  rlag = pow((double)mass, 0.333333333333333);

  if (!flag)
    /* merging */
    r2 = pow(f_m * pow(rlag, espo) * (sigmaD > sigmaD0 ? 1.0 + (sigmaD - sigmaD0) * f_rm : 1.0), 2.0)
       + pow(f_200 * rlag, 2.0);
  else
    /* accretion */
    r2 = pow(f_a * pow(rlag, espo) * (sigmaD > sigmaD0 ? 1.0 + (sigmaD - sigmaD0) * f_ra : 1.0), 2.0)
       + pow(f_200 * rlag, 2.0);

  return r2;
}


/* Thread-safe version of set_weight using GFLUT */

static void fast_set_weight(pos_data *myobj)
{
  if (!ScaleDep.myseg)
    {
      myobj->w   = fast_GrowingMode(       myobj->z, myobj->myk) / fast_GrowingMode(       ScaleDep.z[ScaleDep.myseg], myobj->myk);
#ifdef TWO_LPT
      myobj->w2  = fast_GrowingMode_2LPT(  myobj->z, myobj->myk) / fast_GrowingMode_2LPT(  ScaleDep.z[ScaleDep.myseg], myobj->myk);
#ifdef THREE_LPT
      myobj->w31 = fast_GrowingMode_3LPT_1(myobj->z, myobj->myk) / fast_GrowingMode_3LPT_1(ScaleDep.z[ScaleDep.myseg], myobj->myk);
      myobj->w32 = fast_GrowingMode_3LPT_2(myobj->z, myobj->myk) / fast_GrowingMode_3LPT_2(ScaleDep.z[ScaleDep.myseg], myobj->myk);
#endif
#endif
    }
  else
    {
      myobj->w   = (fast_GrowingMode(                         myobj->z, myobj->myk) - fast_GrowingMode(       ScaleDep.z[ScaleDep.myseg-1], myobj->myk)) /
                   (fast_GrowingMode(       ScaleDep.z[ScaleDep.myseg], myobj->myk) - fast_GrowingMode(       ScaleDep.z[ScaleDep.myseg-1], myobj->myk) );
#ifdef TWO_LPT
      myobj->w2  = (fast_GrowingMode_2LPT(                    myobj->z, myobj->myk) - fast_GrowingMode_2LPT(  ScaleDep.z[ScaleDep.myseg-1], myobj->myk)) /
                   (fast_GrowingMode_2LPT(  ScaleDep.z[ScaleDep.myseg], myobj->myk) - fast_GrowingMode_2LPT(  ScaleDep.z[ScaleDep.myseg-1], myobj->myk) );
#ifdef THREE_LPT
      myobj->w31 = (fast_GrowingMode_3LPT_1(                  myobj->z, myobj->myk) - fast_GrowingMode_3LPT_1(ScaleDep.z[ScaleDep.myseg-1], myobj->myk)) /
                   (fast_GrowingMode_3LPT_1(ScaleDep.z[ScaleDep.myseg], myobj->myk) - fast_GrowingMode_3LPT_1(ScaleDep.z[ScaleDep.myseg-1], myobj->myk));
      myobj->w32 = (fast_GrowingMode_3LPT_2(                  myobj->z, myobj->myk) - fast_GrowingMode_3LPT_2(ScaleDep.z[ScaleDep.myseg-1], myobj->myk)) /
                   (fast_GrowingMode_3LPT_2(ScaleDep.z[ScaleDep.myseg], myobj->myk) - fast_GrowingMode_3LPT_2(ScaleDep.z[ScaleDep.myseg-1], myobj->myk) );
#endif
#endif
    }
}


/* Pre-compute Eulerian positions and sigmaD for all stored particles */

void precompute_particles(void)
{
  int S = Smoothing.Nsmooth - 1;
  double sqrt_var = sqrt(Smoothing.TrueVariance[S]);

#ifdef SCALE_DEPENDENT
  double k_point = Smoothing.k_GM_displ[S];
  k_dens_global = Smoothing.k_GM_dens[S];
#else
  double k_point = params.k_for_GM;
  k_dens_global = params.k_for_GM;
#endif

  int N = (int)subbox.Nstored;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int p = 0; p < N; p++)
    {
      pos_data myobj;
      int ibox, jbox, kbox;

      INDEX_TO_COORD(frag_pos[p], ibox, jbox, kbox, subbox.Lgwbl);

      double z = (double)frag[p].Fmax - 1.0;
      myobj.z = z;
      myobj.myk = k_point;

      fast_set_weight(&myobj);

      myobj.M = 1;
      myobj.q[0] = ibox + SHIFT;
      myobj.q[1] = jbox + SHIFT;
      myobj.q[2] = kbox + SHIFT;
      myobj.v[0] = frag[p].Vel[0];
      myobj.v[1] = frag[p].Vel[1];
      myobj.v[2] = frag[p].Vel[2];
#ifdef TWO_LPT
      myobj.v2[0] = frag[p].Vel_2LPT[0];
      myobj.v2[1] = frag[p].Vel_2LPT[1];
      myobj.v2[2] = frag[p].Vel_2LPT[2];
#ifdef THREE_LPT
      myobj.v31[0] = frag[p].Vel_3LPT_1[0];
      myobj.v31[1] = frag[p].Vel_3LPT_1[1];
      myobj.v31[2] = frag[p].Vel_3LPT_1[2];
      myobj.v32[0] = frag[p].Vel_3LPT_2[0];
      myobj.v32[1] = frag[p].Vel_3LPT_2[1];
      myobj.v32[2] = frag[p].Vel_3LPT_2[2];
#endif
#endif
#ifdef RECOMPUTE_DISPLACEMENTS
      myobj.v_prev[0] = frag[p].Vel_prev[0];
      myobj.v_prev[1] = frag[p].Vel_prev[1];
      myobj.v_prev[2] = frag[p].Vel_prev[2];
#ifdef TWO_LPT
      myobj.v2_prev[0] = frag[p].Vel_2LPT_prev[0];
      myobj.v2_prev[1] = frag[p].Vel_2LPT_prev[1];
      myobj.v2_prev[2] = frag[p].Vel_2LPT_prev[2];
#ifdef THREE_LPT
      myobj.v31_prev[0] = frag[p].Vel_3LPT_1_prev[0];
      myobj.v31_prev[1] = frag[p].Vel_3LPT_1_prev[1];
      myobj.v31_prev[2] = frag[p].Vel_3LPT_1_prev[2];
      myobj.v32_prev[0] = frag[p].Vel_3LPT_2_prev[0];
      myobj.v32_prev[1] = frag[p].Vel_3LPT_2_prev[1];
      myobj.v32_prev[2] = frag[p].Vel_3LPT_2_prev[2];
#endif
#endif
#endif

      /* Compute Eulerian position */
      for (int dim = 0; dim < 3; dim++)
        precomp[p].euler[dim] = q2x(dim, &myobj, subbox.pbc[dim],
                                    (double)subbox.Lgwbl[dim], ORDER_FOR_GROUPS);

      /* Compute sigmaD = sqrt(TrueVariance) * GrowingMode(z, k_dens) */
      precomp[p].sigmaD = sqrt_var * fast_GrowingMode(z, k_dens_global);
    }

  if (!ThisTask)
    printf("[PRECOMP] Pre-computed %d particles\n", N);
}


#ifdef _OPENMP
/* ============================================================
   Subdomain decomposition: split MPI sub-box into smaller
   sub-cubes for OpenMP thread-level parallelism.

   Each thread processes particles in its own subdomain
   independently, creating groups with non-overlapping IDs.
   Boundary interactions are reconciled in a sequential phase.
   ============================================================ */

typedef struct {
  int lo[3];               /* lower corner in Lgwbl coordinates */
  int hi[3];               /* upper corner (exclusive) in Lgwbl coordinates */
  int ngroups_local;       /* number of groups created in this subdomain */
  int group_base;          /* first group ID for this subdomain */
  unsigned long long counters[NCOUNTERS]; /* per-subdomain counters */
} subdomain_info;

/* Boundary pair: records a cross-subdomain interaction to be resolved later */
typedef struct {
  int particle_idx;        /* index of particle in frag-order */
  int neighbor_idx;        /* index of neighbor in frag-order */
  int neighbor_subdomain;  /* subdomain ID of the neighbor */
} boundary_pair;

static subdomain_info *subdomains = NULL;
static int n_subdomains = 0;
static int *subdomain_map = NULL;  /* 3D grid -> subdomain index, for fast lookup */


/* Determine which subdomain a coordinate belongs to */

static inline int subcube_of_coord(int i, int j, int k)
{
  if (subdomain_map)
    return subdomain_map[COORD_TO_INDEX(i, j, k, subbox.Lgwbl)];
  return 0;
}


/* Find the best 3D factorization of nthreads that minimizes surface/volume ratio
   for a box of dimensions Lgwbl[3] */

static void factorize_3d(int n, int Lx, int Ly, int Lz, int *nx, int *ny, int *nz)
{
  double best_ratio = 1.0e30;
  *nx = *ny = *nz = 1;

  for (int ix = 1; ix <= n; ix++)
    {
      if (n % ix != 0) continue;
      int rem = n / ix;
      for (int iy = 1; iy <= rem; iy++)
        {
          if (rem % iy != 0) continue;
          int iz = rem / iy;

          /* Compute surface/volume ratio for this decomposition */
          double sx = (double)Lx / ix;
          double sy = (double)Ly / iy;
          double sz = (double)Lz / iz;
          double vol = sx * sy * sz;
          double surf = 2.0 * (sx*sy + sy*sz + sz*sx);
          double ratio = surf / vol;

          if (ratio < best_ratio)
            {
              best_ratio = ratio;
              *nx = ix;
              *ny = iy;
              *nz = iz;
            }
        }
    }
}


/* Create subdomain partitioning for nthreads OpenMP threads */

static void partition_into_subdomains(int nthreads)
{
  int nx, ny, nz;
  int sd;

  factorize_3d(nthreads, subbox.Lgwbl[_x_], subbox.Lgwbl[_y_], subbox.Lgwbl[_z_],
               &nx, &ny, &nz);

  n_subdomains = nx * ny * nz;

  if (subdomains) free(subdomains);
  subdomains = (subdomain_info *)calloc(n_subdomains, sizeof(subdomain_info));

  /* Compute the maximum number of groups per subdomain.
     We allocate evenly from the PredNpeaks budget. */
  int groups_per_thread = (subbox.PredNpeaks - FILAMENT - 2) / n_subdomains;

  for (int ix = 0; ix < nx; ix++)
    for (int iy = 0; iy < ny; iy++)
      for (int iz = 0; iz < nz; iz++)
        {
          sd = iz + nz * (iy + ny * ix);
          subdomains[sd].lo[_x_] = (subbox.Lgwbl[_x_] * ix) / nx;
          subdomains[sd].hi[_x_] = (subbox.Lgwbl[_x_] * (ix + 1)) / nx;
          subdomains[sd].lo[_y_] = (subbox.Lgwbl[_y_] * iy) / ny;
          subdomains[sd].hi[_y_] = (subbox.Lgwbl[_y_] * (iy + 1)) / ny;
          subdomains[sd].lo[_z_] = (subbox.Lgwbl[_z_] * iz) / nz;
          subdomains[sd].hi[_z_] = (subbox.Lgwbl[_z_] * (iz + 1)) / nz;
          subdomains[sd].ngroups_local = 0;
          subdomains[sd].group_base = FILAMENT + 1 + sd * groups_per_thread;
          memset(subdomains[sd].counters, 0, NCOUNTERS * sizeof(unsigned long long));
        }

  /* Build the spatial lookup map: for each grid cell, store its subdomain index */
  int Ntot = subbox.Lgwbl[_x_] * subbox.Lgwbl[_y_] * subbox.Lgwbl[_z_];
  if (subdomain_map) free(subdomain_map);
  subdomain_map = (int *)malloc(Ntot * sizeof(int));

  for (int i = 0; i < subbox.Lgwbl[_x_]; i++)
    for (int j = 0; j < subbox.Lgwbl[_y_]; j++)
      for (int k = 0; k < subbox.Lgwbl[_z_]; k++)
        {
          int idx = COORD_TO_INDEX(i, j, k, subbox.Lgwbl);
          /* Find which subdomain this cell falls in */
          for (sd = 0; sd < n_subdomains; sd++)
            if (i >= subdomains[sd].lo[_x_] && i < subdomains[sd].hi[_x_] &&
                j >= subdomains[sd].lo[_y_] && j < subdomains[sd].hi[_y_] &&
                k >= subdomains[sd].lo[_z_] && k < subdomains[sd].hi[_z_])
              {
                subdomain_map[idx] = sd;
                break;
              }
        }

  if (!ThisTask)
    printf("[SUBDOMAIN] Partitioned %dx%dx%d grid into %d subdomains (%dx%dx%d)\n",
           subbox.Lgwbl[_x_], subbox.Lgwbl[_y_], subbox.Lgwbl[_z_],
           n_subdomains, nx, ny, nz);
}


/* ============================================================
   Thread-safe versions of key functions for the parallel path.

   These avoid:
   1. Global obj1/obj2 (use thread-local pos_data instead)
   2. GSL spline calls via set_weight (use fast_set_weight instead)
   3. Global state mutations from accretion/merge_groups

   The _ts suffix means "thread-safe".
   ============================================================ */

/* Thread-safe set_point: uses fast_set_weight instead of set_weight */
static void set_point_ts(int i, int j, int k, int ind, PRODFLOAT F, pos_data *myobj)
{
  myobj->z = F - 1.0;
#ifdef SCALE_DEPENDENT
  int S = Smoothing.Nsmooth - 1;
  myobj->myk = Smoothing.k_GM_displ[S];
#else
  myobj->myk = params.k_for_GM;
#endif
  fast_set_weight(myobj);
  myobj->M = 1;
  myobj->q[0] = i + SHIFT;
  myobj->q[1] = j + SHIFT;
  myobj->q[2] = k + SHIFT;
  myobj->v[0] = frag[ind].Vel[0];
  myobj->v[1] = frag[ind].Vel[1];
  myobj->v[2] = frag[ind].Vel[2];
#ifdef TWO_LPT
  myobj->v2[0] = frag[ind].Vel_2LPT[0];
  myobj->v2[1] = frag[ind].Vel_2LPT[1];
  myobj->v2[2] = frag[ind].Vel_2LPT[2];
#ifdef THREE_LPT
  myobj->v31[0] = frag[ind].Vel_3LPT_1[0];
  myobj->v31[1] = frag[ind].Vel_3LPT_1[1];
  myobj->v31[2] = frag[ind].Vel_3LPT_1[2];
  myobj->v32[0] = frag[ind].Vel_3LPT_2[0];
  myobj->v32[1] = frag[ind].Vel_3LPT_2[1];
  myobj->v32[2] = frag[ind].Vel_3LPT_2[2];
#endif
#endif
#ifdef RECOMPUTE_DISPLACEMENTS
  myobj->v_prev[0] = frag[ind].Vel_prev[0];
  myobj->v_prev[1] = frag[ind].Vel_prev[1];
  myobj->v_prev[2] = frag[ind].Vel_prev[2];
#ifdef TWO_LPT
  myobj->v2_prev[0] = frag[ind].Vel_2LPT_prev[0];
  myobj->v2_prev[1] = frag[ind].Vel_2LPT_prev[1];
  myobj->v2_prev[2] = frag[ind].Vel_2LPT_prev[2];
#ifdef THREE_LPT
  myobj->v31_prev[0] = frag[ind].Vel_3LPT_1_prev[0];
  myobj->v31_prev[1] = frag[ind].Vel_3LPT_1_prev[1];
  myobj->v31_prev[2] = frag[ind].Vel_3LPT_1_prev[2];
  myobj->v32_prev[0] = frag[ind].Vel_3LPT_2_prev[0];
  myobj->v32_prev[1] = frag[ind].Vel_3LPT_2_prev[1];
  myobj->v32_prev[2] = frag[ind].Vel_3LPT_2_prev[2];
#endif
#endif
#endif
}


/* Thread-safe set_obj: uses fast_set_weight instead of set_weight */
static void set_obj_ts(int grp, PRODFLOAT F, pos_data *myobj)
{
  myobj->M = groups[grp].Mass;
  myobj->z = F - 1.0;

#ifdef SCALE_DEPENDENT
  myobj->R = pow((double)(myobj->M) * 3.0 / 4.0 / PI, 1.0/3.0) * params.InterPartDist;
  double interp = (1.0 - myobj->R / Smoothing.Rad_GM[0]) * (double)(Smoothing.Nsmooth - 1);
  interp = (interp < 0.0 ? 0.0 : interp);
  int indx = (int)interp;
  double w = interp - (double)indx;
  myobj->myk = pow(10.0, log10(Smoothing.k_GM_displ[indx]) * (1.0 - w) + log10(Smoothing.k_GM_displ[indx + 1]) * w);
#else
  myobj->myk = params.k_for_GM;
#endif

  fast_set_weight(myobj);

  myobj->M = groups[grp].Mass;
  for (int i = 0; i < 3; i++)
    {
      myobj->q[i] = groups[grp].Pos[i];
      myobj->v[i] = groups[grp].Vel[i];
#ifdef TWO_LPT
      myobj->v2[i] = groups[grp].Vel_2LPT[i];
#ifdef THREE_LPT
      myobj->v31[i] = groups[grp].Vel_3LPT_1[i];
      myobj->v32[i] = groups[grp].Vel_3LPT_2[i];
#endif
#endif
#ifdef RECOMPUTE_DISPLACEMENTS
      myobj->v_prev[i] = groups[grp].Vel_prev[i];
#ifdef TWO_LPT
      myobj->v2_prev[i] = groups[grp].Vel_2LPT_prev[i];
#ifdef THREE_LPT
      myobj->v31_prev[i] = groups[grp].Vel_3LPT_1_prev[i];
      myobj->v32_prev[i] = groups[grp].Vel_3LPT_2_prev[i];
#endif
#endif
#endif
    }
}


/* Thread-safe accretion: uses thread-local pos_data objects and fast_set_weight */
static void accretion_ts(int group, int i, int j, int k, int indx, PRODFLOAT F,
                          pos_data *lo1, pos_data *lo2)
{
  /* Safety: skip if group was already merged away */
  if (groups[group].point < 0)
    return;

  set_obj_ts(group, F, lo1);
  set_point_ts(i, j, k, indx, F, lo2);
  update(lo1, lo2);
  set_group(group, lo1);

  if (groups[group].Mass >= params.MinHaloMass && groups[group].t_appear == -1)
    {
      groups[group].t_appear = F;
#ifdef SNAPSHOT
      int i1 = groups[group].point;
      while (linking_list[i1] != groups[group].point)
        {
          frag[i1].zacc = F - 1.0;
          i1 = linking_list[i1];
        }
      frag[i1].zacc = F - 1.0;
#endif
    }

  group_ID[indx] = group;
  linking_list[groups[group].bottom] = indx;
  groups[group].bottom = indx;
  linking_list[indx] = groups[group].point;

#ifdef SNAPSHOT
  if (groups[group].Mass >= params.MinHaloMass)
    frag[indx].zacc = F - 1.0;
#endif
}


/* Thread-safe merge_groups: uses thread-local pos_data objects and fast_set_weight */
static void merge_groups_ts(int grp1, int grp2, PRODFLOAT time,
                             pos_data *lo1, pos_data *lo2)
{
  int i1;

  /* Safety: if grp2 was already merged, its point is -1. Skip. */
  if (groups[grp2].point < 0 || groups[grp1].point < 0)
    return;

#ifdef SNAPSHOT
  if ((groups[grp1].t_appear == -1 || groups[grp2].t_appear == -1)
      && (groups[grp1].Mass + groups[grp2].Mass >= params.MinHaloMass))
    {
      if (groups[grp1].t_appear == -1)
        {
          i1 = groups[grp1].point;
          while (linking_list[i1] != groups[grp1].point)
            { frag[i1].zacc = time - 1.0; i1 = linking_list[i1]; }
          frag[i1].zacc = time - 1.0;
        }
      if (groups[grp2].t_appear == -1)
        {
          i1 = groups[grp2].point;
          while (linking_list[i1] != groups[grp2].point)
            { frag[i1].zacc = time - 1.0; i1 = linking_list[i1]; }
          frag[i1].zacc = time - 1.0;
        }
    }
#endif

  i1 = groups[grp2].point;
  while (linking_list[i1] != groups[grp2].point)
    { group_ID[i1] = grp1; i1 = linking_list[i1]; }
  group_ID[i1] = grp1;

  linking_list[groups[grp1].bottom] = groups[grp2].point;
  linking_list[groups[grp2].bottom] = groups[grp1].point;
  groups[grp1].bottom = groups[grp2].bottom;
  groups[grp2].point = -1;
  groups[grp2].bottom = -1;

  if (groups[grp1].Mass >= params.MinHaloMass && groups[grp2].Mass >= params.MinHaloMass)
    update_history(grp1, grp2, time);

  set_obj_ts(grp1, time, lo1);
  set_obj_ts(grp2, time, lo2);
  update(lo1, lo2);
  set_group(grp1, lo1);

  if (groups[grp1].Mass >= params.MinHaloMass && groups[grp1].t_appear == -1)
    groups[grp1].t_appear = time;
}


/* Thread-safe condition_for_accretion.
   For calls 1,2,3: uses precomp[ind] for particle position (avoids set_point_ts + q2x).
   For call 4 (filament re-accretion at different Fmax): must recompute. */
static void condition_for_accretion_ts(int call, int i, int j, int k, int ind,
                                        PRODFLOAT Fmax, int grp,
                                        double *dd, double *rr,
                                        double sigmaD_part,
                                        pos_data *myobj1, pos_data *myobj2)
{
  double dx, dy, dz, d2;

  /* Safety: if group was merged away, report no accretion */
  if (groups[grp].point < 0)
    { *rr = 1.0; *dd = 1.0e30; return; }

  /* Use pre-computed sigmaD for virial radius */
  if (precomp != NULL && call != 4)
    *rr = virial_fast(groups[grp].Mass, precomp[ind].sigmaD, 1);
  else
    *rr = virial_fast(groups[grp].Mass, sigmaD_part, 1);
  *dd = 100.0 * (*rr);

  /* Group position: must compute (group state changes dynamically) */
  set_obj_ts(grp, Fmax, myobj2);

  if (precomp != NULL && call != 4)
    {
      /* Use pre-computed particle Eulerian position — avoids
         set_point_ts + fast_set_weight + q2x overhead */
      PRODFLOAT gx;

      gx = q2x(0, myobj2, subbox.pbc[0], (double)subbox.Lgwbl[0], ORDER_FOR_GROUPS);
      dx = precomp[ind].euler[0] - gx;
      if (subbox.pbc[0])
        { PRODFLOAT h = (PRODFLOAT)subbox.Lgwbl[0] / 2.; if (dx > h) dx -= subbox.Lgwbl[0]; if (dx < -h) dx += subbox.Lgwbl[0]; }
      d2 = dx * dx;
      if (d2 < *rr)
        {
          gx = q2x(1, myobj2, subbox.pbc[1], (double)subbox.Lgwbl[1], ORDER_FOR_GROUPS);
          dy = precomp[ind].euler[1] - gx;
          if (subbox.pbc[1])
            { PRODFLOAT h = (PRODFLOAT)subbox.Lgwbl[1] / 2.; if (dy > h) dy -= subbox.Lgwbl[1]; if (dy < -h) dy += subbox.Lgwbl[1]; }
          d2 += dy * dy;
          if (d2 < *rr)
            {
              gx = q2x(2, myobj2, subbox.pbc[2], (double)subbox.Lgwbl[2], ORDER_FOR_GROUPS);
              dz = precomp[ind].euler[2] - gx;
              if (subbox.pbc[2])
                { PRODFLOAT h = (PRODFLOAT)subbox.Lgwbl[2] / 2.; if (dz > h) dz -= subbox.Lgwbl[2]; if (dz < -h) dz += subbox.Lgwbl[2]; }
              d2 += dz * dz;
              if (d2 <= *rr)
                *dd = d2;
            }
        }
    }
  else
    {
      /* Fallback: compute particle position from scratch (call 4 or no precomp) */
      set_point_ts(i, j, k, ind, Fmax, myobj1);

      dx = distance(0, myobj1, myobj2);
      d2 = dx * dx;
      if (d2 < *rr)
        {
          dy = distance(1, myobj1, myobj2);
          d2 += dy * dy;
          if (d2 < *rr)
            {
              dz = distance(2, myobj1, myobj2);
              d2 += dz * dz;
              if (d2 <= *rr)
                *dd = d2;
            }
        }
    }
}


/* Thread-safe condition_for_merging */
static void condition_for_merging_ts(PRODFLOAT Fmax, int grp1, int grp2,
                                      int *merge_flag,
                                      double sigmaD1, double sigmaD2,
                                      pos_data *myobj1, pos_data *myobj2)
{
  double rvir1, rvir2, dd, rr, dx, dy, dz;

  *merge_flag = 0;
  /* Safety: skip if either group was merged away */
  if (groups[grp1].point < 0 || groups[grp2].point < 0)
    return;
  rvir1 = virial_fast(groups[grp1].Mass, sigmaD1, 0);
  rvir2 = virial_fast(groups[grp2].Mass, sigmaD2, 0);
  rr = (rvir1 > rvir2 ? rvir1 : rvir2);

  set_obj_ts(grp1, Fmax, myobj1);
  set_obj_ts(grp2, Fmax, myobj2);

  dx = distance(0, myobj1, myobj2);
  dd = dx * dx;
  if (dd < rr)
    {
      dy = distance(1, myobj1, myobj2);
      dd += dy * dy;
      if (dd < rr)
        {
          dz = distance(2, myobj1, myobj2);
          dd += dz * dz;
          if (dd <= rr)
            *merge_flag = 1;
        }
    }
}


/* ============================================================
   Parallel build_groups: Phase 1 (subdomain-parallel) and
   Phase 2 (boundary reconciliation)

   Architecture:
   - Phase 1: Each thread processes particles in its subdomain
     independently. Particles with neighbors in other subdomains
     are flagged as "boundary pending" and deferred.
   - Phase 2: Boundary particles are re-processed sequentially,
     checking accretion/merging conditions against groups from
     other subdomains.

   IMPORTANT: PLC (Past Light Cone) reconstruction is NOT
   parallelized. It is handled in the reconciliation phase or
   through the serial fallback path.
   ============================================================ */

static int build_groups_parallel(int Npeaks, int nstep, unsigned long long *out_counters)
{
  int nthreads = omp_get_max_threads();

  /* Partition the sub-box into OpenMP subdomains */
  partition_into_subdomains(nthreads);

  /* Maximum boundary pairs per thread (conservative estimate) */
  int max_boundary_per_thread = (nstep / nthreads) + 1024;

  /* Allocate per-thread boundary arrays */
  boundary_pair **thread_boundary = (boundary_pair **)malloc(nthreads * sizeof(boundary_pair *));
  int *thread_nboundary = (int *)calloc(nthreads, sizeof(int));
  for (int t = 0; t < nthreads; t++)
    thread_boundary[t] = (boundary_pair *)malloc(max_boundary_per_thread * sizeof(boundary_pair));

  /* ================================================================
     PHASE 1: Parallel processing of subdomains
     Each thread handles particles in its subdomain independently.
     Groups are created with non-overlapping ID ranges.
     Cross-subdomain interactions are recorded for Phase 2.
     ================================================================ */

  double t_start = MPI_Wtime(), t_phase1, t_validate, t_phase2, t_partition;

  if (!ThisTask)
    printf("[PARALLEL] Phase 1: processing %d particles across %d subdomains\n",
           nstep, n_subdomains);

  /* ---- Pre-partition particles by subdomain ----
     This avoids each thread iterating over ALL particles.
     Each subdomain gets a list of particle indices, already in Fmax order
     (since we iterate in frag-order which IS Fmax-descending). */

  int *sd_count = (int *)calloc(n_subdomains, sizeof(int));
  int *sd_offset = (int *)malloc(n_subdomains * sizeof(int));
  int *sd_particles = (int *)malloc(nstep * sizeof(int));

  /* First pass: count particles per subdomain */
  for (int p = 0; p < nstep; p++)
    {
      int ii, jj, kk;
      INDEX_TO_COORD(frag_pos[p], ii, jj, kk, subbox.Lgwbl);
      sd_count[subcube_of_coord(ii, jj, kk)]++;
    }

  /* Compute offsets */
  sd_offset[0] = 0;
  for (int sd = 1; sd < n_subdomains; sd++)
    sd_offset[sd] = sd_offset[sd - 1] + sd_count[sd - 1];

  /* Second pass: fill particle lists (use sd_count as cursor) */
  memset(sd_count, 0, n_subdomains * sizeof(int));
  for (int p = 0; p < nstep; p++)
    {
      int ii, jj, kk;
      INDEX_TO_COORD(frag_pos[p], ii, jj, kk, subbox.Lgwbl);
      int sd = subcube_of_coord(ii, jj, kk);
      sd_particles[sd_offset[sd] + sd_count[sd]] = p;
      sd_count[sd]++;
    }

  if (!ThisTask)
    {
      for (int sd = 0; sd < n_subdomains; sd++)
        printf("[PARTITION] Subdomain %d: %d particles\n", sd, sd_count[sd]);
    }

  t_partition = MPI_Wtime() - t_start;

  /* Compute sigmaD at the current redshift for virial_fast. */
  int S = Smoothing.Nsmooth - 1;
  double sqrt_var = sqrt(Smoothing.TrueVariance[S]);

  double t1 = MPI_Wtime();

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    pos_data obj1_local, obj2_local;
    int my_nboundary = 0;

    /* Determine which subdomains this thread owns */
    int sd_start = (n_subdomains * tid) / nthreads;
    int sd_end = (n_subdomains * (tid + 1)) / nthreads;

    int my_peaks = 0, my_accretions = 0, my_filaments = 0, my_boundary = 0;
    int my_filament_mass = 0;  /* thread-local filament counter (avoids atomic) */

    /* Iterate ONLY over particles in this thread's subdomains */
    for (int my_sd = sd_start; my_sd < sd_end; my_sd++)
      {
        int base = sd_offset[my_sd];
        int count = sd_count[my_sd];

        for (int pidx = 0; pidx < count; pidx++)
          {
        int iz = sd_particles[base + pidx];
        int ibox, jbox, kbox;

        INDEX_TO_COORD(frag_pos[iz], ibox, jbox, kbox, subbox.Lgwbl);

        /* Skip border particles (no PBC) */
        int skip = 0;
        if (!subbox.pbc[_x_] && (ibox == 0 || ibox == subbox.Lgwbl[_x_] - 1)) ++skip;
        if (!subbox.pbc[_y_] && (jbox == 0 || jbox == subbox.Lgwbl[_y_] - 1)) ++skip;
        if (!subbox.pbc[_z_] && (kbox == 0 || kbox == subbox.Lgwbl[_z_] - 1)) ++skip;

        if (skip)
          {
            /* Filament particle at domain border */
            group_ID[iz] = FILAMENT;
            linking_list[iz] = iz;
            continue;
          }

        /* Check neighbors */
        int neigh[NV] = {0};
        int fil_list[NV][4];
        int nf = 0;
        int peak_cond = 1;
        int has_cross_boundary = 0;
        int i1, j1, k1, pos;

        for (int nn = 0; nn < NV; nn++)
          {
            switch (nn)
              {
              case 0:
                i1 = (subbox.pbc[_x_] && ibox == 0 ? subbox.Lgwbl[_x_] - 1 : ibox - 1);
                j1 = jbox; k1 = kbox; break;
              case 1:
                i1 = (subbox.pbc[_x_] && ibox == subbox.Lgwbl[_x_] - 1 ? 0 : ibox + 1);
                j1 = jbox; k1 = kbox; break;
              case 2:
                i1 = ibox;
                j1 = (subbox.pbc[_y_] && jbox == 0 ? subbox.Lgwbl[_y_] - 1 : jbox - 1);
                k1 = kbox; break;
              case 3:
                i1 = ibox;
                j1 = (subbox.pbc[_y_] && jbox == subbox.Lgwbl[_y_] - 1 ? 0 : jbox + 1);
                k1 = kbox; break;
              case 4:
                i1 = ibox; j1 = jbox;
                k1 = (subbox.pbc[_z_] && kbox == 0 ? subbox.Lgwbl[_z_] - 1 : kbox - 1);
                break;
              case 5:
                i1 = ibox; j1 = jbox;
                k1 = (subbox.pbc[_z_] && kbox == subbox.Lgwbl[_z_] - 1 ? 0 : kbox + 1);
                break;
              }

            pos = find_location(i1, j1, k1);
            if (pos >= 0)
              {
                neigh[nn] = group_ID[pos];
                peak_cond &= (frag[iz].Fmax > frag[pos].Fmax);

                /* Check if neighbor is in a different subdomain.
                   We must defer ALL particles with cross-boundary neighbors,
                   not just those whose neighbor already has a group,
                   because the neighbor's thread might not have processed it yet. */
                int nb_sd = subcube_of_coord(i1, j1, k1);
                if (nb_sd != my_sd)
                  has_cross_boundary = 1;
              }
            else
              neigh[nn] = 0;

            /* Store filament neighbors separately */
            if (neigh[nn] == FILAMENT)
              {
                neigh[nn] = 0;
                fil_list[nf][0] = i1;
                fil_list[nf][1] = j1;
                fil_list[nf][2] = k1;
                fil_list[nf][3] = pos;
                nf++;
              }
          }

        clean_list(neigh);
        int neigrp = 0;
        for (int nn = 0; nn < NV; nn++)
          if (neigh[nn] > FILAMENT) neigrp++;

        /* If this particle has cross-boundary neighbors, defer NON-PEAK
           particles to Phase 2.  Peak detection is safe (only compares
           Fmax values, no group state needed).  But accretion/merging
           decisions require seeing all neighbor groups, which might not
           be assigned yet by other threads. */
        if (has_cross_boundary && !peak_cond)
          {
            if (my_nboundary < max_boundary_per_thread)
              {
                /* Record this particle for boundary reconciliation.
                   We mark it as filament temporarily. */
                group_ID[iz] = FILAMENT;
                linking_list[iz] = iz;

                thread_boundary[tid][my_nboundary].particle_idx = iz;
                thread_boundary[tid][my_nboundary].neighbor_idx = -1;
                thread_boundary[tid][my_nboundary].neighbor_subdomain = -1;
                my_nboundary++;
                my_boundary++;
              }
            continue;
          }

        /* Compute sigmaD for this particle for virial_fast */
        double z_part = (double)frag[iz].Fmax - 1.0;
        double sigmaD_part = sqrt_var * fast_GrowingMode(z_part, k_dens_global);

        int good_particle_local =
          (ibox >= subbox.safe[_x_] && ibox < subbox.Lgwbl[_x_] - subbox.safe[_x_] &&
           jbox >= subbox.safe[_y_] && jbox < subbox.Lgwbl[_y_] - subbox.safe[_y_] &&
           kbox >= subbox.safe[_z_] && kbox < subbox.Lgwbl[_z_] - subbox.safe[_z_]);

        unsigned long long int pname =
          COORD_TO_INDEX((long long)((ibox + subbox.stabl[_x_] + MyGrids[0].GSglobal[_x_]) % MyGrids[0].GSglobal[_x_]),
                         (long long)((jbox + subbox.stabl[_y_] + MyGrids[0].GSglobal[_y_]) % MyGrids[0].GSglobal[_y_]),
                         (long long)((kbox + subbox.stabl[_z_] + MyGrids[0].GSglobal[_z_]) % MyGrids[0].GSglobal[_z_]),
                         MyGrids[0].GSglobal);

        if (neigrp > 0 && good_particle_local && neigrp <= 6)
          subdomains[my_sd].counters[neigrp]++;

        int accrflag = 0;
        int to_group = 0;

        /* ----- Peak condition ----- */
        if (peak_cond)
          {
            /* Allocate a new group ID from this subdomain's range.
               No atomic needed: each thread has exclusive access to its subdomain. */
            int new_grp = subdomains[my_sd].ngroups_local++;

            new_grp += subdomains[my_sd].group_base;

            /* Safety: check against allocated groups[] array size,
               NOT against Npeaks (which is the total peak count).
               In subdomain mode, group IDs are non-contiguous and
               can be much larger than Npeaks. */
            if (new_grp >= (int)subbox.PredNpeaks)
              {
                printf("ERROR: task %d thread %d: group ID %d exceeds PredNpeaks %d\n",
                       ThisTask, tid, new_grp, subbox.PredNpeaks);
                continue;
              }

            my_peaks++;
            if (good_particle_local)
              subdomains[my_sd].counters[0]++;

            groups[new_grp].t_peak = frag[iz].Fmax;
            groups[new_grp].t_appear = -1;
            groups[new_grp].t_merge = -1;
            groups[new_grp].Pos[0] = ibox + SHIFT;
            groups[new_grp].Pos[1] = jbox + SHIFT;
            groups[new_grp].Pos[2] = kbox + SHIFT;
            groups[new_grp].Vel[0] = frag[iz].Vel[0];
            groups[new_grp].Vel[1] = frag[iz].Vel[1];
            groups[new_grp].Vel[2] = frag[iz].Vel[2];
#ifdef TWO_LPT
            groups[new_grp].Vel_2LPT[0] = frag[iz].Vel_2LPT[0];
            groups[new_grp].Vel_2LPT[1] = frag[iz].Vel_2LPT[1];
            groups[new_grp].Vel_2LPT[2] = frag[iz].Vel_2LPT[2];
#ifdef THREE_LPT
            groups[new_grp].Vel_3LPT_1[0] = frag[iz].Vel_3LPT_1[0];
            groups[new_grp].Vel_3LPT_1[1] = frag[iz].Vel_3LPT_1[1];
            groups[new_grp].Vel_3LPT_1[2] = frag[iz].Vel_3LPT_1[2];
            groups[new_grp].Vel_3LPT_2[0] = frag[iz].Vel_3LPT_2[0];
            groups[new_grp].Vel_3LPT_2[1] = frag[iz].Vel_3LPT_2[1];
            groups[new_grp].Vel_3LPT_2[2] = frag[iz].Vel_3LPT_2[2];
#endif
#endif
#ifdef RECOMPUTE_DISPLACEMENTS
            groups[new_grp].Vel_prev[0] = frag[iz].Vel_prev[0];
            groups[new_grp].Vel_prev[1] = frag[iz].Vel_prev[1];
            groups[new_grp].Vel_prev[2] = frag[iz].Vel_prev[2];
#ifdef TWO_LPT
            groups[new_grp].Vel_2LPT_prev[0] = frag[iz].Vel_2LPT_prev[0];
            groups[new_grp].Vel_2LPT_prev[1] = frag[iz].Vel_2LPT_prev[1];
            groups[new_grp].Vel_2LPT_prev[2] = frag[iz].Vel_2LPT_prev[2];
#ifdef THREE_LPT
            groups[new_grp].Vel_3LPT_1_prev[0] = frag[iz].Vel_3LPT_1_prev[0];
            groups[new_grp].Vel_3LPT_1_prev[1] = frag[iz].Vel_3LPT_1_prev[1];
            groups[new_grp].Vel_3LPT_1_prev[2] = frag[iz].Vel_3LPT_1_prev[2];
            groups[new_grp].Vel_3LPT_2_prev[0] = frag[iz].Vel_3LPT_2_prev[0];
            groups[new_grp].Vel_3LPT_2_prev[1] = frag[iz].Vel_3LPT_2_prev[1];
            groups[new_grp].Vel_3LPT_2_prev[2] = frag[iz].Vel_3LPT_2_prev[2];
#endif
#endif
#endif
            groups[new_grp].Mass = 1;
            groups[new_grp].name = pname;
            groups[new_grp].good = good_particle_local;
            groups[new_grp].point = iz;
            groups[new_grp].bottom = iz;
            groups[new_grp].ll = new_grp;
            groups[new_grp].halo_app = new_grp;

            group_ID[iz] = new_grp;
            linking_list[iz] = iz;

            if (params.MinHaloMass == 1)
              {
                groups[new_grp].t_appear = frag[iz].Fmax;
#ifdef SNAPSHOT
                frag[iz].zacc = frag[iz].Fmax - 1;
#endif
              }
          }
        else if (neigrp == 1)
          {
            /* Single neighboring group: check accretion */
            double d2, r2;
            condition_for_accretion_ts(1, ibox, jbox, kbox, iz, frag[iz].Fmax,
                                       neigh[0], &d2, &r2, sigmaD_part,
                                       &obj1_local, &obj2_local);
            if (d2 < r2)
              {
                if (good_particle_local)
                  subdomains[my_sd].counters[7]++;
                accrflag = 1;
                to_group = neigh[0];
                accretion_ts(to_group, ibox, jbox, kbox, iz, frag[iz].Fmax,
                             &obj1_local, &obj2_local);
              }
            else
              {
                if (good_particle_local)
                  subdomains[my_sd].counters[12]++;
                my_filament_mass++;
                group_ID[iz] = FILAMENT;
                linking_list[iz] = iz;
              }
          }
        else if (neigrp > 1)
          {
            /* Multiple neighboring groups: check accretion to nearest, then merging */
            double d2, r2, ratio, best_ratio;
            int accgrp = -1;

            best_ratio = pow(10.0 * subbox.Lgwbl[_x_], 2.0);
            for (int ig1 = 0; ig1 < neigrp; ig1++)
              {
                condition_for_accretion_ts(2, ibox, jbox, kbox, iz, frag[iz].Fmax,
                                            neigh[ig1], &d2, &r2, sigmaD_part,
                                            &obj1_local, &obj2_local);
                ratio = d2 / r2;
                if (ratio < 1.0 && ratio < best_ratio)
                  {
                    best_ratio = ratio;
                    accgrp = ig1;
                  }
              }

            if (accgrp >= 0)
              {
                if (good_particle_local)
                  {
                    subdomains[my_sd].counters[7]++;
                    subdomains[my_sd].counters[8]++;
                  }
                accrflag = 1;
                to_group = neigh[accgrp];
                accretion_ts(neigh[accgrp], ibox, jbox, kbox, iz, frag[iz].Fmax,
                             &obj1_local, &obj2_local);
              }

            /* Check merging between group pairs */
            int merge[NV][NV];
            int nmerge = 0;
            for (int ig1 = 0; ig1 < neigrp; ig1++)
              for (int ig2 = 0; ig2 < ig1; ig2++)
                {
                  merge[ig1][ig2] = 0;
                  int merge_flag;
                  condition_for_merging_ts(frag[iz].Fmax, neigh[ig1], neigh[ig2],
                                            &merge_flag, sigmaD_part, sigmaD_part,
                                            &obj1_local, &obj2_local);
                  if (merge_flag)
                    {
                      merge[ig1][ig2] = 1;
                      nmerge++;
                    }
                }

            if (nmerge > 0)
              {
                for (int ig1 = 0; ig1 < neigrp; ig1++)
                  for (int ig2 = 0; ig2 < ig1; ig2++)
                    if (merge[ig1][ig2] == 1 && neigh[ig1] != neigh[ig2])
                      {
                        if (good_particle_local)
                          subdomains[my_sd].counters[10]++;

                        int small_g, large_g;
                        if (groups[neigh[ig1]].Mass > groups[neigh[ig2]].Mass)
                          {
                            merge_groups_ts(neigh[ig1], neigh[ig2], frag[iz].Fmax,
                                            &obj1_local, &obj2_local);
                            large_g = neigh[ig1];
                            small_g = neigh[ig2];
                          }
                        else
                          {
                            merge_groups_ts(neigh[ig2], neigh[ig1], frag[iz].Fmax,
                                            &obj1_local, &obj2_local);
                            small_g = neigh[ig1];
                            large_g = neigh[ig2];
                          }
                        if (to_group == small_g)
                          to_group = large_g;
                        for (int ig3 = 0; ig3 < neigrp; ig3++)
                          if (neigh[ig3] == small_g)
                            neigh[ig3] = large_g;

                        if (groups[large_g].Mass < 5 * groups[small_g].Mass && good_particle_local)
                          subdomains[my_sd].counters[11]++;
                      }
              }

            /* If not yet accreted, try again after merging */
            if (accgrp == -1)
              {
                clean_list(neigh);
                neigrp = 0;
                for (int nn = 0; nn < NV; nn++)
                  if (neigh[nn] > FILAMENT) neigrp++;

                best_ratio = pow(10.0 * subbox.Lgwbl[_x_], 2.0);
                accgrp = -1;
                for (int ig1 = 0; ig1 < neigrp; ig1++)
                  {
                    condition_for_accretion_ts(3, ibox, jbox, kbox, iz, frag[iz].Fmax,
                                                neigh[ig1], &d2, &r2, sigmaD_part,
                                                &obj1_local, &obj2_local);
                    ratio = d2 / r2;
                    if (ratio < best_ratio)
                      {
                        best_ratio = ratio;
                        accgrp = ig1;
                      }
                  }

                if (best_ratio < 1.0)
                  {
                    if (good_particle_local)
                      {
                        subdomains[my_sd].counters[7]++;
                        subdomains[my_sd].counters[9]++;
                      }
                    accrflag = 1;
                    to_group = neigh[accgrp];
                    accretion_ts(neigh[accgrp], ibox, jbox, kbox, iz, frag[iz].Fmax,
                                 &obj1_local, &obj2_local);
                  }
                else
                  {
                    if (good_particle_local)
                      subdomains[my_sd].counters[12]++;
                    my_filament_mass++;
                    group_ID[iz] = FILAMENT;
                    linking_list[iz] = iz;
                  }
              }
          }
        else
          {
            /* No neighboring groups: filament */
            if (good_particle_local)
              subdomains[my_sd].counters[12]++;
            my_filament_mass++;
            group_ID[iz] = FILAMENT;
            linking_list[iz] = iz;
          }

        /* Filament re-accretion: only for filament particles in THIS subdomain.
           Cross-subdomain filament particles are left for Phase 2 to handle.
           Without this check, two threads could accrete the same filament particle
           into different groups, corrupting both linking lists. */
        if (accrflag && nf)
          {
            for (int ifil = 0; ifil < nf; ifil++)
              {
                /* Skip filament particles outside this thread's subdomains */
                int fil_sd = subcube_of_coord(fil_list[ifil][0], fil_list[ifil][1], fil_list[ifil][2]);
                if (fil_sd < sd_start || fil_sd >= sd_end)
                  continue;

                double d2f, r2f;
                condition_for_accretion_ts(4, fil_list[ifil][0], fil_list[ifil][1],
                                            fil_list[ifil][2], fil_list[ifil][3],
                                            frag[iz].Fmax, to_group,
                                            &d2f, &r2f, sigmaD_part,
                                            &obj1_local, &obj2_local);
                if (d2f < r2f)
                  fil_list[ifil][3] *= -1;
              }
            for (int ifil = 0; ifil < nf; ifil++)
              if (fil_list[ifil][3] < 0)
                {
                  fil_list[ifil][3] *= -1;
                  accretion_ts(to_group, fil_list[ifil][0], fil_list[ifil][1],
                              fil_list[ifil][2], fil_list[ifil][3], frag[iz].Fmax,
                              &obj1_local, &obj2_local);
                  my_filament_mass--;
                }
          }
          } /* end for pidx */
      } /* end for my_sd */

    thread_nboundary[tid] = my_nboundary;

    /* Accumulate thread-local filament mass into global (single atomic at end) */
    #pragma omp atomic
    groups[FILAMENT].Mass += my_filament_mass;

    /* Per-thread diagnostic output */
    #pragma omp critical
    {
      int my_ngroups = 0;
      for (int sd = sd_start; sd < sd_end; sd++)
        my_ngroups += subdomains[sd].ngroups_local;

      printf("[PARALLEL] Task %d Thread %d: subdomains [%d,%d), peaks=%d, boundary=%d, "
             "groups_created=%d\n",
             ThisTask, tid, sd_start, sd_end, my_peaks, my_boundary, my_ngroups);
      fflush(stdout);
    }
  } /* end omp parallel */

  t_phase1 = MPI_Wtime() - t1;

  /* Free pre-partition arrays */
  free(sd_count);
  free(sd_offset);
  free(sd_particles);

  /* ================================================================
     PHASE 2: Boundary reconciliation (sequential)
     Process boundary particles that had cross-subdomain neighbors.
     These are handled using the standard serial condition functions
     since they need to see the final state of all subdomains.
     ================================================================ */

  int total_boundary = 0;
  for (int t = 0; t < nthreads; t++)
    total_boundary += thread_nboundary[t];

  if (!ThisTask)
    printf("[PARALLEL] Phase 2: reconciling %d boundary particles out of %d total\n",
           total_boundary, nstep);

  /* Update global ngroups to reflect all subdomain group creation */
  ngroups = FILAMENT;
  {
    int total_groups_created = 0;
    for (int sd = 0; sd < n_subdomains; sd++)
      {
        int top = subdomains[sd].group_base + subdomains[sd].ngroups_local - 1;
        if (top > ngroups)
          ngroups = top;
        total_groups_created += subdomains[sd].ngroups_local;
      }
    if (!ThisTask)
      printf("[PARALLEL] Phase 1 created %d groups total (ngroups=%d)\n",
             total_groups_created, ngroups);
  }

  /* DIAGNOSTIC: Validate ALL linking lists before Phase 2.
     Enable with -DVALIDATE_LINKING_LISTS at compile time. */
#ifdef VALIDATE_LINKING_LISTS
  {
    double tv0 = MPI_Wtime();
    if (!ThisTask)
      {
        int n_corrupted = 0, n_checked = 0;
        for (int sd = 0; sd < n_subdomains; sd++)
          for (int g = subdomains[sd].group_base;
               g < subdomains[sd].group_base + subdomains[sd].ngroups_local; g++)
            {
              if (groups[g].point < 0 || groups[g].Mass <= 0) continue;
              n_checked++;
              int p = groups[g].point, count = 0, max = groups[g].Mass + 10;
              while (linking_list[p] != groups[g].point && ++count < max)
                p = linking_list[p];
              if (count >= max)
                { n_corrupted++;
                  if (n_corrupted <= 5)
                    printf("[VALIDATE] CORRUPTED: grp=%d sd=%d point=%d Mass=%d\n",
                           g, sd, groups[g].point, groups[g].Mass); }
            }
        printf("[VALIDATE] Checked %d groups, %d corrupted (%.4fs)\n",
               n_checked, n_corrupted, MPI_Wtime() - tv0);
      }
  }
#endif

  double t2 = MPI_Wtime();

  int phase2_done = 0;
  for (int t = 0; t < nthreads; t++)
    {
      for (int b = 0; b < thread_nboundary[t]; b++)
        {
          int iz = thread_boundary[t][b].particle_idx;
          int ibox, jbox, kbox;

          if (!ThisTask && (phase2_done % 100 == 0))
            {
              printf("[PARALLEL] Phase 2 progress: %d/%d\n", phase2_done, total_boundary);
              fflush(stdout);
            }
          phase2_done++;

          /* Skip if this particle was already accreted by a previous
             boundary particle's filament re-accretion */
          if (group_ID[iz] != FILAMENT)
            continue;

          INDEX_TO_COORD(frag_pos[iz], ibox, jbox, kbox, subbox.Lgwbl);

          /* Re-run the full neighbor check using serial condition functions */
          int neigh_b[NV] = {0};
          int fil_list_b[NV][4];
          int nf_b = 0;
          int peak_cond_b = 1;
          int i1, j1, k1, pos;

          good_particle =
            (ibox >= subbox.safe[_x_] && ibox < subbox.Lgwbl[_x_] - subbox.safe[_x_] &&
             jbox >= subbox.safe[_y_] && jbox < subbox.Lgwbl[_y_] - subbox.safe[_y_] &&
             kbox >= subbox.safe[_z_] && kbox < subbox.Lgwbl[_z_] - subbox.safe[_z_]);

          particle_name =
            COORD_TO_INDEX((long long)((ibox + subbox.stabl[_x_] + MyGrids[0].GSglobal[_x_]) % MyGrids[0].GSglobal[_x_]),
                           (long long)((jbox + subbox.stabl[_y_] + MyGrids[0].GSglobal[_y_]) % MyGrids[0].GSglobal[_y_]),
                           (long long)((kbox + subbox.stabl[_z_] + MyGrids[0].GSglobal[_z_]) % MyGrids[0].GSglobal[_z_]),
                           MyGrids[0].GSglobal);

          for (int nn = 0; nn < NV; nn++)
            {
              switch (nn)
                {
                case 0:
                  i1 = (subbox.pbc[_x_] && ibox == 0 ? subbox.Lgwbl[_x_] - 1 : ibox - 1);
                  j1 = jbox; k1 = kbox; break;
                case 1:
                  i1 = (subbox.pbc[_x_] && ibox == subbox.Lgwbl[_x_] - 1 ? 0 : ibox + 1);
                  j1 = jbox; k1 = kbox; break;
                case 2:
                  i1 = ibox;
                  j1 = (subbox.pbc[_y_] && jbox == 0 ? subbox.Lgwbl[_y_] - 1 : jbox - 1);
                  k1 = kbox; break;
                case 3:
                  i1 = ibox;
                  j1 = (subbox.pbc[_y_] && jbox == subbox.Lgwbl[_y_] - 1 ? 0 : jbox + 1);
                  k1 = kbox; break;
                case 4:
                  i1 = ibox; j1 = jbox;
                  k1 = (subbox.pbc[_z_] && kbox == 0 ? subbox.Lgwbl[_z_] - 1 : kbox - 1);
                  break;
                case 5:
                  i1 = ibox; j1 = jbox;
                  k1 = (subbox.pbc[_z_] && kbox == subbox.Lgwbl[_z_] - 1 ? 0 : kbox + 1);
                  break;
                }

              pos = find_location(i1, j1, k1);
              if (pos >= 0)
                {
                  neigh_b[nn] = group_ID[pos];
                  peak_cond_b &= (frag[iz].Fmax > frag[pos].Fmax);
                }
              else
                neigh_b[nn] = 0;

              if (neigh_b[nn] == FILAMENT)
                {
                  neigh_b[nn] = 0;
                  fil_list_b[nf_b][0] = i1;
                  fil_list_b[nf_b][1] = j1;
                  fil_list_b[nf_b][2] = k1;
                  fil_list_b[nf_b][3] = pos;
                  nf_b++;
                }
            }

          clean_list(neigh_b);

          /* Filter out invalid group references: groups in the "gap"
             between subdomain ranges, or groups that were merged away.
             These have Mass=0 or point<0. */
          for (int nn = 0; nn < NV; nn++)
            if (neigh_b[nn] > FILAMENT &&
                (groups[neigh_b[nn]].Mass <= 0 || groups[neigh_b[nn]].point < 0))
              neigh_b[nn] = 0;

          int neigrp_b = 0;
          for (int nn = 0; nn < NV; nn++)
            if (neigh_b[nn] > FILAMENT) neigrp_b++;

          int accrflag_b = 0;
          int to_group_b = 0;

          /* Process using standard serial logic (uses global obj1, obj2 - safe here) */
          if (peak_cond_b && neigrp_b == 0)
            {
              /* Late peak: create group sequentially */
              ngroups++;
              if (ngroups <= Npeaks + 2)
                {
                  groups[ngroups].t_peak = frag[iz].Fmax;
                  groups[ngroups].t_appear = -1;
                  groups[ngroups].t_merge = -1;
                  groups[ngroups].Pos[0] = ibox + SHIFT;
                  groups[ngroups].Pos[1] = jbox + SHIFT;
                  groups[ngroups].Pos[2] = kbox + SHIFT;
                  groups[ngroups].Vel[0] = frag[iz].Vel[0];
                  groups[ngroups].Vel[1] = frag[iz].Vel[1];
                  groups[ngroups].Vel[2] = frag[iz].Vel[2];
#ifdef TWO_LPT
                  groups[ngroups].Vel_2LPT[0] = frag[iz].Vel_2LPT[0];
                  groups[ngroups].Vel_2LPT[1] = frag[iz].Vel_2LPT[1];
                  groups[ngroups].Vel_2LPT[2] = frag[iz].Vel_2LPT[2];
#ifdef THREE_LPT
                  groups[ngroups].Vel_3LPT_1[0] = frag[iz].Vel_3LPT_1[0];
                  groups[ngroups].Vel_3LPT_1[1] = frag[iz].Vel_3LPT_1[1];
                  groups[ngroups].Vel_3LPT_1[2] = frag[iz].Vel_3LPT_1[2];
                  groups[ngroups].Vel_3LPT_2[0] = frag[iz].Vel_3LPT_2[0];
                  groups[ngroups].Vel_3LPT_2[1] = frag[iz].Vel_3LPT_2[1];
                  groups[ngroups].Vel_3LPT_2[2] = frag[iz].Vel_3LPT_2[2];
#endif
#endif
#ifdef RECOMPUTE_DISPLACEMENTS
                  groups[ngroups].Vel_prev[0] = frag[iz].Vel_prev[0];
                  groups[ngroups].Vel_prev[1] = frag[iz].Vel_prev[1];
                  groups[ngroups].Vel_prev[2] = frag[iz].Vel_prev[2];
#ifdef TWO_LPT
                  groups[ngroups].Vel_2LPT_prev[0] = frag[iz].Vel_2LPT_prev[0];
                  groups[ngroups].Vel_2LPT_prev[1] = frag[iz].Vel_2LPT_prev[1];
                  groups[ngroups].Vel_2LPT_prev[2] = frag[iz].Vel_2LPT_prev[2];
#ifdef THREE_LPT
                  groups[ngroups].Vel_3LPT_1_prev[0] = frag[iz].Vel_3LPT_1_prev[0];
                  groups[ngroups].Vel_3LPT_1_prev[1] = frag[iz].Vel_3LPT_1_prev[1];
                  groups[ngroups].Vel_3LPT_1_prev[2] = frag[iz].Vel_3LPT_1_prev[2];
                  groups[ngroups].Vel_3LPT_2_prev[0] = frag[iz].Vel_3LPT_2_prev[0];
                  groups[ngroups].Vel_3LPT_2_prev[1] = frag[iz].Vel_3LPT_2_prev[1];
                  groups[ngroups].Vel_3LPT_2_prev[2] = frag[iz].Vel_3LPT_2_prev[2];
#endif
#endif
#endif
                  groups[ngroups].Mass = 1;
                  groups[ngroups].name = particle_name;
                  groups[ngroups].good = good_particle;
                  groups[ngroups].point = iz;
                  groups[ngroups].bottom = iz;
                  groups[ngroups].ll = ngroups;
                  groups[ngroups].halo_app = ngroups;
                  group_ID[iz] = ngroups;
                  linking_list[iz] = iz;
                  if (params.MinHaloMass == 1)
                    {
                      groups[ngroups].t_appear = frag[iz].Fmax;
#ifdef SNAPSHOT
                      frag[iz].zacc = frag[iz].Fmax - 1;
#endif
                    }
                }
            }
          else if (neigrp_b == 1)
            {
              double d2, r2;
              condition_for_accretion(1, ibox, jbox, kbox, iz, frag[iz].Fmax,
                                      neigh_b[0], &d2, &r2);
              if (d2 < r2)
                {
                  accrflag_b = 1;
                  to_group_b = neigh_b[0];
                  accretion(to_group_b, ibox, jbox, kbox, iz, frag[iz].Fmax);
                }
              else
                {
                  groups[FILAMENT].Mass++;
                  group_ID[iz] = FILAMENT;
                  linking_list[iz] = iz;
                }
            }
          else if (neigrp_b > 1)
            {
              double d2, r2, ratio, best_ratio;
              int accgrp = -1;

              best_ratio = pow(10.0 * subbox.Lgwbl[_x_], 2.0);
              for (int ig1 = 0; ig1 < neigrp_b; ig1++)
                {
                  condition_for_accretion(2, ibox, jbox, kbox, iz, frag[iz].Fmax,
                                          neigh_b[ig1], &d2, &r2);
                  ratio = d2 / r2;
                  if (ratio < 1.0 && ratio < best_ratio)
                    {
                      best_ratio = ratio;
                      accgrp = ig1;
                    }
                }

              if (accgrp >= 0)
                {
                  accrflag_b = 1;
                  to_group_b = neigh_b[accgrp];
                  accretion(neigh_b[accgrp], ibox, jbox, kbox, iz, frag[iz].Fmax);
                }

              /* Check merging between group pairs */
              int merge_b[NV][NV];
              int nmerge_b = 0, merge_flag_b;
              for (int ig1 = 0; ig1 < neigrp_b; ig1++)
                for (int ig2 = 0; ig2 < ig1; ig2++)
                  {
                    merge_b[ig1][ig2] = 0;
                    condition_for_merging(frag[iz].Fmax, neigh_b[ig1], neigh_b[ig2], &merge_flag_b);
                    if (merge_flag_b)
                      {
                        merge_b[ig1][ig2] = 1;
                        nmerge_b++;
                      }
                  }

              if (nmerge_b > 0)
                {
                  for (int ig1 = 0; ig1 < neigrp_b; ig1++)
                    for (int ig2 = 0; ig2 < ig1; ig2++)
                      if (merge_b[ig1][ig2] == 1 && neigh_b[ig1] != neigh_b[ig2])
                        {
                          int small_g, large_g;
                          if (groups[neigh_b[ig1]].Mass > groups[neigh_b[ig2]].Mass)
                            {
                              merge_groups(neigh_b[ig1], neigh_b[ig2], frag[iz].Fmax);
                              large_g = neigh_b[ig1];
                              small_g = neigh_b[ig2];
                            }
                          else
                            {
                              merge_groups(neigh_b[ig2], neigh_b[ig1], frag[iz].Fmax);
                              small_g = neigh_b[ig1];
                              large_g = neigh_b[ig2];
                            }
                          if (to_group_b == small_g)
                            to_group_b = large_g;
                          for (int ig3 = 0; ig3 < neigrp_b; ig3++)
                            if (neigh_b[ig3] == small_g)
                              neigh_b[ig3] = large_g;
                        }
                }

              if (accgrp == -1)
                {
                  clean_list(neigh_b);
                  neigrp_b = 0;
                  for (int nn = 0; nn < NV; nn++)
                    if (neigh_b[nn] > FILAMENT) neigrp_b++;

                  best_ratio = pow(10.0 * subbox.Lgwbl[_x_], 2.0);
                  accgrp = -1;
                  for (int ig1 = 0; ig1 < neigrp_b; ig1++)
                    {
                      condition_for_accretion(3, ibox, jbox, kbox, iz, frag[iz].Fmax,
                                              neigh_b[ig1], &d2, &r2);
                      ratio = d2 / r2;
                      if (ratio < best_ratio)
                        {
                          best_ratio = ratio;
                          accgrp = ig1;
                        }
                    }

                  if (best_ratio < 1.0)
                    {
                      accrflag_b = 1;
                      to_group_b = neigh_b[accgrp];
                      accretion(neigh_b[accgrp], ibox, jbox, kbox, iz, frag[iz].Fmax);
                    }
                  else
                    {
                      groups[FILAMENT].Mass++;
                      group_ID[iz] = FILAMENT;
                      linking_list[iz] = iz;
                    }
                }
            }
          else
            {
              /* No neighbors: filament */
              groups[FILAMENT].Mass++;
              group_ID[iz] = FILAMENT;
              linking_list[iz] = iz;
            }

          /* Filament re-accretion for boundary particles */
          if (accrflag_b && nf_b)
            {
              for (int ifil = 0; ifil < nf_b; ifil++)
                {
                  double d2f, r2f;
                  condition_for_accretion(4, fil_list_b[ifil][0], fil_list_b[ifil][1],
                                          fil_list_b[ifil][2], fil_list_b[ifil][3],
                                          frag[iz].Fmax, to_group_b, &d2f, &r2f);
                  if (d2f < r2f)
                    fil_list_b[ifil][3] *= -1;
                }
              for (int ifil = 0; ifil < nf_b; ifil++)
                if (fil_list_b[ifil][3] < 0)
                  {
                    fil_list_b[ifil][3] *= -1;
                    accretion(to_group_b, fil_list_b[ifil][0], fil_list_b[ifil][1],
                              fil_list_b[ifil][2], fil_list_b[ifil][3], frag[iz].Fmax);
                    groups[FILAMENT].Mass--;
                  }
            }
        }
    }

  t_phase2 = MPI_Wtime() - t2;

  if (!ThisTask)
    printf("[TIMING] partition=%.4fs  Phase1=%.4fs  Phase2=%.4fs  total=%.4fs  boundary=%d/%d (%.1f%%)\n",
           t_partition, t_phase1, t_phase2, MPI_Wtime() - t_start,
           total_boundary, nstep, 100.0 * total_boundary / nstep);

  /* Aggregate counters from all subdomains into the output array */
  for (int sd = 0; sd < n_subdomains; sd++)
    for (int c = 0; c < NCOUNTERS; c++)
      out_counters[c] += subdomains[sd].counters[c];

  /* Clean up */
  for (int t = 0; t < nthreads; t++)
    free(thread_boundary[t]);
  free(thread_boundary);
  free(thread_nboundary);

  if (subdomain_map)
    {
      free(subdomain_map);
      subdomain_map = NULL;
    }
  if (subdomains)
    {
      free(subdomains);
      subdomains = NULL;
    }

  return 0;
}

#endif /* _OPENMP */

#ifdef PLC
const gsl_root_fsolver_type *brent;
gsl_root_fsolver *solver;
gsl_function cPLC;
int thisgroup;
int replicate[3];
double brent_err;
#define SAFEPLC 3

double condition_PLC(PRODFLOAT);
double condition_F(double, void *);
int store_PLC(PRODFLOAT);
double find_brent(double, double);
#endif

int build_groups(int Npeaks, double zstop, int first_call)
{

  /* The algorithm for group construction runs as follows:
     loop on all collapsed particle
     + for each particle check its neighbours
     + if it is a peak of Fmax, make it a one-particle halo
     + if it touches only one halo:
       -> check if it should be accreted
       -> if it should, accrete it
       -> otherwise, tag it as filament
     + if it touches more than one halo:
       -> check if it should be accreted to one halo
       -> if it should, choose the one it gets nearest to
       -> then check if all the halo pairs should merge
       -> if they should, merge them
       -> if the particle was not accreted before, re-check it
       -> otherwise, tag it as filament
     + if it touches only filaments, tag it as filament
   */


  int merge[NV][NV], neigh[NV], fil_list[NV][4];
  static int iout,nstep,nstep_p;
  int nn,ifil,this_z,neigrp,nf,pos;
  int iz,i1,j1,k1,skip;
  int ig3,small,large,to_group,accgrp,ig1,ig2;
  int accrflag,nmerge,peak_cond;
  double ratio,best_ratio,d2,r2,cputmp;
  int merge_flag;
  int ibox,jbox,kbox;

  static int last_z=0;

  /*
    counters of various events:
    0 number of peaks
    i number of particles with i neighbours (i<=6)
    7 number of accretion events
    8 number of accretion events before checking a merger
    9 number of accretion events after checking a merger
    10 number of merging events
    11 number of major mergers
    12 number of filament particles
    13 number of accreted filament particles
    14 number of good halos
  */

  static unsigned long long counters[NCOUNTERS],all_counters[NCOUNTERS];

#ifdef PLC
  static int plc_started=0, last_check_done=0;
  int irep, save, mysave, storex, storey, storez;
  static double NextF_PLC, DeltaF_PLC;
  double aa, bb, Fplc;
#endif

  if (first_call)
    {
      /* this part contains initializations that should be done at
         first call */
      ngroups=FILAMENT;           /* group list starts from FILAMENTS */
      /* filaments are not grouped */
      for (i1=0; i1<=FILAMENT; i1++)
        {
          groups[i1].point=-1;
          groups[i1].bottom=-1;
          groups[i1].good=0;
        }
      iout=0;

      /* sets the counter to zero */
      for (i1=0; i1<NCOUNTERS; i1++)
        {
          counters[i1]=0; 
          all_counters[i1]=0;
        }


      if (!ThisTask)
        printf("[%s] Starting the fragmentation process to redshift %7.4f\n",fdate(),zstop);

      /* nstep is the number of collapsed particles that will be
         checked by the code. In classic fragmentation this is the
         number of particles that collapse by the end of the run. In
         default fragmentation this selection is performed at
         distribution time, so this is the number of stored particles.
      */
#ifdef CLASSIC_FRAGMENTATION
      nstep=0;
      while (frag[indices[nstep]].Fmax >= outputs.Flast)
        nstep++;
#else
      nstep=subbox.Nstored;
#endif
      nstep_p=nstep/20;

#ifdef PLC
      /* initialization of the root finding routine used by the PLC code */
      cPLC.function = &condition_F;
      brent  = gsl_root_fsolver_brent;
      solver = gsl_root_fsolver_alloc (brent);
      DeltaF_PLC  = 0.9; // CAPIRE COME FISSARLO
      NextF_PLC   = plc.Fstart * DeltaF_PLC;
      brent_err   = 1.e-2 * params.InterPartDist;
      plc.Nstored = plc.Nstored_last = 0;
#endif

      first_call=0;
    }
  else
    {
      if (!ThisTask)
        printf("[%s] Restarting the fragmentation process to redshift %7.4f\n",fdate(),zstop);
    }

  /************************************************************************
                    START OF THE CYCLE ON COLLAPSED PARTICLES
   ************************************************************************/

#ifdef _OPENMP
  /* Use the parallel path when OpenMP is available, we have multiple threads,
     the GFLUT tables are initialized, and PLC is not active (PLC uses
     global mutable state that is not thread-safe).
     The parallel path handles group construction within subdomains and
     then reconciles boundary interactions sequentially. */
  {
    int use_parallel = 0;  /* Serial path: parallel subdomain approach gives non-reproducible results */
#ifdef PLC
    use_parallel = 0;  /* PLC reconstruction requires the serial path */
#endif
#ifdef CLASSIC_FRAGMENTATION
    use_parallel = 0;  /* Classic fragmentation uses different indexing */
#endif
    if (use_parallel && first_call == 0 && last_z == 0)
      {
        if (!ThisTask)
          printf("[%s] Using OpenMP parallel path with %d threads\n",
                 fdate(), omp_get_max_threads());

        if (build_groups_parallel(Npeaks, nstep, counters))
          return 1;

        /* After parallel path, write all pending output catalogs.
           The serial path writes outputs inside the main loop when
           Fmax crosses output thresholds; here we write them all at once.
           NOTE: Catalogs will reflect the final group state, not intermediate
           states at each output redshift. For multiple output redshifts,
           the serial path should be used for time-accurate catalogs. */
        {
          double cputmp_par;
          while (iout < outputs.n)
            {
              cputmp_par = MPI_Wtime();

              if (!ThisTask)
                printf("[%s] Writing output at z=%f (parallel path)\n", fdate(),
                       outputs.z[iout]);

              fflush(stdout);
              MPI_Barrier(MPI_COMM_WORLD);

              if (write_catalog(iout))
                return 1;
              if (compute_mf(iout))
                return 1;
              if (iout == outputs.n - 1)
                {
                  if (write_histories())
                    return 1;
                }

              cputime.io += MPI_Wtime() - cputmp_par;
              iout++;
            }
        }

        last_z = nstep;
        goto build_groups_statistics;
      }
  }
#endif

  for (this_z=last_z; this_z<nstep; this_z++)
    {
      /* In classic fragmentation the particles must be addressed in
         order of collapse time, while in default fragmentation they
         are already in that order */
#ifdef CLASSIC_FRAGMENTATION
      iz=indices[this_z];
#else
      iz=this_z;
#endif


#ifdef PLC
      /* here it checks if it is time to write the stored PLC halos.
         If it is the case, it waits for all the tasks to get to the
         same point and writes the catalog up to this point
       */
      if (plc_started && frag[iz].Fmax < NextF_PLC && frag[iz].Fmax >= plc.Fstop)
        {
          if (!ThisTask)
            printf("[%s] Syncing tasks for PLC...  Nmax=%d, Nstored=%d, Nstored_last=%d, Fmax=%f\n",fdate(),plc.Nmax, plc.Nstored, plc.Nstored_last,frag[iz].Fmax);

          /* each task checks if the plc buffer is full and it is
             necessary to write the plc catalog. The criterion is that
             you need at least space for SAFEPLC times the number of
             halos that have been updated after last check */
          mysave = (plc.Nmax - plc.Nstored < SAFEPLC * (plc.Nstored - plc.Nstored_last));

          MPI_Reduce(&mysave, &save, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Bcast(&save, 1, MPI_INT, 0, MPI_COMM_WORLD);

          if (!ThisTask)
            printf("[%s] %d tasks require to store PLC halos at F=%6.3F...\n",fdate(),save,frag[iz].Fmax);

          if (save)
            {
              /* Save PLC halos if at least one task asks to */
              if (write_PLC(0))
                return 1;
              plc.Nstored=0;
            }

          /* next update will be at this time */
          NextF_PLC *= DeltaF_PLC;
          plc.Nstored_last = plc.Nstored;
            
        }
#endif

      /* More initializations */

      neigrp=0;               /* number of neighbouring groups */
      nf=0;                   /* number of neighbouring filament points */
      accrflag=0;             /* if =1 all the neighbouring filaments are accreted */
      for (i1=0; i1<NV; i1++)
        neigh[i1]=0;          /* number of neighbours */

      /*******************************************/
      /* PARTICLE COORDINATES AND PEAK CONDITION */
      /*******************************************/
#ifdef CLASSIC_FRAGMENTATION
      INDEX_TO_COORD(iz,ibox,jbox,kbox,subbox.Lgwbl);
#else
      INDEX_TO_COORD(frag_pos[iz],ibox,jbox,kbox,subbox.Lgwbl);
#endif

      /* skips the peak condition if the point is at the border (and PBCs are not active) */
      skip=0;
      if ( !subbox.pbc[_x_] && (ibox==0 || ibox==subbox.Lgwbl[_x_]-1) ) ++skip;
      if ( !subbox.pbc[_y_] && (jbox==0 || jbox==subbox.Lgwbl[_y_]-1) ) ++skip;
      if ( !subbox.pbc[_z_] && (kbox==0 || kbox==subbox.Lgwbl[_z_]-1) ) ++skip;

      particle_name = 
        COORD_TO_INDEX((long long)((ibox + subbox.stabl[_x_] + MyGrids[0].GSglobal[_x_])%MyGrids[0].GSglobal[_x_]),
                       (long long)((jbox + subbox.stabl[_y_] + MyGrids[0].GSglobal[_y_])%MyGrids[0].GSglobal[_y_]),
                       (long long)((kbox + subbox.stabl[_z_] + MyGrids[0].GSglobal[_z_])%MyGrids[0].GSglobal[_z_]),
                       MyGrids[0].GSglobal);

      good_particle = ( ibox>=subbox.safe[_x_] && ibox<subbox.Lgwbl[_x_]-subbox.safe[_x_] && 
                        jbox>=subbox.safe[_y_] && jbox<subbox.Lgwbl[_y_]-subbox.safe[_y_] && 
                        kbox>=subbox.safe[_z_] && kbox<subbox.Lgwbl[_z_]-subbox.safe[_z_] );

      if (!skip)
        {
          peak_cond=1;   
          /* checks whether the neighbouring particles collapse later */

          for (nn=0; nn<NV; nn++)
            {
              /* coordinates of the neighbouring particle */
              switch (nn)
                {
                case 0:
                  i1=( subbox.pbc[_x_] && ibox==0 ? subbox.Lgwbl[_x_]-1 : ibox-1 );
                  j1=jbox;
                  k1=kbox;
                  break;
                case 1:
                  i1=( subbox.pbc[_x_] && ibox==subbox.Lgwbl[_x_]-1 ? 0 : ibox+1 );
                  j1=jbox;
                  k1=kbox;
                  break;
                case 2:
                  i1=ibox;
                  j1=( subbox.pbc[_y_] && jbox==0 ? subbox.Lgwbl[_y_]-1 : jbox-1 );
                  k1=kbox;
                  break;
                case 3:
                  i1=ibox;
                  j1=( subbox.pbc[_y_] && jbox==subbox.Lgwbl[_y_]-1 ? 0 : jbox+1 );
                  k1=kbox;
                  break;
                case 4:
                  i1=ibox;
                  j1=jbox;
                  k1=( subbox.pbc[_z_] && kbox==0 ? subbox.Lgwbl[_z_]-1 : kbox-1 );
                  break;
                case 5:
                  i1=ibox;
                  j1=jbox;
                  k1=( subbox.pbc[_z_] && kbox==subbox.Lgwbl[_z_]-1 ? 0 : kbox+1 );
                  break;
                }

              /* accessing the neighbouring particle differs in
                 classic and default fragmentation: in classic
                 fragmentation the information on the neighbour is
                 immediately obtained, in standard fragmentation the
                 neighbour must be seeked in the particle list */
#ifdef CLASSIC_FRAGMENTATION
              pos = COORD_TO_INDEX(i1,j1,k1,subbox.Lgwbl);
              neigh[nn] = group_ID[pos];
              peak_cond &= (frag[iz].Fmax > frag[pos].Fmax);
#else
              pos = find_location(i1,j1,k1);
              if (pos>=0)
                {
                  neigh[nn] = group_ID[pos];
                  peak_cond &= (frag[iz].Fmax > frag[pos].Fmax);
                }
              else
                neigh[nn] = 0;
#endif

              /* neighbouring filaments are stored separately */
              if (neigh[nn]==FILAMENT)
                {
                  neigh[nn]=0;
                  fil_list[nf][0]=i1;
                  fil_list[nf][1]=j1;
                  fil_list[nf][2]=k1;
#ifdef CLASSIC_FRAGMENTATION
                  fil_list[nf][3]=pos;
#else
                  fil_list[nf][3]=pos;
#endif
                  nf++;
                }

            }

          /* Cleans the list of neighbouring groups removing duplicates */
          clean_list(neigh);

          /* Number of neighbouring groups */
          for (nn=neigrp=0; nn<NV; nn++)
            if (neigh[nn]>FILAMENT)
              neigrp++;

          if (neigrp>0 && good_particle) 
            counters[neigrp]++;

#ifdef PLC
          /* Past light cone on-the-fly reconstruction: */

          /* is it time to reconstruct the PLC? */
          if (frag[iz].Fmax<plc.Fstart && frag[iz].Fmax>=plc.Fstop)
            {

              /* check if this is the first call */
              if (!plc_started)
                {
                  plc_started=1;
                  if (!ThisTask)
                    printf("[%s] Starting PLC reconstruction, Task 0 will store at most %d halos\n",fdate(),plc.Nmax);
                  cputmp=MPI_Wtime();
                }

              /* PBCs are switched off for this check */
              storex=subbox.pbc[_x_];
              storey=subbox.pbc[_y_];
              storez=subbox.pbc[_z_];
              subbox.pbc[_x_]=subbox.pbc[_y_]=subbox.pbc[_z_]=0;

              /* the check is performed on all neighbouring groups */
              for (ig1=0; ig1<neigrp; ig1++)
                {
                  /* is the group good and massive enough? */
                  if (neigh[ig1] > FILAMENT &&
                      groups[neigh[ig1]].good && 
                      groups[neigh[ig1]].Mass >= params.MinHaloMass)
                    {
                      thisgroup=neigh[ig1];

                      /* loop on replications */
                      /* this loop may be threaded or ported to GPUs,
                         though it's relatively fast */
                      for (irep=0; irep<plc.Nreplications; irep++)
                        /* checks that the redshift falls in the
                           range of the replication */
                        if (!(frag[iz].Fmax > plc.repls[irep].F1 || 
                              groups[thisgroup].Flast < plc.repls[irep].F2))
                          {
                            replicate[0]=plc.repls[irep].i;
                            replicate[1]=plc.repls[irep].j;
                            replicate[2]=plc.repls[irep].k;
                            /* this computes the difference between
                               the group distance from the observer
                               and its comoving distance at that z */
                            bb=condition_PLC(frag[iz].Fmax);

                            /* in this unlikely case we catch the
                               group just on the PLC */
                            if (bb==0.0)
                              {
                                if (store_PLC(frag[iz].Fmax))
                                  return 1;
                              }
                            else if (bb>0.)
                              {
                                /* if it is outside the PLC then check
                                   whether it has just passed since
                                   last check */
                                aa=condition_PLC(groups[thisgroup].Flast);
                                if (aa<0.0)
                                  {
                                    /* in this case the group has
                                       passed through the PLC since
                                       last check, then compute its
                                       time of PLC cross */
                                    if ( (Fplc=find_brent(groups[thisgroup].Flast,frag[iz].Fmax)) == -99.00)
                                      return 1;
                                    /* and store it */
                                    if (store_PLC(Fplc))
                                      return 1;
                                  }
                              }
                          }
                    }
                  /* updates the Flast field */
                  groups[neigh[ig1]].Flast=frag[iz].Fmax;
              
                }
              /* restoring PBCs */
              subbox.pbc[_x_]=storex;
              subbox.pbc[_y_]=storey;
              subbox.pbc[_z_]=storez;

            }
          else if (plc.Fstart>0. && frag[iz].Fmax<plc.Fstart)
            {
              /* if nothing must be done, only update the Flast field */
              for (ig1=0; ig1<neigrp; ig1++)
                groups[neigh[ig1]].Flast=frag[iz].Fmax;
            }

#endif
        }
      else
        {
          /* this closes the if (!skip) condition above: if the
             particle is at the border of the domain it will not be a
             peak and will have no neighbours */
          peak_cond=0;
          neigrp=0;
        }

      /* Is the point a peak? */
      if (peak_cond)
        {
          /**********************************************************************
                                     FIRST CASE: PEAK
          **********************************************************************/

          /* New group */
          if (good_particle) 
            counters[0]++;

          ngroups++;

          /* this paranoid check will raise an error in case of major bugs */
          if (ngroups > Npeaks+2)
            {
              printf("OH MY DEAR, TASK %d FOUND TOO MANY GROUPS AT z=%f, STEP %d OF %d, THIS SHOULD NOT HAPPEN!\n",ThisTask,frag[iz].Fmax-1.0,this_z,nstep);
              return 1;
            }

          /* sets the properties of a one-particle group */
          groups[ngroups].t_peak=frag[iz].Fmax;
          groups[ngroups].t_appear=-1;
          groups[ngroups].t_merge=-1;
          groups[ngroups].Pos[0]=ibox+SHIFT;
          groups[ngroups].Pos[1]=jbox+SHIFT;
          groups[ngroups].Pos[2]=kbox+SHIFT;
          groups[ngroups].Vel[0]=frag[iz].Vel[0];
          groups[ngroups].Vel[1]=frag[iz].Vel[1];
          groups[ngroups].Vel[2]=frag[iz].Vel[2];
#ifdef TWO_LPT
          groups[ngroups].Vel_2LPT[0]=frag[iz].Vel_2LPT[0];
          groups[ngroups].Vel_2LPT[1]=frag[iz].Vel_2LPT[1];
          groups[ngroups].Vel_2LPT[2]=frag[iz].Vel_2LPT[2];
#ifdef THREE_LPT
          groups[ngroups].Vel_3LPT_1[0]=frag[iz].Vel_3LPT_1[0];
          groups[ngroups].Vel_3LPT_1[1]=frag[iz].Vel_3LPT_1[1];
          groups[ngroups].Vel_3LPT_1[2]=frag[iz].Vel_3LPT_1[2];
          groups[ngroups].Vel_3LPT_2[0]=frag[iz].Vel_3LPT_2[0];
          groups[ngroups].Vel_3LPT_2[1]=frag[iz].Vel_3LPT_2[1];
          groups[ngroups].Vel_3LPT_2[2]=frag[iz].Vel_3LPT_2[2];
#endif
#endif
#ifdef RECOMPUTE_DISPLACEMENTS
          groups[ngroups].Vel_prev[0]=frag[iz].Vel_prev[0];
          groups[ngroups].Vel_prev[1]=frag[iz].Vel_prev[1];
          groups[ngroups].Vel_prev[2]=frag[iz].Vel_prev[2];
#ifdef TWO_LPT
          groups[ngroups].Vel_2LPT_prev[0]=frag[iz].Vel_2LPT_prev[0];
          groups[ngroups].Vel_2LPT_prev[1]=frag[iz].Vel_2LPT_prev[1];
          groups[ngroups].Vel_2LPT_prev[2]=frag[iz].Vel_2LPT_prev[2];
#ifdef THREE_LPT
          groups[ngroups].Vel_3LPT_1_prev[0]=frag[iz].Vel_3LPT_1_prev[0];
          groups[ngroups].Vel_3LPT_1_prev[1]=frag[iz].Vel_3LPT_1_prev[1];
          groups[ngroups].Vel_3LPT_1_prev[2]=frag[iz].Vel_3LPT_1_prev[2];
          groups[ngroups].Vel_3LPT_2_prev[0]=frag[iz].Vel_3LPT_2_prev[0];
          groups[ngroups].Vel_3LPT_2_prev[1]=frag[iz].Vel_3LPT_2_prev[1];
          groups[ngroups].Vel_3LPT_2_prev[2]=frag[iz].Vel_3LPT_2_prev[2];
#endif
#endif
#endif
          groups[ngroups].Mass=1;
          groups[ngroups].name=particle_name;
          groups[ngroups].good = good_particle;
          groups[ngroups].point = iz;
          groups[ngroups].bottom = iz;
          groups[ngroups].ll=ngroups;
          groups[ngroups].halo_app=ngroups;
#ifdef PLC
          if (frag[iz].Fmax > plc.Fstart)
            groups[ngroups].Flast=plc.Fstart;
          else
            groups[ngroups].Flast=frag[iz].Fmax;
#endif

          group_ID[iz]=ngroups;
          linking_list[iz]=iz;
          if (params.MinHaloMass==1)
            {
              groups[ngroups].t_appear=frag[iz].Fmax;
#ifdef SNAPSHOT
              frag[iz].zacc=frag[iz].Fmax-1;
#endif
            }

        }
      else if (neigrp==1)
        {

          /**********************************************************************
                                    SECOND CASE: 1 GROUP
           **********************************************************************/

          /* if the points touches only one group, check whether to accrete the point on it */

          condition_for_accretion(1,ibox,jbox,kbox,iz,frag[iz].Fmax,neigh[0],&d2,&r2);

          if (d2<r2)
            {

              /*********************
                accretion on a group!
              *********************/

              if (good_particle) 
                counters[7]++;
              accrflag=1;
              to_group=neigh[0];
              accretion(to_group,ibox,jbox,kbox,iz,frag[iz].Fmax);
            }
          else
             {
               /*********
                filament!
                *********/

               if (good_particle) 
                 counters[12]++;
               groups[FILAMENT].Mass++;
               group_ID[iz]=FILAMENT;
               linking_list[iz]=iz;
             }

        }
      else if (neigrp>1)
        {
          /**********************************************************************
                                   THIRD CASE: >1 GROUP
           **********************************************************************/

          /* In this case the point touches more than one group */

          /*********************
           accretion on a group?
           *********************/

          best_ratio=pow(10.*subbox.Lgwbl[_x_],2.0);
          accgrp=-1;
          for (ig1=0; ig1<neigrp; ig1++)
            {
              condition_for_accretion(2,ibox,jbox,kbox,iz,frag[iz].Fmax,neigh[ig1],&d2,&r2);
              ratio=d2/r2;
              if (ratio<1.0 && ratio<best_ratio)
                {
                  best_ratio=ratio;
                  accgrp=ig1;
                }
            }

          if (accgrp>=0)
            {
              if (good_particle) 
                {
                  counters[7]++;
                  counters[8]++;
                }
              accrflag=1;
              to_group=neigh[accgrp];
              accretion(neigh[accgrp],ibox,jbox,kbox,iz,frag[iz].Fmax);
            }

          /* Then checks by pairs whether the groups must be merged together */

          nmerge=0;
          for (ig1=0; ig1<neigrp; ig1++)
            for (ig2=0; ig2<ig1; ig2++)
              {
                merge[ig1][ig2]=0;
                condition_for_merging(frag[iz].Fmax,neigh[ig1],neigh[ig2],&merge_flag);
                if (merge_flag)
                  {
                    merge[ig1][ig2]=1;
                    nmerge++;
                  }
              }

          /******************
           merging of groups!
           ******************/

          /* The group number of the largest group is preserved */

          if (nmerge>0)
            {
              for (ig1=0; ig1<neigrp; ig1++)
                for (ig2=0; ig2<ig1; ig2++)
                  if (merge[ig1][ig2]==1 && neigh[ig1]!=neigh[ig2])
                    {
                      if (good_particle) 
                        counters[10]++;
                      if (groups[neigh[ig1]].Mass > groups[neigh[ig2]].Mass)
                        {
                          merge_groups(neigh[ig1],neigh[ig2],frag[iz].Fmax);
                          large=neigh[ig1];
                          small=neigh[ig2];
                        }
                      else
                        {
                          merge_groups(neigh[ig2],neigh[ig1],frag[iz].Fmax);
                          small=neigh[ig1];
                          large=neigh[ig2];
                        }
                      if (to_group==small) 
                        to_group=large;
                      for (ig3=0; ig3<neigrp; ig3++)
                        if (neigh[ig3]==small) 
                          neigh[ig3]=large;

                      if (groups[large].Mass < 5*groups[small].Mass && good_particle)
                        counters[11]++;
                    }
            }

          /* If relevant, it tries again to accrete the particle */

          if (accgrp==-1)
            {
              clean_list(neigh);

              /* Number of neighbouring groups */
              for (nn=neigrp=0; nn<NV; nn++)
                if (neigh[nn]>FILAMENT)
                  neigrp++;

              best_ratio=pow(10.*subbox.Lgwbl[_x_],2.0);
              accgrp=-1;
              for (ig1=0; ig1<neigrp; ig1++)
                {
                  condition_for_accretion(3,ibox,jbox,kbox,iz,frag[iz].Fmax,neigh[ig1],&d2,&r2);
                  ratio=d2/r2;
                  if (ratio<best_ratio)
                    {
                      best_ratio=ratio;
                      accgrp=ig1;
                    }
                }

              if (best_ratio<1)
                {
                  if (good_particle) 
                    {
                      counters[7]++;
                      counters[9]++;
                    }
                  accrflag=1;
                  to_group=neigh[accgrp];
                  accretion(neigh[accgrp],ibox,jbox,kbox,iz,frag[iz].Fmax);
                }
              else
                {
                  /* If the point has not been accreted at all: */

                  /*********
                   filament!
                   *********/

                  if (good_particle)
                    counters[12]++;
                  groups[FILAMENT].Mass++;
                  group_ID[iz]=FILAMENT;
                  linking_list[iz]=iz;
                }
            }
        }
      else
        {
	  /**********************************************************************
                                   FOURTH CASE: FILAMENTS
	  **********************************************************************/
          if (good_particle)
            counters[12]++;
          groups[FILAMENT].Mass++;
          group_ID[iz]=FILAMENT;
          linking_list[iz]=iz;

	  /****************/
          /* end of cases */
	  /****************/

        }

      /* If the particle was accreted somewhere, checks whether to
         accrete to the same halo all the neighbouring filaments;
         first it checks conditions for all filament particles, then
         it accretes those that should */

      if (accrflag && nf && !skip)
        {
	  /* loop on filament particles */
          for (ifil=0; ifil<nf; ifil++)
            {
              condition_for_accretion(4,fil_list[ifil][0], fil_list[ifil][1],fil_list[ifil][2],
                                      fil_list[ifil][3],frag[iz].Fmax,to_group, &d2,&r2);
	      /* tags filaments that should be accreted (the condition
		 should be checked without changing the halo) */
              if (d2<r2)
                fil_list[ifil][3]*=-1;
            }
	  /* accrete all filament particles */
          for (ifil=0; ifil<nf; ifil++)
            if (fil_list[ifil][3]<0)
              {
                fil_list[ifil][3]*=-1;
                accretion(to_group, fil_list[ifil][0], fil_list[ifil][1],
                          fil_list[ifil][2],fil_list[ifil][3],frag[iz].Fmax);

                groups[FILAMENT].Mass--;

                if ( fil_list[ifil][0] >= subbox.safe[_x_] && 
                     fil_list[ifil][0] <  subbox.Lgwbl[_x_]-subbox.safe[_x_] && 
                     fil_list[ifil][1] >= subbox.safe[_y_] && 
                     fil_list[ifil][1] <  subbox.Lgwbl[_y_]-subbox.safe[_y_] && 
                     fil_list[ifil][2] >= subbox.safe[_z_] && 
                     fil_list[ifil][2] <  subbox.Lgwbl[_z_]-subbox.safe[_z_] )
                  {
                    counters[7]++;
                    counters[13]++;
                    counters[12]--;
                  }
              }
	}

#ifdef PLC
      /* if this is the end of the cycle for PLC, perform a last check
	 on all halos to see if some has passed through the PLC in
	 the meantime */
      /* Note: this part is largely a replication of the PLC code above */
      if (plc.Fstart>0 && !last_check_done &&
          (this_z==nstep-1 || frag[iz].Fmax<plc.Fstop))
        {

          /* first write to disc what you have */
          if (write_PLC(0))
            return 1;
          plc.Nstored=0;

	  /* PBCs are switched off for this check */
          storex=subbox.pbc[_x_];
          storey=subbox.pbc[_y_];
          storez=subbox.pbc[_z_];
          subbox.pbc[_x_]=subbox.pbc[_y_]=subbox.pbc[_z_]=0;

          last_check_done=1;
	  /* the check is performed on ALL groups */
          for (ig1=FILAMENT+1; ig1<=ngroups; ig1++)
            {
              /* is the group alive, good and massive enough? */
              if (groups[ig1].point >= 0 && groups[ig1].good &&
                  groups[ig1].Mass >= params.MinHaloMass)
                {
                  thisgroup=ig1;

                  /* loop on replications */
		  /* this loop may be threaded or ported to GPUs,
		     though it's relatively fast */
                  for (irep=0; irep<plc.Nreplications; irep++)
		    /* checks that the redshift falls in the
		       range of the replication */
                    if (groups[ig1].Flast > plc.repls[irep].F2)
                      {
                        replicate[0]=plc.repls[irep].i;
                        replicate[1]=plc.repls[irep].j;
                        replicate[2]=plc.repls[irep].k;
			/* this computes the difference between the
			   group distance from the observer and its
			   comoving distance at that z */
                        bb=condition_PLC(plc.Fstop);

                        /* in this unlikely case we catch the group
			   just on the PLC */
                        if (bb==0.0)
                          {
                            if (store_PLC(plc.Fstop))
                              return 1;
                          }
                        else if (bb>0.)
                          {
			    /* if it is outside the PLC then check
			       whether it has just passed since
			       last check */
                            aa=condition_PLC(groups[ig1].Flast);
                            if (aa<0.0)
                              {
                                /* in this case the group has passed
				   through the PLC since last time */
                                if ( (Fplc=find_brent(groups[ig1].Flast,plc.Fstop)) == -99.0)
                                  return 1;
                                if (store_PLC(Fplc))
                                  return 1;
                              }
                          }
                      }
                }
            }

          /* restoring PBCs */
          subbox.pbc[_x_]=storex;
          subbox.pbc[_y_]=storey;
          subbox.pbc[_z_]=storez;

          if (!ThisTask)
            printf("[%s] PLC: Last check on groups done, Task 0 stored %d halos (max:%d)\n",
                   fdate(),plc.Nstored,plc.Nmax);
          cputime.plc += MPI_Wtime()-cputmp;

          if (write_PLC(1))
            return 1;
        }
#endif


      /************************************************************************
                                  CLOSING THE CYCLE
       ************************************************************************/

      /* prints the fraction of steps done */
      if (!ThisTask && !(this_z%nstep_p))
        printf("[%s] *** %3d%% done, F = %6.2f,  z = %6.2f\n",fdate(),
               this_z/(nstep_p)*5,frag[iz].Fmax,
               frag[iz].Fmax-1.0);

      /* Write output if relevant. 
         The while cycle is here because it may happen that some
         outputs are requested when there are very few collapsed
         particles, and if no particle collapses between two catalogs
         one catalog may not be written; better to have an empty
         catalog, or two identical ones */
      while (this_z==nstep-1 || frag[iz].Fmax < outputs.F[iout])
        {
          cputmp=MPI_Wtime();

          if (!ThisTask)
            printf("[%s] Writing output at z=%f\n",fdate(), 
                   outputs.z[iout]);

          fflush(stdout);
          MPI_Barrier(MPI_COMM_WORLD);

	  /* halo catalog */
          if (write_catalog(iout))
            return 1;

	  /* halo mass function */
          if (compute_mf(iout))
            return 1;

	  /* merger histories, only at the last redshift */
          if (iout==outputs.n-1)
            {         
              if (write_histories())
                return 1;
            }

          cputime.io += MPI_Wtime() - cputmp;

	  /* if we are at the end, just break out of the while cycle */
          iout++;
          if (this_z==nstep-1)
            break;
        }


      if (this_z!=nstep-1 && frag[iz].Fmax < zstop+1.0)
        {
          if (!ThisTask)
            printf("[%s] Pausing fragmentation process\n",fdate());
          last_z=this_z+1;
          return 0;
        }

      /************************************************************************
                          END OF DO-CYCLE ON COLLAPSED POINTS
       ************************************************************************/
    }

 build_groups_statistics:

  /* Counters */
  for (ig1=FILAMENT+1; ig1<=ngroups; ig1++)
    if (groups[ig1].point >= 0 && groups[ig1].good)
      counters[14]++;

  MPI_Reduce(counters, all_counters, NCOUNTERS, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  /* statistics */
  if (!ThisTask)
    {
      printf("Total number of peaks:                 %Lu\n",all_counters[0]);
      printf("Total number of good halos:            %Lu\n",all_counters[14]);
      printf("Particles with N neighbouring groups:  %Lu %Lu %Lu %Lu %Lu %Lu\n",
             all_counters[1],all_counters[2],all_counters[3],
             all_counters[4],all_counters[5],all_counters[6]);
      printf("Total number of accretion events:      %Lu\n",all_counters[ 7]);
      printf("Accretion before evaluating merger:    %Lu\n",all_counters[ 8]);
      printf("Accretion after evaluating merger:     %Lu\n",all_counters[ 9]);
      printf("Accretion of filament particles:       %Lu\n",all_counters[13]);
      printf("\n");
      printf("Global stats at the final redshift:\n");      
      printf("Total number of merger events:         %Lu\n",all_counters[10]);
      printf("Total number of major merger events:   %Lu\n",all_counters[11]);
      printf("Final number of filament particles:    %Lu\n",all_counters[12]);
      printf("Final number of particles in halos:    %Lu\n",all_counters[0]+all_counters[ 7]);
      printf("Total number of collapsed particles:   %Lu\n",all_counters[0] + all_counters[ 7] + all_counters[12]);
      printf("Total number of uncollapsed particles: %Lu\n",MyGrids[0].Ntotal - all_counters[0] - all_counters[ 7] - all_counters[12]);
      printf("\n");
    }


#ifdef SNAPSHOT
  /* Saving group_ID in the frag structure for the snapshot*/
  groups[0].name=0;
  groups[FILAMENT].name=FILAMENT;
  for (iz=0; iz<subbox.Nstored; iz++)
    frag[iz].group_ID = groups[group_ID[iz]].name;
#endif

#ifdef PLC

  /* close the root solver */
  gsl_root_fsolver_free (solver);

#endif

  /* Bye! */

  return 0;
}

void clean_list(int *arr)
{
  /* this function removes replications in the neighbour list and
     moves all non-empty values at the beginning */
  int i,j,a,done;

  for (j=1; j<NV; j++)
    {
      for (i=j-1; i>=0; i--)
        if (arr[i]==arr[j])
          arr[j]=0;

      a=arr[j];
      done=0;

      for (i=j-1; i>=0; i--)
        {
          if (arr[i]>0)
            {
              done=1;
              break;
            }
          arr[i+1]=arr[i];
        }

      if (!done) 
        i=-1;
      arr[i+1]=a;
    }
}


#define RMS0 1.0



PRODFLOAT virial(int mass,PRODFLOAT F,int flag)
{
  /* This function is at the root of the decision of whether particles
     should be accreted and halos should be merged. For a group of a
     given mass (in number of particles) it returns the SQUARE of its
     "virial radius", that is the capture radius for accretion and
     merging events. 
     flag=1 for accretion, flag=1 for merging.
  */

  PRODFLOAT r2,rlag,sigmaD;

  /* the Lagrangian radius in grid units is just the cubic root of the
  halo mass */
  rlag = pow((double)mass,0.333333333333333);
  int S=Smoothing.Nsmooth-1;
  /* this is the linear density rms on the grid */
  sigmaD = sqrt(Smoothing.TrueVariance[S]) * 
#ifdef SCALE_DEPENDENT
    GrowingMode((double)F-1.,Smoothing.k_GM_dens[S]); //  ATTENZIONE, QUESTO NON E` DEL TUTTO CORRETTO IN MG
#else
    GrowingMode((double)F-1.,params.k_for_GM);
#endif
  if (!flag)
    /*  merging */
    r2 = pow( f_m * pow(rlag, espo) * (sigmaD>sigmaD0 ? 1.0+(sigmaD-sigmaD0)*f_rm : 1.0) , 2.0 ) + pow( f_200 * rlag, 2.0 );
  else
    /* accretion */
    r2 = pow( f_a * pow(rlag, espo) * (sigmaD>sigmaD0 ? 1.0+(sigmaD-sigmaD0)*f_ra : 1.0) , 2.0 ) + pow( f_200 * rlag, 2.0 );



   /*-----------------MODIFIED GRAVITY WITH f(R)-----------------*/

  // DEVO CAPIRE COS'E` QUESTO CODICE QUI SOTTO!!!

#ifdef SALTALA
  // FILE *output_file_rlag = fopen("output_rlag_values.txt", "w");
  int interpolation_Done = 0;

  for (int iradius = 0; iradius <= S  ; iradius++) {
    if (rlag * params.InterPartDist >= Smoothing.Radius[iradius]) {

      // Check if interpolation has already been done for this rlag
      if (interpolation_Done == 0) { 

	// Perform reverse linear interpolation to find the k value
	//printf("your rlag is smaller than this smoothing radius: %.6f\n", Smoothing.Radius[iradius - 1]);
	double r_lower = Smoothing.Radius[iradius];
	double r_upper = Smoothing.Radius[iradius - 1];
	double k_lower = Smoothing.k_GM_dens[iradius];
	double k_upper = Smoothing.k_GM_dens[iradius - 1];

	// Reverse linear interpolation formula
	// Risultato identico all'interpolazione lineare di GSL. Si potrebbe usare per l'interpolazione nei tempi di collasso? Così si può portare su GPU senza GSL !
	// double k_interpolated = k_lower + (k_upper - k_lower) * ((rlag * params.InterPartDist - r_lower) / (r_upper - r_lower));
                                
	// Define arrays for r and k values
	double r_values[2] = {r_lower, r_upper};
	double k_values[2] = {k_lower, k_upper};

	// Perform linear interpolation using GSL
	gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, 2);
	gsl_interp_init(interp, r_values, k_values, 2);
                
	double r_value = rlag * params.InterPartDist;
	double k_interpolated = gsl_interp_eval(interp, r_values, k_values, r_value, NULL);

	//Print data to the terminal for debugging in an organized layout
	//printf("rlag: %.6f\t r_lower: %.6f\t r_upper: %.6f\t k_lower: %.6f\t k_upper: %.6f\t k_interpolated: %.6f\n",
	//rlag * params.InterPartDist, r_lower, r_upper, k_lower, k_upper, k_interpolated);
	//printf("\n");
	// Set the flag to indicate that interpolation has been done for this rlag
	interpolation_Done = 1;
	// gsl_interp_free(interp);
      }
      
    }
    
  }
        
  // fclose(output_file_rlag);

#endif

  return r2;

}


/* ACCRETION AND MERGING */

void merge_groups(int grp1,int grp2,PRODFLOAT time)
{
  /* Merges grp2 into grp1 */

  int i1;

  /* Safety: skip if either group was already merged or uninitialized */
  if (groups[grp1].point < 0 || groups[grp2].point < 0 ||
      groups[grp1].Mass <= 0 || groups[grp2].Mass <= 0)
    return;

  /* Safety limit for linking list traversals */
  int maxiter = groups[grp1].Mass + groups[grp2].Mass + 10;

#ifdef SNAPSHOT
  /* updates zacc of particles in case one of the halos (or both)
     was below the MinHaloMass threshold but the merged halo is above
     the threshold */
  if ( (groups[grp1].t_appear==-1 || groups[grp2].t_appear==-1)
       && (groups[grp1].Mass + groups[grp2].Mass >= params.MinHaloMass) )
    {
      int iter;
      if ( groups[grp1].t_appear==-1 )
        {
          i1=groups[grp1].point;
          iter=0;
          while (linking_list[i1]!=groups[grp1].point && ++iter<maxiter)
            {
              frag[i1].zacc=time-1.0;
              i1=linking_list[i1];
            }
          if (iter>=maxiter)
            { printf("ERROR: infinite loop in merge_groups SNAPSHOT grp1=%d Mass=%d\n",grp1,groups[grp1].Mass); fflush(stdout); return; }
          frag[i1].zacc=time-1.0;
        }

      if ( groups[grp2].t_appear==-1 )
        {
          i1=groups[grp2].point;
          iter=0;
          while (linking_list[i1]!=groups[grp2].point && ++iter<maxiter)
            {
              frag[i1].zacc=time-1.0;
              i1=linking_list[i1];
            }
          if (iter>=maxiter)
            { printf("ERROR: infinite loop in merge_groups SNAPSHOT grp2=%d Mass=%d\n",grp2,groups[grp2].Mass); fflush(stdout); return; }
          frag[i1].zacc=time-1.0;
        }
    }
#endif

  /* updates the linking list */
  {
    int iter=0;
    i1=groups[grp2].point;
    while (linking_list[i1] != groups[grp2].point && ++iter<maxiter)
      {
        group_ID[i1]=grp1;
        i1=linking_list[i1];
      }
    if (iter>=maxiter)
      { printf("ERROR: infinite loop in merge_groups main grp2=%d point=%d Mass=%d\n",grp2,groups[grp2].point,groups[grp2].Mass); fflush(stdout); return; }
    group_ID[i1]=grp1;
  }

  /* joins the linking lists */
  linking_list[groups[grp1].bottom]=groups[grp2].point;
  linking_list[groups[grp2].bottom]=groups[grp1].point;
  groups[grp1].bottom=groups[grp2].bottom;
  groups[grp2].point=-1;
  groups[grp2].bottom=-1;

  /* updates merger history in the new group */
  if (groups[grp1].Mass >= params.MinHaloMass && groups[grp2].Mass >= params.MinHaloMass)
    update_history(grp1,grp2,time);

  /* updates the centre position and masses, and the existence flag */
  set_obj(grp1,time,&obj1);
  set_obj(grp2,time,&obj2);

  update(&obj1,&obj2);
  set_group(grp1,&obj1);

  if (groups[grp1].Mass >= params.MinHaloMass && groups[grp1].t_appear==-1)
    groups[grp1].t_appear=time;

  /* Bye! */
}


void update_history(int g1,int g2,PRODFLOAT time)
{
  /* Upgrade the history of a group: g1 flows into g2 */

  int old_i;

  /* if groups have no branches */
  if (groups[g1].ll == g1 && groups[g2].ll == g2)
    {
      groups[g1].ll=g2;
      groups[g2].ll=g1;
    }
  /* if g1 has branches but g2 is a single halo */
  else if (groups[g1].ll != g1 && groups[g2].ll == g2)
    {
      groups[g2].ll=g1;
      old_i=g1;
      while (groups[old_i].ll != g1)
        old_i=groups[old_i].ll;
      groups[old_i].ll=g2;
    }
  /* if g1 is a single halo and g2 has branches */
  else if (groups[g1].ll == g1 && groups[g2].ll != g2)
    {
      old_i=g2;
      while (groups[old_i].ll != g2)
        {
          old_i=groups[old_i].ll;
          groups[old_i].halo_app=g1;
        }
      groups[g2].halo_app=g1;
      groups[g1].ll=groups[g2].ll;
      groups[g2].ll=g1;
    }
  else
    /* both the groups have branches */
    {
      old_i=g2;
      while (groups[old_i].ll != g2)
        {
          old_i=groups[old_i].ll;
          groups[old_i].halo_app=g1;
        }
      old_i=g1;
      while (groups[old_i].ll != g1)
        old_i=groups[old_i].ll;
      groups[old_i].ll=groups[g2].ll;
      groups[g2].ll=g1;
    }

  groups[g2].halo_app=g1;
  groups[g2].t_merge=time;
  groups[g2].mass_at_merger=groups[g1].Mass;
  groups[g2].merged_with=g1;
}


void accretion(int group,int i,int j,int k,int indx,PRODFLOAT F)
{
  /* Accretes the point (i,j,k) on the group */

  /* Safety: skip if group was merged away or uninitialized */
  if (groups[group].point < 0 || groups[group].Mass <= 0)
    return;

  set_obj(group,F,&obj1);
  set_point(i,j,k,indx,F,&obj2);
  update(&obj1,&obj2);
  set_group(group,&obj1);

  /* if the group goes above the MinHaloMass threshold, set its appearing time */
  if (groups[group].Mass >= params.MinHaloMass && groups[group].t_appear==-1)
    {
      groups[group].t_appear=F;

#ifdef SNAPSHOT
      /* updates zacc for all group particles */
      {
        int i1=groups[group].point;
        int iter=0, maxiter=groups[group].Mass+10;
        while (linking_list[i1]!=groups[group].point && ++iter<maxiter)
          {
            frag[i1].zacc=F-1.0;
            i1=linking_list[i1];
          }
        if (iter>=maxiter)
          { printf("ERROR: infinite loop in accretion SNAPSHOT grp=%d Mass=%d\n",group,groups[group].Mass); fflush(stdout); }
        else
          frag[i1].zacc=F-1.0;
      }
#endif
    }

  group_ID[indx]=group;

  /* Updates the linking list */
  linking_list[groups[group].bottom]=indx;
  groups[group].bottom=indx;
  linking_list[indx]=groups[group].point;

#ifdef SNAPSHOT
  /* accretion redshift */
  if (groups[group].Mass >= params.MinHaloMass)
    frag[indx].zacc=F-1.0;
#endif
}

/* CONDITIONS */


void condition_for_accretion(int call, int i,int j,int k,int ind,PRODFLOAT Fmax, int grp,double *dd,double *rr)
{
  /* Checks whether a particle should be accreted on a group */

  double dx=0,dy=0,dz=0,d2;

  /* compute the capture radius of the group */
  *rr=virial(groups[grp].Mass,Fmax,1);
  *dd=100.* *rr;

  set_point(i,j,k,ind,Fmax,&obj1);
  set_obj(grp,Fmax,&obj2);

  dy=dz=0.0;

  /* to shorten the calculation, distances are checked dim by dim */
  dx=distance(0,&obj1,&obj2);
  d2=dx*dx;
  if (d2<*rr)
    {
      dy=distance(1,&obj1,&obj2);
      d2+=dy*dy;
      if (d2<*rr)
        {
          dz=distance(2,&obj1,&obj2);
          d2+=dz*dz;
          if (d2<=*rr)
            *dd=d2;
        }
    }

}


void condition_for_merging(PRODFLOAT Fmax,int grp1,int grp2,int *merge_flag)
{
  /* Checks whether two groups must be merged */

  double rvir1,rvir2,dd,rr,dx,dy,dz;

  *merge_flag=0;
  rvir1=virial(groups[grp1].Mass,Fmax,0);
  rvir2=virial(groups[grp2].Mass,Fmax,0);
  rr=(rvir1>rvir2 ? rvir1 : rvir2);

  set_obj(grp1,Fmax,&obj1);
  set_obj(grp2,Fmax,&obj2);

  dx=distance(0,&obj1,&obj2);
  dd=dx*dx;
  if (dd<rr)
    {
      dy=distance(1,&obj1,&obj2);
      dd+=dy*dy;
      if (dd<rr)
        {
          dz=distance(2,&obj1,&obj2);
          dd+=dz*dz;
          if (dd<=rr)
            *merge_flag=1;
        }
    }
}



/* DISPLACEMENTS */

void set_obj(int grp,PRODFLOAT F,pos_data *myobj)
{
  /* sets all the quantities needed to handle a group */

  myobj->M=groups[grp].Mass;
  myobj->z=F-1.0;

#ifdef SCALE_DEPENDENT
  /* Group velocities are averaged over group particles, so their growth  
     should be computed on a different scale */

  /* Lagrangian radius of the object */
  // QUESTA INTERPOLAZIONE SAREBBE DA CONTROLLARE
  myobj->R=pow((double)(myobj->M)*3./4./PI,1./3.)*params.InterPartDist;  // COSTANTE GIUSTA?
  double interp=(1.-myobj->R/Smoothing.Rad_GM[0])*(double)(Smoothing.Nsmooth-1);
  interp=(interp<0.?0.:interp);
  int indx=(int)interp;
  double w=interp-(double)indx;

  /* the scale to compute the growth rate is obtaine by linear
     interpolation of log k in time */
  myobj->myk= pow(10., log10(Smoothing.k_GM_displ[indx])*(1.-w) + log10(Smoothing.k_GM_displ[indx+1])*w);
#else
  myobj->myk=params.k_for_GM;
#endif

  set_weight(myobj);

  myobj->M=groups[grp].Mass;
  for (int i=0;i<3;i++)
    {
      myobj->q[i]=groups[grp].Pos[i];
      myobj->v[i]=groups[grp].Vel[i];
#ifdef TWO_LPT
      myobj->v2[i]=groups[grp].Vel_2LPT[i];
#ifdef THREE_LPT
      myobj->v31[i]=groups[grp].Vel_3LPT_1[i];
      myobj->v32[i]=groups[grp].Vel_3LPT_2[i];
#endif
#endif

#ifdef RECOMPUTE_DISPLACEMENTS
      myobj->v_prev[i]=groups[grp].Vel_prev[i];
#ifdef TWO_LPT
      myobj->v2_prev[i]=groups[grp].Vel_2LPT_prev[i];
#ifdef THREE_LPT
      myobj->v31_prev[i]=groups[grp].Vel_3LPT_1_prev[i];
      myobj->v32_prev[i]=groups[grp].Vel_3LPT_2_prev[i];
#endif
#endif
#endif

    }

}


void set_weight(pos_data *myobj)
{
  /* sets the weight in a pos_data object for interpolating displacements */
  if (!ScaleDep.myseg)
    {
      /* at the first fragmentation segment, the interpolation weight
	 is just the growing mode at redshift F-1 divided by the "final" one */
      myobj->w   = GrowingMode(       myobj->z, myobj->myk) / GrowingMode(       ScaleDep.z[ScaleDep.myseg], myobj->myk);
#ifdef TWO_LPT
      myobj->w2  = GrowingMode_2LPT(  myobj->z, myobj->myk) / GrowingMode_2LPT(  ScaleDep.z[ScaleDep.myseg], myobj->myk);
#ifdef THREE_LPT
      myobj->w31 = GrowingMode_3LPT_1(myobj->z, myobj->myk) / GrowingMode_3LPT_1(ScaleDep.z[ScaleDep.myseg], myobj->myk);
      myobj->w32 = GrowingMode_3LPT_2(myobj->z, myobj->myk) / GrowingMode_3LPT_2(ScaleDep.z[ScaleDep.myseg], myobj->myk);
#endif
#endif
    }
  else
    {
      /* after, the weight linearly interpolates between the two redshifts */
      myobj->w   = (GrowingMode(                         myobj->z, myobj->myk) - GrowingMode(       ScaleDep.z[ScaleDep.myseg-1], myobj->myk)) /
                   (GrowingMode(       ScaleDep.z[ScaleDep.myseg], myobj->myk) - GrowingMode(       ScaleDep.z[ScaleDep.myseg-1], myobj->myk) );
#ifdef TWO_LPT
      myobj->w2  = (GrowingMode_2LPT(                    myobj->z, myobj->myk) - GrowingMode_2LPT(  ScaleDep.z[ScaleDep.myseg-1], myobj->myk)) /
                   (GrowingMode_2LPT(  ScaleDep.z[ScaleDep.myseg], myobj->myk) - GrowingMode_2LPT(  ScaleDep.z[ScaleDep.myseg-1], myobj->myk) );
#ifdef THREE_LPT
      myobj->w31 = (GrowingMode_3LPT_1(                  myobj->z, myobj->myk) - GrowingMode_3LPT_1(ScaleDep.z[ScaleDep.myseg-1], myobj->myk)) /
                   (GrowingMode_3LPT_1(ScaleDep.z[ScaleDep.myseg], myobj->myk) - GrowingMode_3LPT_1(ScaleDep.z[ScaleDep.myseg-1], myobj->myk));
      myobj->w32 = (GrowingMode_3LPT_2(                  myobj->z, myobj->myk) - GrowingMode_3LPT_2(ScaleDep.z[ScaleDep.myseg-1], myobj->myk)) /
                   (GrowingMode_3LPT_2(ScaleDep.z[ScaleDep.myseg], myobj->myk) - GrowingMode_3LPT_2(ScaleDep.z[ScaleDep.myseg-1], myobj->myk) );   
#endif
#endif
    }

}

void set_obj_vel(int grp,PRODFLOAT F,pos_data *myobj)
{
  /* this sets the growth rates for peculiar velocities for a group */
  PRODFLOAT fac=Hubble(myobj->z)/(1.+myobj->z)*params.InterPartDist;

  myobj->Dv  = fac * fomega(myobj->z, myobj->myk);
#ifdef TWO_LPT
  myobj->D2v = fac * fomega_2LPT(myobj->z, myobj->myk);
#ifdef THREE_LPT
  myobj->D31v = fac * fomega_3LPT_1(myobj->z, myobj->myk);
  myobj->D32v = fac * fomega_3LPT_2(myobj->z, myobj->myk);
#endif
#endif

}



void set_point(int i,int j,int k,int ind,PRODFLOAT F,pos_data *myobj)
{
  /* sets all the quantities needed to handle a particle */

  myobj->z=F-1.0;

#ifdef SCALE_DEPENDENT
  int S=Smoothing.Nsmooth-1;
  myobj->myk=Smoothing.k_GM_displ[S];
#else
  myobj->myk=params.k_for_GM;
#endif

  set_weight(myobj);

  myobj->M=1;
  myobj->q[0]=i+SHIFT;
  myobj->q[1]=j+SHIFT;
  myobj->q[2]=k+SHIFT;
  myobj->v[0]=frag[ind].Vel[0];
  myobj->v[1]=frag[ind].Vel[1];
  myobj->v[2]=frag[ind].Vel[2];
#ifdef TWO_LPT
  myobj->v2[0]=frag[ind].Vel_2LPT[0];
  myobj->v2[1]=frag[ind].Vel_2LPT[1];
  myobj->v2[2]=frag[ind].Vel_2LPT[2];
#ifdef THREE_LPT
  myobj->v31[0]=frag[ind].Vel_3LPT_1[0];
  myobj->v31[1]=frag[ind].Vel_3LPT_1[1];
  myobj->v31[2]=frag[ind].Vel_3LPT_1[2];
  myobj->v32[0]=frag[ind].Vel_3LPT_2[0];
  myobj->v32[1]=frag[ind].Vel_3LPT_2[1];
  myobj->v32[2]=frag[ind].Vel_3LPT_2[2];
#endif
#endif

#ifdef RECOMPUTE_DISPLACEMENTS
  myobj->v_prev[0]=frag[ind].Vel_prev[0];
  myobj->v_prev[1]=frag[ind].Vel_prev[1];
  myobj->v_prev[2]=frag[ind].Vel_prev[2];
#ifdef TWO_LPT
  myobj->v2_prev[0]=frag[ind].Vel_2LPT_prev[0];
  myobj->v2_prev[1]=frag[ind].Vel_2LPT_prev[1];
  myobj->v2_prev[2]=frag[ind].Vel_2LPT_prev[2];
#ifdef THREE_LPT
  myobj->v31_prev[0]=frag[ind].Vel_3LPT_1_prev[0];
  myobj->v31_prev[1]=frag[ind].Vel_3LPT_1_prev[1];
  myobj->v31_prev[2]=frag[ind].Vel_3LPT_1_prev[2];
  myobj->v32_prev[0]=frag[ind].Vel_3LPT_2_prev[0];
  myobj->v32_prev[1]=frag[ind].Vel_3LPT_2_prev[1];
  myobj->v32_prev[2]=frag[ind].Vel_3LPT_2_prev[2];
#endif
#endif

#endif

}


void set_group(int grp,pos_data *myobj)
{
  /* copies relevant information from an object to a group data */

  groups[grp].Mass=myobj->M;
  for (int i=0;i<3;i++)
    {
      groups[grp].Pos[i]=myobj->q[i];
      groups[grp].Vel[i]=myobj->v[i];
#ifdef TWO_LPT
      groups[grp].Vel_2LPT[i]=myobj->v2[i];
#ifdef THREE_LPT
      groups[grp].Vel_3LPT_1[i]=myobj->v31[i];
      groups[grp].Vel_3LPT_2[i]=myobj->v32[i];
#endif
#endif

#ifdef RECOMPUTE_DISPLACEMENTS
      groups[grp].Vel_prev[i]=myobj->v_prev[i];
#ifdef TWO_LPT
      groups[grp].Vel_2LPT_prev[i]=myobj->v2_prev[i];
#ifdef THREE_LPT
      groups[grp].Vel_3LPT_1_prev[i]=myobj->v31_prev[i];
      groups[grp].Vel_3LPT_2_prev[i]=myobj->v32_prev[i];
#endif
#endif
#endif
    }
}


PRODFLOAT q2x(int i, pos_data *myobj, int pbc, double Box, int order)
{
  /* it moves an object from the Lagrangian to the Eulerian space,
   in sub-box coordinates */

  PRODFLOAT pos;

  if (!ScaleDep.myseg)
    {
      /* if the redshift falls before the end of the first segment
         (including the case when fragmentation is not segmented)
         then there is no need to interpolate */
      pos = myobj->q[i] + myobj->w * myobj->v[i];
#ifdef TWO_LPT
      if (order>1)
        pos += myobj->w2 * myobj->v2[i];
#ifdef THREE_LPT
      if (order>2)
        pos += myobj->w31 * myobj->v31[i]
            +  myobj->w32 * myobj->v32[i];
#endif
#endif

    }
#ifdef RECOMPUTE_DISPLACEMENTS
  else
    {
      /* else interpolate the velocities among two segments */
      pos = myobj->q[i] + (1.-myobj->w)*myobj->v_prev[i] + myobj->w*myobj->v[i];
#ifdef TWO_LPT
      if (order>1)
        pos += (1.-myobj->w2)*myobj->v2_prev[i] + myobj->w2*myobj->v2[i];
#ifdef THREE_LPT
      if (order>2) 
        pos += (1.-myobj->w31)*myobj->v31_prev[i] + myobj->w31*myobj->v32[i]
	    +  (1.-myobj->w32)*myobj->v32_prev[i] + myobj->w32*myobj->v32[i];
#endif
#endif
    }
#endif

  if (pbc)
    {
      /* impose periodic boundary conditions if required */
      if (pos>=Box) pos-=Box;
      if (pos< 0.0) pos+=Box;
    }

  return pos;
}



PRODFLOAT vel(int i, pos_data *myobj)
{
  /* this gives the velocity of a group in km/s */
  PRODFLOAT vv;

  if (!ScaleDep.myseg)
    {
      /* if the redshift falls before the end of the first segment
         (including the case when fragmentation is not segmented)
         then there is no need to interpolate */
      vv = myobj->v[i] * myobj->Dv * myobj->w;
#ifdef TWO_LPT
      vv += myobj->v2[i] * myobj->D2v  * myobj->w2;
#ifdef THREE_LPT
      vv += myobj->v31[i] * myobj->D31v * myobj->w31
         +  myobj->v32[i] * myobj->D32v * myobj->w32;
#endif
#endif

    }
#ifdef RECOMPUTE_DISPLACEMENTS
  else
    {
      /* else interpolate the velocities among two segments */
      vv = (myobj->v_prev[i]*(1.-myobj->w) + myobj->v[i]*myobj->w) * myobj->Dv;
#ifdef TWO_LPT
      vv += (myobj->v2_prev[i]*(1.-myobj->w2) + myobj->v2[i]*myobj->w2) * myobj->D2v;
#ifdef THREE_LPT
      vv += (myobj->v31_prev[i]*(1.-myobj->w31) + myobj->v31[i]*myobj->w31) * myobj->D31v
         +  (myobj->v32_prev[i]*(1.-myobj->w32) + myobj->v32[i]*myobj->w32) * myobj->D32v;
#endif
#endif
    }
#endif

  return vv;
}


PRODFLOAT distance(int i,pos_data *obj1,pos_data *obj2)
{
  /* return the 1D distance of two objects, respecting PBCs */

  PRODFLOAT d;

  /* here displacements are computed at the ORDER_FOR_GROUPS order */
  d = q2x(i,obj2,subbox.pbc[i],(double)subbox.Lgwbl[i],ORDER_FOR_GROUPS)
    - q2x(i,obj1,subbox.pbc[i],(double)subbox.Lgwbl[i],ORDER_FOR_GROUPS);

  if (subbox.pbc[i])
    {
      PRODFLOAT halfL=(PRODFLOAT)subbox.Lgwbl[i]/2.;
      if (d >  halfL) d -= subbox.Lgwbl[i];
      if (d < -halfL) d += subbox.Lgwbl[i];
    }

  return d;
}


/* ACCRETION AND MERGING */


void update(pos_data *obj1, pos_data *obj2)
{
  /* Updates a group after a merging or accretion */

  double d;
  int i;

  for (i=0;i<3;i++)
    {
      d = fabs(obj1->q[i] - obj2->q[i]);
      if (!subbox.pbc[i])
        obj1->q[i] = (obj1->q[i]*obj1->M + obj2->q[i]*obj2->M)/(double)(obj1->M + obj2->M);
      else
        {
          PRODFLOAT halfL=subbox.Lgwbl[i]/2.;
          /* PBC must be considered */
          if (d <= halfL)
            /* in this case the distance without PBC is correct */
            obj1->q[i] = (obj1->q[i]*obj1->M + obj2->q[i]*obj2->M)/(double)(obj1->M + obj2->M);
          else if (obj1->q[i] > halfL)
            /* in this case xc is in the second half, and obj2->q[i] -> obj2->q[i]+Lgrid */
            obj1->q[i] = (obj1->q[i]*obj1->M+(obj2->q[i]+subbox.Lgwbl[i])*obj2->M)/(double)(obj1->M+obj2->M);
          else
            /* in this case xc is in the first half, and obj2->q[i] -> obj2->q[i]-Lgrid */
            obj1->q[i] = (obj1->q[i]*obj1->M+(obj2->q[i]-subbox.Lgwbl[i])*obj2->M)/(double)(obj1->M+obj2->M);

          /* checks PBC again */
          if (subbox.pbc[i] && obj1->q[i]>subbox.Lgwbl[i]) obj1->q[i]-=subbox.Lgwbl[i];
          if (subbox.pbc[i] && obj1->q[i]<0) obj1->q[i]+=subbox.Lgwbl[i];
        }

      /* velocity */
      obj1->v[i] = (obj1->v[i]*obj1->M + obj2->v[i]*obj2->M)/(double)(obj1->M + obj2->M);
#ifdef TWO_LPT
      obj1->v2[i] = (obj1->v2[i]*obj1->M + obj2->v2[i]*obj2->M)/(double)(obj1->M + obj2->M);
#ifdef THREE_LPT
      obj1->v31[i] = (obj1->v31[i]*obj1->M + obj2->v31[i]*obj2->M)/(double)(obj1->M + obj2->M);
      obj1->v32[i] = (obj1->v32[i]*obj1->M + obj2->v32[i]*obj2->M)/(double)(obj1->M + obj2->M);
#endif
#endif

#ifdef RECOMPUTE_DISPLACEMENTS

      obj1->v_prev[i] = (obj1->v_prev[i]*obj1->M + obj2->v_prev[i]*obj2->M)/(double)(obj1->M + obj2->M);
#ifdef TWO_LPT
      obj1->v2_prev[i] = (obj1->v2_prev[i]*obj1->M + obj2->v2_prev[i]*obj2->M)/(double)(obj1->M + obj2->M);
#ifdef THREE_LPT
      obj1->v31_prev[i] = (obj1->v31_prev[i]*obj1->M + obj2->v31_prev[i]*obj2->M)/(double)(obj1->M + obj2->M);
      obj1->v32_prev[i] = (obj1->v32_prev[i]*obj1->M + obj2->v32_prev[i]*obj2->M)/(double)(obj1->M + obj2->M);
#endif
#endif

#endif
    }

  obj1->M+=obj2->M;

  /* bye! */
}

#ifdef PLC
#define MAX_ITER 100

double condition_F(double F, void *p)
{
  return condition_PLC((PRODFLOAT)F);
}

double condition_PLC(PRODFLOAT F)
{
  /* Condition for a group being inside or outside the PLC 
     It computes the difference between the comoving distance of the
     group from the observer (given the replication) and the comoving
     distance at redshift z=F-1 */

  int i;
  double diff1,condition;

  set_obj(thisgroup,F,&obj1);

  for (i=0, condition=0.0; i<3; i++)
    {
      /* displacement is done up to ORDER_FOR_CATALOG */
      diff1 = q2x(i,&obj1,subbox.pbc[i],(double)subbox.Lgwbl[i],ORDER_FOR_CATALOG) + subbox.stabl[i] - ( plc.center[i] -
                                          MyGrids[0].GSglobal[i]*replicate[i] );
      condition+=diff1*diff1;
    }

  condition = sqrt(condition) - ComovingDistance((double)F-1.0)/params.InterPartDist;

  return condition;
}


int store_PLC(PRODFLOAT F)
{
  /* store halos in the PLC structure, ready to be written on file */

  int i,ii;
  PRODFLOAT x[3];
  double rhor,theta,phi;
  static int give_message=0;

  if (plc.Nstored==plc.Nmax)
    {
      if (!give_message)
        {
          printf("ERROR on task %d: PLC storage overshooted\n",ThisTask);
          printf("The PLC output will be incomplete\n");
          fflush(stdout);
          give_message=1;
        }
      return 0;
    }

  set_obj(thisgroup,F,&obj1);
  set_obj_vel(thisgroup,F,&obj1);
  for (i=0; i<3; i++)
    {
      /* displacement is done up to ORDER_FOR_CATALOG */
      x[i] = params.InterPartDist * 
        ( q2x(i,&obj1,subbox.pbc[i],(double)subbox.Lgwbl[i],ORDER_FOR_CATALOG) + subbox.stabl[i] - ( plc.center[i] - MyGrids[0].GSglobal[i]*replicate[i] ) );
    }

  coord_transformation_cartesian_polar(x,&rhor,&theta,&phi);
  if (90.-theta<params.PLCAperture)
    {

      plcgroups[plc.Nstored].z    = F-1.0;
      plcgroups[plc.Nstored].Mass = groups[thisgroup].Mass;
      plcgroups[plc.Nstored].name = groups[thisgroup].name;

      for (i=0; i<3; i++)
        {
          /* displacement is done up to ORDER_FOR_CATALOG */
          plcgroups[plc.Nstored].x[i] = x[i];
          plcgroups[plc.Nstored].v[i] = vel(i,&obj1);
        }
      /* plcgroups[plc.Nstored].rhor=rhor; */
      /* plcgroups[plc.Nstored].theta=theta; */
      /* plcgroups[plc.Nstored].phi=phi; */


      int iz = (int)((plcgroups[plc.Nstored].z - params.LastzForPLC)/plc.delta_z);
      if (iz==plc.nzbins)
        iz--;
      plc.nz[iz]+=1.0;

      plc.Nstored++;
    }

  return 0;
}

void coord_transformation_cartesian_polar(PRODFLOAT *x, double *rho, double *theta, double *phi)
{
  /* transformation from cartesian coordinates to polar */

  *rho   = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
  if (*rho>0)
    {
      *theta = -acos((x[0]*plc.zvers[0]+x[1]*plc.zvers[1]+x[2]*plc.zvers[2])/ *rho) * 180./PI + 90.;
      *phi   = atan2(x[0]*plc.yvers[0]+x[1]*plc.yvers[1]+x[2]*plc.yvers[2],
                     x[0]*plc.xvers[0]+x[1]*plc.xvers[1]+x[2]*plc.xvers[2]) * 180./PI;  
      if (*phi<0) *phi+=360.;
    }
  else
    {
      *theta=90.0;
      *phi=0.0;
    }
}


double find_brent(double x_hi, double x_lo)
{
  /* brent root finder for a numerical equation */

  int iter, status;
  double r;

  gsl_root_fsolver_set (solver, &cPLC, x_lo, x_hi);

  iter=0;
  do
    {
      iter++;
      status = gsl_root_fsolver_iterate (solver);
      r = gsl_root_fsolver_root (solver);
      x_lo = gsl_root_fsolver_x_lower (solver);
      x_hi = gsl_root_fsolver_x_upper (solver);
      /* status = gsl_root_test_interval (x_lo, x_hi, brent_err, 0.001); */
      /* if (status == GSL_SUCCESS) */
      /*        return r; */
      
      if (fabs(condition_PLC(r)) < brent_err)
        return r;

      else
        status=GSL_CONTINUE;

    }
  while (status == GSL_CONTINUE && iter < MAX_ITER);

  printf("ERROR on task %d: find_brent could not converge - %f %f\n",ThisTask,x_hi,x_lo);
  return -99.0;

}

#endif

/* Quick construction of groups */
int quick_build_groups(int Npeaks)
{
  /* limited and quick version of build_groups:
     no PLC, no counters, no outputs */
  int merge[NV][NV], neigh[NV], fil_list[NV][4];
  int nn,ifil,neigrp,pos;
  int iz,i1,j1,k1,skip;
  int ig3,small,large,nf,to_group,accgrp,ig1,ig2;
  int accrflag,nstep,nmerge,nstep_p,peak_cond;
  double ratio,best_ratio,d2,r2;
  int merge_flag;
  int ibox,jbox,kbox;

  /* Initializations */

  ngroups=FILAMENT;           // number of groups + filaments
  groups[FILAMENT].point = groups[FILAMENT].bottom = subbox.Nstored; // filaments are not grouped!
  for (i1=0; i1<FILAMENT; i1++)  // this is probably unnecessary, but better add it
    {
      groups[i1].point=-1;
      groups[i1].good=0;
    }

  if (!ThisTask)
    printf("[%s] Starting the quick fragmentation process\n",fdate());

  /* Calculates the number of steps required */

  nstep=subbox.Nstored;
  nstep_p=nstep/5;

  /************************************************************************
                        START OF THE CYCLE ON POINTS
   ************************************************************************/
  for (iz=0; iz<nstep; iz++)
    {
      /* More initializations */

      neigrp=0;               // number of neighbouring groups
      nf=0;                   // number of neighbouring filament points
      accrflag=0;             // if =1 all the neighbouring filaments are accreted
      for (i1=0; i1<NV; i1++)
        neigh[i1]=0;          // number of neighbours

      /* grid coordinates from the indices (sub-box coordinates) */
      INDEX_TO_COORD(frag_pos[iz],ibox,jbox,kbox,subbox.Lgwbl);

      /* skips if the point is at the border (and PBCs are not active) */
      skip=0;
      if ( !subbox.pbc[_x_] && (ibox==0 || ibox==subbox.Lgwbl[_x_]-1) ) ++skip;
      if ( !subbox.pbc[_y_] && (jbox==0 || jbox==subbox.Lgwbl[_y_]-1) ) ++skip;
      if ( !subbox.pbc[_z_] && (kbox==0 || kbox==subbox.Lgwbl[_z_]-1) ) ++skip;

      particle_name = 
        COORD_TO_INDEX((long long)((ibox + subbox.stabl[_x_] + MyGrids[0].GSglobal[_x_])%MyGrids[0].GSglobal[_x_]),
                       (long long)((jbox + subbox.stabl[_y_] + MyGrids[0].GSglobal[_y_])%MyGrids[0].GSglobal[_y_]),
                       (long long)((kbox + subbox.stabl[_z_] + MyGrids[0].GSglobal[_z_])%MyGrids[0].GSglobal[_z_]),
                       MyGrids[0].GSglobal);

      good_particle = ( ibox>=subbox.safe[_x_] && ibox<subbox.Lgwbl[_x_]-subbox.safe[_x_] && 
                        jbox>=subbox.safe[_y_] && jbox<subbox.Lgwbl[_y_]-subbox.safe[_y_] && 
                        kbox>=subbox.safe[_z_] && kbox<subbox.Lgwbl[_z_]-subbox.safe[_z_] );

      if (!skip)
        {
          peak_cond=1;   
          /* checks whether the neighbouring particles collapse later */
          for (nn=0; nn<NV; nn++)
            {
              switch (nn)
                {
                case 0:
                  i1=( subbox.pbc[_x_] && ibox==0 ? subbox.Lgwbl[_x_]-1 : ibox-1 );
                  j1=jbox;
                  k1=kbox;
                  break;
                case 1:
                  i1=( subbox.pbc[_x_] && ibox==subbox.Lgwbl[_x_]-1 ? 0 : ibox+1 );
                  j1=jbox;
                  k1=kbox;
                  break;
                case 2:
                  i1=ibox;
                  j1=( subbox.pbc[_y_] && jbox==0 ? subbox.Lgwbl[_y_]-1 : jbox-1 );
                  k1=kbox;
                  break;
                case 3:
                  i1=ibox;
                  j1=( subbox.pbc[_y_] && jbox==subbox.Lgwbl[_y_]-1 ? 0 : jbox+1 );
                  k1=kbox;
                  break;
                case 4:
                  i1=ibox;
                  j1=jbox;
                  k1=( subbox.pbc[_z_] && kbox==0 ? subbox.Lgwbl[_z_]-1 : kbox-1 );
                  break;
                case 5:
                  i1=ibox;
                  j1=jbox;
                  k1=( subbox.pbc[_z_] && kbox==subbox.Lgwbl[_z_]-1 ? 0 : kbox+1 );
                  break;
                }

              pos = find_location(i1,j1,k1);
              if (pos>=0)
                {
                  neigh[nn] = group_ID[pos];
                  peak_cond &= (frag[iz].Fmax > frag[pos].Fmax);
                }
              else
                neigh[nn] = 0;

              if (neigh[nn]==FILAMENT)
                {
                  neigh[nn]=0;
                  fil_list[nf][0]=i1;
                  fil_list[nf][1]=j1;
                  fil_list[nf][2]=k1;
                  fil_list[nf][3]=pos;
                  nf++;
                }

            }

          /* Cleans the list of neighbouring groups */
          clean_list(neigh);

          /* Number of neighbouring groups */
          for (nn=neigrp=0; nn<NV; nn++)
            if (neigh[nn]>FILAMENT)
              neigrp++;
        }
      else
        {
          peak_cond=0;
          neigrp=0;
        }


      /* Is the point a peak? */
      if (peak_cond)
       {
         /**********************************************************************
                                   FIRST CASE: PEAK
         **********************************************************************/
         /* New group */
        ngroups++;
        groups[ngroups].t_peak=frag[iz].Fmax;
        groups[ngroups].t_appear=-1;
        groups[ngroups].t_merge=-1;
        groups[ngroups].Pos[0]=ibox+SHIFT;
        groups[ngroups].Pos[1]=jbox+SHIFT;
        groups[ngroups].Pos[2]=kbox+SHIFT;
        groups[ngroups].Vel[0]=frag[iz].Vel[0];
        groups[ngroups].Vel[1]=frag[iz].Vel[1];
        groups[ngroups].Vel[2]=frag[iz].Vel[2];
#ifdef TWO_LPT
        groups[ngroups].Vel_2LPT[0]=frag[iz].Vel_2LPT[0];
        groups[ngroups].Vel_2LPT[1]=frag[iz].Vel_2LPT[1];
        groups[ngroups].Vel_2LPT[2]=frag[iz].Vel_2LPT[2];
#ifdef THREE_LPT
        groups[ngroups].Vel_3LPT_1[0]=frag[iz].Vel_3LPT_1[0];
        groups[ngroups].Vel_3LPT_1[1]=frag[iz].Vel_3LPT_1[1];
        groups[ngroups].Vel_3LPT_1[2]=frag[iz].Vel_3LPT_1[2];
        groups[ngroups].Vel_3LPT_2[0]=frag[iz].Vel_3LPT_2[0];
        groups[ngroups].Vel_3LPT_2[1]=frag[iz].Vel_3LPT_2[1];
        groups[ngroups].Vel_3LPT_2[2]=frag[iz].Vel_3LPT_2[2];
#endif
#endif
        groups[ngroups].Mass=1;
        groups[ngroups].name=particle_name;
        groups[ngroups].good = good_particle;
        groups[ngroups].point = iz;
        groups[ngroups].bottom = iz;
        groups[ngroups].ll=ngroups;
        groups[ngroups].halo_app=ngroups;
        group_ID[iz]=ngroups;
        linking_list[iz]=iz;

       }
     else if (neigrp==1)
       {

         /**********************************************************************
                                  SECOND CASE: 1 GROUP
          **********************************************************************/
         /* if the points touches only one group, check whether to accrete the point on it */

         condition_for_accretion(1,ibox,jbox,kbox,iz,frag[iz].Fmax,neigh[0],&d2,&r2);
         if (d2<r2)
             {
               accrflag=1;
               to_group=neigh[0];
               accretion(to_group,ibox,jbox,kbox,iz,frag[iz].Fmax);
             }
           else
             {
               groups[FILAMENT].Mass++;
               group_ID[iz]=FILAMENT;
               linking_list[iz]=iz;
             }
       }
     else if (neigrp>1)
       {
         /**********************************************************************
                                   THIRD CASE: >1 GROUP
          **********************************************************************/
         /* In this case the point touches more than one group */

         best_ratio=pow(10.*subbox.Lgwbl[_x_],2.0);
         accgrp=-1;
         for (ig1=0; ig1<neigrp; ig1++)
           {
             condition_for_accretion(2,ibox,jbox,kbox,iz,frag[iz].Fmax,neigh[ig1],&d2,&r2);
               ratio=d2/r2;
             if (ratio<1.0 && ratio<best_ratio)
             {
               best_ratio=ratio;
               accgrp=ig1;
             }
           }
         if (accgrp>=0)
           {
             accrflag=1;
             to_group=neigh[accgrp];
             accretion(neigh[accgrp],ibox,jbox,kbox,iz,frag[iz].Fmax);
           }
         /* Then checks whether the groups must be merged together */
         nmerge=0;
         for (ig1=0; ig1<neigrp; ig1++)
           for (ig2=0; ig2<ig1; ig2++)
             {
               merge[ig1][ig2]=0;
               condition_for_merging(frag[iz].Fmax,neigh[ig1],neigh[ig2],&merge_flag);
               if (merge_flag)
                 {
                   merge[ig1][ig2]=1;
                   nmerge++;
                 }
             }

         /******************
          merging of groups!
          ******************/

         /* The group number of the largest group is preserved */

         if (nmerge>0)
           {
             for (ig1=0; ig1<neigrp; ig1++)
               for (ig2=0; ig2<ig1; ig2++)
                 if (merge[ig1][ig2]==1 && neigh[ig1]!=neigh[ig2])
                   {
                     if (groups[neigh[ig1]].Mass > groups[neigh[ig2]].Mass)
                       {
                         merge_groups(neigh[ig1],neigh[ig2],frag[iz].Fmax);
                         large=neigh[ig1];
                         small=neigh[ig2];
                       }
                    else
                      {
                        merge_groups(neigh[ig2],neigh[ig1],frag[iz].Fmax);
                        small=neigh[ig1];
                        large=neigh[ig2];
                      }
                     if (to_group==small) 
                       to_group=large;
                     for (ig3=0; ig3<neigrp; ig3++)
                       if (neigh[ig3]==small) 
                         neigh[ig3]=large;
                   }
           }

         /* If relevant, it tries again to accrete the particle */

         if (accgrp==-1)
           {
             clean_list(neigh);

             /* Number of neighbouring groups */
             for (nn=neigrp=0; nn<NV; nn++)
               if (neigh[nn]>FILAMENT)
                 neigrp++;

             best_ratio=pow(10.*subbox.Lgwbl[_x_],2.0);
             accgrp=-1;
             for (ig1=0; ig1<neigrp; ig1++)
               {
                 condition_for_accretion(3,ibox,jbox,kbox,iz,frag[iz].Fmax,neigh[ig1],&d2,&r2);
                   ratio=d2/r2;
                 if (ratio<best_ratio)
                   {
                     best_ratio=ratio;
                     accgrp=ig1;
                   }
               }

             if (best_ratio<1)
               {
                 accrflag=1;
                 to_group=neigh[accgrp];
                 accretion(neigh[accgrp],ibox,jbox,kbox,iz,frag[iz].Fmax);
               }
             else
               {
                 /* If the point has not been accreted at all: */
                 groups[FILAMENT].Mass++;
                 group_ID[iz]=FILAMENT;
                 linking_list[iz]=iz;
               }
           }
       }
     else
       {
         /**********************************************************************
                                 FOURTH CASE: FILAMENTS
          **********************************************************************/
         groups[FILAMENT].Mass++;
         group_ID[iz]=FILAMENT;
         linking_list[iz]=iz;

         /* end of cases */
       }

      /* Checks whether to accrete all the neighbouring filaments;
         first it checks conditions for all filament particles,  
         then it accretes those that should */
      if (accrflag && nf && !skip)
        {
          for (ifil=0; ifil<nf; ifil++)
            {
              condition_for_accretion(4,fil_list[ifil][0], fil_list[ifil][1],fil_list[ifil][2],
                                      fil_list[ifil][3],frag[iz].Fmax,to_group, &d2,&r2);
              if (d2<r2)
                fil_list[ifil][3]*=-1;
            }
          
          for (ifil=0; ifil<nf; ifil++)
            if (fil_list[ifil][3]<0)
              {
                fil_list[ifil][3]*=-1;
                accretion(to_group, fil_list[ifil][0], fil_list[ifil][1],
                          fil_list[ifil][2],fil_list[ifil][3],frag[iz].Fmax);
                groups[FILAMENT].Mass--;
              }
          }

      /************************************************************************
                          END OF DO-CYCLE ON COLLAPSED POINTS
       ************************************************************************/

      /* Fraction of steps done */
      if (!ThisTask && !(iz%nstep_p))
        printf("[%s] *** %3d%% done, F = %6.2f,  z = %6.2f\n",fdate(),
               iz/(nstep_p)*20,frag[iz].Fmax,
               frag[iz].Fmax-1.0
               );

    }

  return 0;
}


int update_map(unsigned int *nadd)
{

  int group,ig,jg,kg,size,size2,rr,i,j,k,i1,j1,k1;
  memset(frag_map_update, 0, subbox.maplength*sizeof(unsigned int));
  nadd[0]=nadd[1]=0;

  for (group=FILAMENT+1; group<ngroups; group++)
    {
      ig=(int)(groups[group].Pos[0]+0.5);
      jg=(int)(groups[group].Pos[1]+0.5);
      kg=(int)(groups[group].Pos[2]+0.5);
      size=(int)(params.BoundaryLayerFactor*pow((double)groups[group].Mass/4.188790205,0.333333333333333)+0.5);
      size2=size*size;
      
      for (i1=ig-size; i1<ig+size; i1++)
        {
          if (i1<0 || i1>=subbox.Lgwbl[_x_])
                {
                  if (subbox.pbc[_x_])
                    i = ( i1<0 ? i1+subbox.Lgwbl[_x_] : i1-subbox.Lgwbl[_x_] );
                  else
                    i = -1;
                }
              else
                i=i1;

          for (j1=jg-size; j1<jg+size; j1++)
            {
              if (j1<0 || j1>=subbox.Lgwbl[_y_])
                {
                  if (subbox.pbc[_y_])
                    j = ( j1<0 ? j1+subbox.Lgwbl[_y_] : j1-subbox.Lgwbl[_y_] );
                  else
                    j = -1;
                }
              else
                j=j1;
              
              for (k1=kg-size; k1<kg+size; k1++)
                {
                  if (k1<0 || k1>=subbox.Lgwbl[_z_])
                    {
                      if (subbox.pbc[_z_])
                        k = ( k1<0 ? k1+subbox.Lgwbl[_z_] : k1-subbox.Lgwbl[_z_] );
                      else
                        k = -1;
                    }
                  else
                    k=k1;

                  if (i<0 || j<0 || k<0)
                    {
                      nadd[1]++;
                      continue;
                    }

                  if (!get_map_bit_coord(i,j,k))
                    {
                      rr=(i1-ig)*(i1-ig)+(j1-jg)*(j1-jg)+(k1-kg)*(k1-kg);
                      if (rr<=size2)
                        {
                          set_mapup_bit(i,j,k);
                          nadd[0]++;
                        }
                    }
                }
            }
        }
    }

  return 0;
}

