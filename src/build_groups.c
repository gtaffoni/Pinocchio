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

/* Make per-particle globals thread-private so that accretion(),
   merge_groups(), condition_for_accretion(), and condition_for_merging()
   can be called safely from parallel regions: each thread gets its own
   copies of obj1/obj2, particle_name, and good_particle.            */
#ifdef _OPENMP
#pragma omp threadprivate(obj1, obj2, particle_name, good_particle)
#endif

void set_weight(pos_data *);

/* ================================================================
   3D TILING FOR OpenMP PARALLEL GROUP FORMATION
   Tile edge T=8 grid spacings: safe for M_max ~ 10^4 particles.
   With f_a=0.18, max accretion radius = 0.18*(10^4)^(1/3) ~ 3.95 < T/2=4.
   Colors 0-7 from 8-color 3D checkerboard: no same-color tiles adjacent.
   ================================================================ */
#ifdef _OPENMP
#ifndef TILE_SIZE
#define TILE_SIZE 8
#endif
/* Local buffer dimensions: tile + 1-cell shell on each side */
#define LS   (TILE_SIZE + 2)        /* buffer side length: 10 for TILE_SIZE=8  */
#define LS2  (LS * LS)              /* 100                                     */
#define LS3  (LS * LS * LS)         /* 1000                                    */
#define TILE_SIZE3 (TILE_SIZE * TILE_SIZE * TILE_SIZE) /* 512 for TILE_SIZE=8  */
/* Index into the local buffer for local coords (li,lj,lk) ∈ [0,LS)          */
#define LIDX(li,lj,lk) ((li)*LS2 + (lj)*LS + (lk))

typedef struct {
  int *particles;  /* frag-order indices in descending Fmax order */
  int  n;          /* number of particles in this tile             */
} tile_t;

/**
 * @brief Per-thread volume buffer for tile processing (Fase A cache).
 *
 * Holds:
 *  - Frag[]:       Cached product_data for all TILE_SIZE3 tile particles.
 *                  Eliminates DRAM accesses to frag[iz].Vel at peak creation
 *                  and provides frag[iz].Fmax from L2 cache.
 *  - LocalGroups[]: Cached group_data for groups created inside this tile.
 *                  set_obj/set_group route through get_group_ptr() which
 *                  returns a pointer here for local groups, avoiding DRAM
 *                  round-trips on every accretion/merge update.
 *  - local_gids[]: Global group IDs for each LocalGroups entry.
 *  - n_local:      Number of active local groups (typically < 50 per tile).
 *  - pos_cache[]:  Cached frag_pos[] values for tile particles (replaces
 *                  the Phase-2b stack array tl_self_pos).
 */
typedef struct {
    product_data *Frag;                       /* heap, ALIGN-aligned [TILE_SIZE3]  */
    group_data   *LocalGroups;                /* heap [TILE_SIZE3 + 2]             */
    int           local_gids[TILE_SIZE3 + 2]; /* global gid → local slot mapping   */
    int           n_local;                    /* slots used in LocalGroups          */
    int           pos_cache[TILE_SIZE3];      /* frag_pos[] cache for tile particles*/
} tile_vol_t;

/* Tile infrastructure: module-level statics, valid between tile_build and tile_free */
static tile_t    *tiles_g        = NULL;
static int        n_tiles_g      = 0;
static int        NT_g[3]        = {0,0,0};
static int       *color_list_g[8];
static int        color_cnt_g[8];

/* Per-thread volume buffers: allocated once per thread in tile_build() */
static tile_vol_t *tile_vols_g   = NULL;

/* Thread-private pointer to the current thread's volume buffer.
   Set at the start of each tile, cleared at the end. */
static tile_vol_t *cur_tv_g      = NULL;
#pragma omp threadprivate(cur_tv_g)

/**
 * @brief Build per-tile particle lists and 8-color index arrays.
 *
 * Partitions particles in frag-order range [from_z, to_z] into tiles
 * of edge T grid spacings. Tiles are classified into 8 colors by their
 * (tx%2, ty%2, tz%2) checkerboard index.  Within each tile the
 * particles appear in descending Fmax order because the outer loop
 * iterates in that order.
 *
 * @param from_z  First frag-order index to include (inclusive).
 * @param to_z    Last  frag-order index to include (inclusive).
 * @param T       Tile edge length in grid spacings.
 */
static void tile_build(int from_z, int to_z, int T)
{
  int i, tid, color, ix, iy, iz_t;
  int *cnt;

  NT_g[0] = (subbox.Lgwbl[0] + T - 1) / T;
  NT_g[1] = (subbox.Lgwbl[1] + T - 1) / T;
  NT_g[2] = (subbox.Lgwbl[2] + T - 1) / T;
  n_tiles_g = NT_g[0] * NT_g[1] * NT_g[2];

  tiles_g = (tile_t *)calloc(n_tiles_g, sizeof(tile_t));
  cnt     = (int     *)calloc(n_tiles_g, sizeof(int));
  if (!tiles_g || !cnt)
    {
      fprintf(stderr, "tile_build: out of memory\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

  /* Pass 1: count particles per tile */
  for (i = from_z; i <= to_z; i++)
    {
      INDEX_TO_COORD(frag_pos[i], ix, iy, iz_t, subbox.Lgwbl);
      tid = (ix/T)*NT_g[1]*NT_g[2] + (iy/T)*NT_g[2] + (iz_t/T);
      cnt[tid]++;
    }

  /* Allocate per-tile particle arrays */
  for (tid = 0; tid < n_tiles_g; tid++)
    if (cnt[tid] > 0)
      {
        tiles_g[tid].particles = (int *)malloc(cnt[tid] * sizeof(int));
        if (!tiles_g[tid].particles)
          {
            fprintf(stderr, "tile_build: out of memory for tile %d\n", tid);
            MPI_Abort(MPI_COMM_WORLD, 1);
          }
      }

  /* Pass 2: fill particle lists in Fmax-descending order (preserved
     because outer loop runs from_z..to_z in that order)             */
  memset(cnt, 0, n_tiles_g * sizeof(int));
  for (i = from_z; i <= to_z; i++)
    {
      INDEX_TO_COORD(frag_pos[i], ix, iy, iz_t, subbox.Lgwbl);
      tid = (ix/T)*NT_g[1]*NT_g[2] + (iy/T)*NT_g[2] + (iz_t/T);
      tiles_g[tid].particles[cnt[tid]] = i;
      cnt[tid]++;
      tiles_g[tid].n = cnt[tid];
    }
  free(cnt);

  /* Organize tiles by 8-color checkerboard index */
  memset(color_cnt_g, 0, sizeof(color_cnt_g));
  for (tid = 0; tid < n_tiles_g; tid++)
    {
      int tx = tid / (NT_g[1]*NT_g[2]);
      int ty = (tid % (NT_g[1]*NT_g[2])) / NT_g[2];
      int tz = tid % NT_g[2];
      color = (tx%2) | ((ty%2)<<1) | ((tz%2)<<2);
      color_cnt_g[color]++;
    }
  for (color = 0; color < 8; color++)
    {
      color_list_g[color] = (int *)malloc(color_cnt_g[color] * sizeof(int));
      if (!color_list_g[color] && color_cnt_g[color] > 0)
        {
          fprintf(stderr, "tile_build: out of memory for color_list %d\n", color);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
  memset(color_cnt_g, 0, sizeof(color_cnt_g));
  for (tid = 0; tid < n_tiles_g; tid++)
    {
      int tx = tid / (NT_g[1]*NT_g[2]);
      int ty = (tid % (NT_g[1]*NT_g[2])) / NT_g[2];
      int tz = tid % NT_g[2];
      color = (tx%2) | ((ty%2)<<1) | ((tz%2)<<2);
      color_list_g[color][color_cnt_g[color]++] = tid;
    }

  /* Allocate per-thread volume buffers (Fase A) */
  {
    int t, nthreads = omp_get_max_threads();
    tile_vols_g = (tile_vol_t *)calloc(nthreads, sizeof(tile_vol_t));
    if (!tile_vols_g)
      {
        fprintf(stderr, "tile_build: out of memory for tile_vols_g\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    for (t = 0; t < nthreads; t++)
      {
        tile_vol_t *tv = &tile_vols_g[t];
        if (posix_memalign((void**)&tv->Frag, ALIGN,
                            TILE_SIZE3 * sizeof(product_data)) != 0)
          {
            fprintf(stderr, "tile_build: posix_memalign failed for Frag[%d]\n", t);
            MPI_Abort(MPI_COMM_WORLD, 1);
          }
        tv->LocalGroups = (group_data *)malloc((TILE_SIZE3 + 2) * sizeof(group_data));
        if (!tv->LocalGroups)
          {
            fprintf(stderr, "tile_build: out of memory for LocalGroups[%d]\n", t);
            MPI_Abort(MPI_COMM_WORLD, 1);
          }
        tv->n_local = 0;
      }
  }
}

/**
 * @brief Free all memory allocated by tile_build().
 */
static void tile_free(void)
{
  int tid, color;
  if (tiles_g)
    {
      for (tid = 0; tid < n_tiles_g; tid++)
        free(tiles_g[tid].particles);
      free(tiles_g);
      tiles_g = NULL;
    }
  for (color = 0; color < 8; color++)
    {
      free(color_list_g[color]);
      color_list_g[color] = NULL;
    }
  n_tiles_g = 0;

  /* Free per-thread volume buffers (Fase A) */
  if (tile_vols_g)
    {
      int t, nthreads = omp_get_max_threads();
      for (t = 0; t < nthreads; t++)
        {
          free(tile_vols_g[t].Frag);
          free(tile_vols_g[t].LocalGroups);
        }
      free(tile_vols_g);
      tile_vols_g = NULL;
    }
}

/**
 * @brief Fill thread-local cache buffers for one tile + 1-cell shell.
 *
 * Copies group_ID[] and frag[].Fmax into compact local arrays so that
 * the 6-neighbor accesses during tile processing hit L1 cache instead
 * of random DRAM locations.
 *
 * Buffer layout: tl_local_*[LIDX(li,lj,lk)] where (li,lj,lk) ∈ [0,LS).
 * li=0 corresponds to grid column sx-1, li=LS-1 to sx+TILE_SIZE.
 * Boundary wrapping follows subbox PBC flags; out-of-domain slots are
 * filled with (gid=0, fmax=0) which are safe sentinel values.
 *
 * @param sx            Tile start coordinate (x), in subbox grid units.
 * @param sy            Tile start coordinate (y).
 * @param sz            Tile start coordinate (z).
 * @param tl_local_gid  Output: group_ID values, size LS3.
 * @param tl_local_fmax Output: Fmax values, size LS3.
 */
static inline void fill_local_buffer(
    int sx, int sy, int sz,
    int *tl_local_gid, PRODFLOAT *tl_local_fmax)
{
  const int *Lgwbl = subbox.Lgwbl;
  int li, lj, lk;

  for (li = 0; li < LS; li++)
    {
      int gi  = sx - 1 + li;
      int gii = gi;
      int vi  = 1;
      if (gi < 0)
        { if (subbox.pbc[_x_]) gii = gi + Lgwbl[_x_]; else vi = 0; }
      else if (gi >= Lgwbl[_x_])
        { if (subbox.pbc[_x_]) gii = gi - Lgwbl[_x_]; else vi = 0; }

      for (lj = 0; lj < LS; lj++)
        {
          int gj  = sy - 1 + lj;
          int gjj = gj;
          int vj  = 1;
          if (gj < 0)
            { if (subbox.pbc[_y_]) gjj = gj + Lgwbl[_y_]; else vj = 0; }
          else if (gj >= Lgwbl[_y_])
            { if (subbox.pbc[_y_]) gjj = gj - Lgwbl[_y_]; else vj = 0; }

          for (lk = 0; lk < LS; lk++)
            {
              int gk  = sz - 1 + lk;
              int gkk = gk;
              int vk  = 1;
              if (gk < 0)
                { if (subbox.pbc[_z_]) gkk = gk + Lgwbl[_z_]; else vk = 0; }
              else if (gk >= Lgwbl[_z_])
                { if (subbox.pbc[_z_]) gkk = gk - Lgwbl[_z_]; else vk = 0; }

              int lidx = LIDX(li, lj, lk);
              if (vi && vj && vk)
                {
                  int gpos = COORD_TO_INDEX(gii, gjj, gkk, Lgwbl);
                  int fp   = sorted_pos[gpos];
                  if (fp >= 0)
                    {
                      tl_local_gid [lidx] = group_ID[fp];
                      tl_local_fmax[lidx] = frag[fp].Fmax;
                    }
                  else
                    {
                      tl_local_gid [lidx] = 0;
                      tl_local_fmax[lidx] = (PRODFLOAT)0;
                    }
                }
              else
                {
                  tl_local_gid [lidx] = 0;
                  tl_local_fmax[lidx] = (PRODFLOAT)0;
                }
            }
        }
    }
}

/* ----------------------------------------------------------------
   Fase A helpers: tile_vol_init, tile_vol_flush
   (get_group_ptr is defined after #endif _OPENMP — must be visible to
    set_obj / set_group which are compiled unconditionally)
   ---------------------------------------------------------------- */

/**
 * @brief Pre-load per-tile caches into the volume buffer.
 *
 * Copies the complete product_data and frag_pos entry for every particle
 * in @p tile into the thread-local @p tv buffer.  This single streaming
 * pass replaces the separate Phase-2 (Fmax) and Phase-2b (frag_pos)
 * pre-load loops and adds Vel caching at no extra cost.
 *
 * Resets the local group list (n_local = 0) for this tile.
 *
 * @param tv    Destination volume buffer (current thread's slot).
 * @param tile  Tile descriptor (particle list + count).
 */
static void tile_vol_init(tile_vol_t *tv, tile_t *tile)
{
  int _pi;
  tv->n_local = 0;
  for (_pi = 0; _pi < tile->n; _pi++)
    {
      int _iz           = tile->particles[_pi];
      tv->Frag[_pi]     = frag[_iz];       /* full product_data: Fmax + Vel + … */
      tv->pos_cache[_pi] = frag_pos[_iz];  /* grid-position index for this particle */
    }
}

/**
 * @brief Write back local group Vel/Pos fields to global groups[].
 *
 * During tile processing, set_group() updates Vel and Pos only in the
 * thread-local LocalGroups[] cache (Mass is kept in sync immediately).
 * This function propagates the final Vel/Pos values back to the global
 * groups[] array so that post-tile code (output, update_map, etc.)
 * sees correct data.  Groups merged during this tile (point == -1) are
 * skipped — their Vel/Pos are no longer meaningful.
 *
 * @param tv  Volume buffer whose LocalGroups entries are to be flushed.
 */
static void tile_vol_flush(tile_vol_t *tv)
{
  int li, k;
  for (li = 0; li < tv->n_local; li++)
    {
      int gid = tv->local_gids[li];
      if (groups[gid].point < 0)
        continue;  /* merged away during this tile */
      group_data *lg = &tv->LocalGroups[li];
      for (k = 0; k < 3; k++)
        {
          groups[gid].Pos[k] = lg->Pos[k];
          groups[gid].Vel[k] = lg->Vel[k];
#ifdef TWO_LPT
          groups[gid].Vel_2LPT[k] = lg->Vel_2LPT[k];
#ifdef THREE_LPT
          groups[gid].Vel_3LPT_1[k] = lg->Vel_3LPT_1[k];
          groups[gid].Vel_3LPT_2[k] = lg->Vel_3LPT_2[k];
#endif /* THREE_LPT */
#endif /* TWO_LPT */
#ifdef RECOMPUTE_DISPLACEMENTS
          groups[gid].Vel_prev[k] = lg->Vel_prev[k];
#ifdef TWO_LPT
          groups[gid].Vel_2LPT_prev[k] = lg->Vel_2LPT_prev[k];
#ifdef THREE_LPT
          groups[gid].Vel_3LPT_1_prev[k] = lg->Vel_3LPT_1_prev[k];
          groups[gid].Vel_3LPT_2_prev[k] = lg->Vel_3LPT_2_prev[k];
#endif /* THREE_LPT */
#endif /* TWO_LPT */
#endif /* RECOMPUTE_DISPLACEMENTS */
        }
    }
}

#endif /* _OPENMP */

/* ----------------------------------------------------------------
   Fase A: get_group_ptr — always compiled (called from set_obj/set_group)
   ---------------------------------------------------------------- */

/**
 * @brief Return a pointer to the group_data for @p global_gid.
 *
 * In OpenMP builds, if a tile volume buffer is active for the current thread
 * AND the group was created inside the current tile, returns a pointer to the
 * thread-local cached copy in LocalGroups[] (L2 cache).  Otherwise returns
 * the global groups[] array entry.  The linear scan over n_local (typically
 * < 50 groups per tile) takes ≈5 ns — negligible vs the DRAM miss avoided.
 *
 * In non-OpenMP builds the function always returns &groups[global_gid].
 *
 * @param global_gid  Global group ID (index into groups[]).
 * @return Pointer to the authoritative group_data for that group.
 */
static inline group_data *get_group_ptr(int global_gid)
{
#ifdef _OPENMP
  if (cur_tv_g)
    {
      int li;
      for (li = 0; li < cur_tv_g->n_local; li++)
        if (cur_tv_g->local_gids[li] == global_gid)
          return &cur_tv_g->LocalGroups[li];
    }
#endif
  return &groups[global_gid];
}

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


/* Thread-safe version of set_weight using GFLUT — retained for future OpenMP/GPU path */

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


/* NOTE: precompute_particles() and precomp_allocate/free() have been removed.
   The CPU precomputation infrastructure is replaced by precompute_particles_gpu()
   in gpu_offload.c for the GPU path. The serial fragmentation path computes
   particle positions on the fly via set_point() / q2x(). */

/* NOTE: The OpenMP parallel subdomain path (build_groups_parallel) has been
   removed. Group formation uses the serial path exclusively. A new parallel
   implementation will be developed on the feat/fragmentation-openmp branch,
   using this serial path as the reference algorithm. */

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

#if defined(_OPENMP) && !defined(CLASSIC_FRAGMENTATION)
  /* ============================================================
     OPENMP PATH: 8-color 3D checkerboard tiling parallel loop.

     Algorithm:
       1. Find the epoch range [last_z, to_z_ep] to process this call.
       2. Build tile infrastructure partitioning particles into tiles of
          TILE_SIZE grid spacings; assign each tile one of 8 colors via
          color = (tx%2) | ((ty%2)<<1) | ((tz%2)<<2).
       3. Process colors 0..7 SEQUENTIALLY.  Within each color pass,
          all tiles of that color are processed IN PARALLEL via
          `#pragma omp parallel for schedule(dynamic)`.
       4. After all 8 color passes, write any pending outputs, handle
          PLC post-pass (serial), update progress stats, handle pause.
       5. Jump to build_groups_statistics label.

     Safety guarantee: same-color tiles are separated by >= TILE_SIZE=8
     grid spacings.  With f_a ~ 0.18 and M_max ~ 10^4, the maximum
     accretion radius is ~3.95 < 4 = TILE_SIZE/2, so no two particles
     from different same-color tiles can belong to the same group.
     Peak conditions are also safe: same-color tile peaks are not
     Lagrangian neighbors.

     Thread safety:
       - ngroups increment / Npeaks check: #pragma omp atomic capture (lock-free)
       - groups[FILAMENT].Mass incr/decr:  thread-local tl_filament_delta, one
         atomic add per thread per color pass (eliminates cache-line bouncing)
       - counters[]: thread-local tl_cnt[], reduced via critical at end
       - group_ID[], linking_list[]:  each particle writes its own index
       - groups[my_group] init after critical: thread has sole ownership
       - accretion(), merge_groups(): safe – same-color tiles share no groups

     Note on small differences vs serial: boundary particles may see
     neighbors from unprocessed (later-color) tiles as group_ID=0, which
     may give slightly different accretion decisions than serial.  This
     is acceptable per the design document.
     ============================================================ */
  {
    /* ---- Determine epoch range for this call ---- */
    int to_z_ep = nstep - 1;
    {
      int tz;
      for (tz = last_z; tz < nstep; tz++)
        if (frag[tz].Fmax < zstop + 1.0) { to_z_ep = tz - 1; break; }
    }

    if (to_z_ep >= last_z)
      {
        /* ---- Build tile structure ---- */
        tile_build(last_z, to_z_ep, TILE_SIZE);

        /* ---- 8-color parallel loop ---- */
        int tcolor;
        for (tcolor = 0; tcolor < 8; tcolor++)
          {
            int par_error = 0;  /* set to 1 if fatal error inside parallel */

#pragma omp parallel shared(par_error)
            {
              /* Thread-local variables (avoid false sharing, all on stack) */
              int tl_iz, tl_pi, tl_ci;
              int tl_ibox, tl_jbox, tl_kbox;
              int tl_skip, tl_peak_cond;
              int tl_neigrp, tl_nf, tl_accrflag;
              int tl_neigh[NV];
              int tl_fil_list[NV][4];
              int tl_merge_arr[NV][NV];
              int tl_nmerge, tl_merge_flag;
              int tl_ig1, tl_ig2, tl_ig3;
              int tl_i1, tl_j1, tl_k1, tl_nn;
              double tl_d2, tl_r2, tl_ratio, tl_best_ratio;
              int tl_accgrp, tl_to_group = -1, tl_large, tl_small, tl_pos, tl_ifil;
              int tl_my_group;
              int tl_filament_delta;   /* thread-local FILAMENT.Mass delta */
              unsigned long long tl_cnt[NCOUNTERS];
              /* Local cache buffers: tile + 1-cell shell, fits in L1 (8 KB) */
              int       tl_local_gid [LS3];
              PRODFLOAT tl_local_fmax[LS3];
              /* Fase A: tl_self_fmax[] and tl_self_pos[] removed — subsumed by
                 cur_tv_g->Frag[pi].Fmax and cur_tv_g->pos_cache[pi] which are
                 populated once per tile by tile_vol_init().                      */
              int tile_sx, tile_sy, tile_sz; /* tile start coords for this tile */
              memset(tl_cnt, 0, sizeof(tl_cnt));
              tl_filament_delta = 0;

#pragma omp for schedule(dynamic,1)
              for (tl_ci = 0; tl_ci < color_cnt_g[tcolor]; tl_ci++)
                {
                  /* Skip remaining tiles if a fatal error was detected */
                  if (par_error) continue;

                  int tl_tid = color_list_g[tcolor][tl_ci];
                  tile_t *tile = &tiles_g[tl_tid];

                  /* Compute tile start coordinates from tile ID */
                  tile_sx = (tl_tid / (NT_g[1]*NT_g[2]))       * TILE_SIZE;
                  tile_sy = ((tl_tid / NT_g[2]) % NT_g[1])     * TILE_SIZE;
                  tile_sz = (tl_tid % NT_g[2])                  * TILE_SIZE;

                  /* Pre-load tile + shell into L1-sized local buffers */
                  fill_local_buffer(tile_sx, tile_sy, tile_sz,
                                    tl_local_gid, tl_local_fmax);

                  /* Fase A: unified pre-load — copies full product_data (Fmax,
                     Vel, Vel_2LPT, …) and frag_pos for all tile particles into
                     the per-thread volume buffer.  Replaces the previous Phase 2
                     (Fmax-only) and Phase 2b (frag_pos) separate init loops.    */
                  cur_tv_g = &tile_vols_g[omp_get_thread_num()];
                  tile_vol_init(cur_tv_g, tile);

                  /* Process particles in this tile in Fmax-descending order */
                  for (tl_pi = 0; tl_pi < tile->n; tl_pi++)
                    {
                      tl_iz = tile->particles[tl_pi];

                      /* === PARTICLE SETUP === */
                      tl_neigrp  = 0;
                      tl_nf      = 0;
                      tl_accrflag = 0;
                      for (tl_nn = 0; tl_nn < NV; tl_nn++) tl_neigh[tl_nn] = 0;

                      INDEX_TO_COORD(cur_tv_g->pos_cache[tl_pi], tl_ibox, tl_jbox, tl_kbox,
                                     subbox.Lgwbl);

                      tl_skip = 0;
                      if (!subbox.pbc[_x_] &&
                          (tl_ibox == 0 || tl_ibox == subbox.Lgwbl[_x_]-1)) ++tl_skip;
                      if (!subbox.pbc[_y_] &&
                          (tl_jbox == 0 || tl_jbox == subbox.Lgwbl[_y_]-1)) ++tl_skip;
                      if (!subbox.pbc[_z_] &&
                          (tl_kbox == 0 || tl_kbox == subbox.Lgwbl[_z_]-1)) ++tl_skip;

                      /* particle_name and good_particle are threadprivate */
                      particle_name =
                        COORD_TO_INDEX(
                          (long long)((tl_ibox + subbox.stabl[_x_] +
                                       MyGrids[0].GSglobal[_x_]) % MyGrids[0].GSglobal[_x_]),
                          (long long)((tl_jbox + subbox.stabl[_y_] +
                                       MyGrids[0].GSglobal[_y_]) % MyGrids[0].GSglobal[_y_]),
                          (long long)((tl_kbox + subbox.stabl[_z_] +
                                       MyGrids[0].GSglobal[_z_]) % MyGrids[0].GSglobal[_z_]),
                          MyGrids[0].GSglobal);

                      good_particle =
                        (tl_ibox >= subbox.safe[_x_] &&
                         tl_ibox <  subbox.Lgwbl[_x_] - subbox.safe[_x_] &&
                         tl_jbox >= subbox.safe[_y_] &&
                         tl_jbox <  subbox.Lgwbl[_y_] - subbox.safe[_y_] &&
                         tl_kbox >= subbox.safe[_z_] &&
                         tl_kbox <  subbox.Lgwbl[_z_] - subbox.safe[_z_]);

                      /* Own position in local buffer — valid for every particle in tile,
                         used in 6-neighbor loop and in post-particle buffer update.       */
                      int tl_li_own = tl_ibox - tile_sx + 1;
                      int tl_lj_own = tl_jbox - tile_sy + 1;
                      int tl_lk_own = tl_kbox - tile_sz + 1;

                      if (!tl_skip)
                        {
                          tl_peak_cond = 1;

                          /* === 6-NEIGHBOR LOOP === */

                          for (tl_nn = 0; tl_nn < NV; tl_nn++)
                            {
                              /* tl_lidx: index into local buffer (pre-PBC offset)
                                 tl_i1/j1/k1: PBC-wrapped grid coords (for filament) */
                              int tl_lidx;
                              switch (tl_nn)
                                {
                                case 0:
                                  tl_lidx = LIDX(tl_li_own-1, tl_lj_own, tl_lk_own);
                                  tl_i1 = (subbox.pbc[_x_] && tl_ibox == 0 ?
                                           subbox.Lgwbl[_x_]-1 : tl_ibox-1);
                                  tl_j1 = tl_jbox; tl_k1 = tl_kbox;
                                  break;
                                case 1:
                                  tl_lidx = LIDX(tl_li_own+1, tl_lj_own, tl_lk_own);
                                  tl_i1 = (subbox.pbc[_x_] &&
                                           tl_ibox == subbox.Lgwbl[_x_]-1 ?
                                           0 : tl_ibox+1);
                                  tl_j1 = tl_jbox; tl_k1 = tl_kbox;
                                  break;
                                case 2:
                                  tl_lidx = LIDX(tl_li_own, tl_lj_own-1, tl_lk_own);
                                  tl_i1 = tl_ibox;
                                  tl_j1 = (subbox.pbc[_y_] && tl_jbox == 0 ?
                                           subbox.Lgwbl[_y_]-1 : tl_jbox-1);
                                  tl_k1 = tl_kbox;
                                  break;
                                case 3:
                                  tl_lidx = LIDX(tl_li_own, tl_lj_own+1, tl_lk_own);
                                  tl_i1 = tl_ibox;
                                  tl_j1 = (subbox.pbc[_y_] &&
                                           tl_jbox == subbox.Lgwbl[_y_]-1 ?
                                           0 : tl_jbox+1);
                                  tl_k1 = tl_kbox;
                                  break;
                                case 4:
                                  tl_lidx = LIDX(tl_li_own, tl_lj_own, tl_lk_own-1);
                                  tl_i1 = tl_ibox; tl_j1 = tl_jbox;
                                  tl_k1 = (subbox.pbc[_z_] && tl_kbox == 0 ?
                                           subbox.Lgwbl[_z_]-1 : tl_kbox-1);
                                  break;
                                case 5:
                                  tl_lidx = LIDX(tl_li_own, tl_lj_own, tl_lk_own+1);
                                  tl_i1 = tl_ibox; tl_j1 = tl_jbox;
                                  tl_k1 = (subbox.pbc[_z_] &&
                                           tl_kbox == subbox.Lgwbl[_z_]-1 ?
                                           0 : tl_kbox+1);
                                  break;
                                default:
                                  tl_lidx = 0; tl_i1 = tl_j1 = tl_k1 = 0; /* unreachable */
                                }

                              /* Read neighbor state from L1-cached local buffer.
                                 tl_local_fmax > 0 iff a collapsed particle exists at
                                 that position (fill_local_buffer stores 0 when fp<0). */
                              int tl_ngid         = tl_local_gid [tl_lidx];
                              PRODFLOAT tl_nfmax  = tl_local_fmax[tl_lidx];
                              tl_neigh[tl_nn] = tl_ngid;
                              if (tl_nfmax > (PRODFLOAT)0)
                                tl_peak_cond &= (cur_tv_g->Frag[tl_pi].Fmax > tl_nfmax);

                              if (tl_ngid == FILAMENT)
                                {
                                  tl_neigh[tl_nn] = 0;
                                  /* For filament accretion we need the frag-order index */
                                  tl_pos = find_location(tl_i1, tl_j1, tl_k1);
                                  tl_fil_list[tl_nf][0] = tl_i1;
                                  tl_fil_list[tl_nf][1] = tl_j1;
                                  tl_fil_list[tl_nf][2] = tl_k1;
                                  tl_fil_list[tl_nf][3] = tl_pos;
                                  tl_nf++;
                                }
                            } /* end 6-neighbor loop */

                          /* Remove duplicates from neighbor list */
                          clean_list(tl_neigh);

                          for (tl_nn = tl_neigrp = 0; tl_nn < NV; tl_nn++)
                            if (tl_neigh[tl_nn] > FILAMENT) tl_neigrp++;

                          if (tl_neigrp > 0 && good_particle)
                            tl_cnt[tl_neigrp]++;
                        }
                      else
                        {
                          tl_peak_cond = 0;
                          tl_neigrp    = 0;
                        }

                      /* === CASE 1: PEAK === */
                      if (tl_peak_cond)
                        {
                          /* Shorthand: Fmax from the L2-resident Frag[] cache */
                          PRODFLOAT tl_Fmax_pi = cur_tv_g->Frag[tl_pi].Fmax;

                          if (good_particle) tl_cnt[0]++;

                          /* Lock-free peak allocation via atomic capture.
                             LOCK XADD on x86 / LDADD on ARM — zero wait.    */
#pragma omp atomic capture
                          tl_my_group = ++ngroups;

                          if (tl_my_group > Npeaks + 2)
                            {
#pragma omp atomic write
                              par_error = 1;
                            }
                          if (par_error) break;

                          /* Fase A: create a thread-local cached copy of the new
                             group's data fields (Vel, Pos, Mass).  Subsequent
                             set_obj/set_group calls for this group will be served
                             from LocalGroups[] (L2 cache) rather than global
                             groups[] (DRAM).
                             Structural fields (point, bottom, ll, …) are written
                             directly to global groups[] — these are accessed by
                             accretion/merge_groups via global index, and are only
                             written once at creation time.                         */
                          {
                            int _li = cur_tv_g->n_local++;
                            cur_tv_g->local_gids[_li] = tl_my_group;
                            group_data *_lg = &cur_tv_g->LocalGroups[_li];

                            /* Data fields — read from Frag[] (L2, eliminates DRAM) */
                            _lg->t_peak   = tl_Fmax_pi;
                            _lg->t_appear = -1;
                            _lg->t_merge  = -1;
                            _lg->Pos[0]   = tl_ibox + SHIFT;
                            _lg->Pos[1]   = tl_jbox + SHIFT;
                            _lg->Pos[2]   = tl_kbox + SHIFT;
                            _lg->Vel[0]   = cur_tv_g->Frag[tl_pi].Vel[0];
                            _lg->Vel[1]   = cur_tv_g->Frag[tl_pi].Vel[1];
                            _lg->Vel[2]   = cur_tv_g->Frag[tl_pi].Vel[2];
#ifdef TWO_LPT
                            _lg->Vel_2LPT[0] = cur_tv_g->Frag[tl_pi].Vel_2LPT[0];
                            _lg->Vel_2LPT[1] = cur_tv_g->Frag[tl_pi].Vel_2LPT[1];
                            _lg->Vel_2LPT[2] = cur_tv_g->Frag[tl_pi].Vel_2LPT[2];
#ifdef THREE_LPT
                            _lg->Vel_3LPT_1[0] = cur_tv_g->Frag[tl_pi].Vel_3LPT_1[0];
                            _lg->Vel_3LPT_1[1] = cur_tv_g->Frag[tl_pi].Vel_3LPT_1[1];
                            _lg->Vel_3LPT_1[2] = cur_tv_g->Frag[tl_pi].Vel_3LPT_1[2];
                            _lg->Vel_3LPT_2[0] = cur_tv_g->Frag[tl_pi].Vel_3LPT_2[0];
                            _lg->Vel_3LPT_2[1] = cur_tv_g->Frag[tl_pi].Vel_3LPT_2[1];
                            _lg->Vel_3LPT_2[2] = cur_tv_g->Frag[tl_pi].Vel_3LPT_2[2];
#endif /* THREE_LPT */
#endif /* TWO_LPT */
#ifdef RECOMPUTE_DISPLACEMENTS
                            _lg->Vel_prev[0] = cur_tv_g->Frag[tl_pi].Vel_prev[0];
                            _lg->Vel_prev[1] = cur_tv_g->Frag[tl_pi].Vel_prev[1];
                            _lg->Vel_prev[2] = cur_tv_g->Frag[tl_pi].Vel_prev[2];
#ifdef TWO_LPT
                            _lg->Vel_2LPT_prev[0] = cur_tv_g->Frag[tl_pi].Vel_2LPT_prev[0];
                            _lg->Vel_2LPT_prev[1] = cur_tv_g->Frag[tl_pi].Vel_2LPT_prev[1];
                            _lg->Vel_2LPT_prev[2] = cur_tv_g->Frag[tl_pi].Vel_2LPT_prev[2];
#ifdef THREE_LPT
                            _lg->Vel_3LPT_1_prev[0] = cur_tv_g->Frag[tl_pi].Vel_3LPT_1_prev[0];
                            _lg->Vel_3LPT_1_prev[1] = cur_tv_g->Frag[tl_pi].Vel_3LPT_1_prev[1];
                            _lg->Vel_3LPT_1_prev[2] = cur_tv_g->Frag[tl_pi].Vel_3LPT_1_prev[2];
                            _lg->Vel_3LPT_2_prev[0] = cur_tv_g->Frag[tl_pi].Vel_3LPT_2_prev[0];
                            _lg->Vel_3LPT_2_prev[1] = cur_tv_g->Frag[tl_pi].Vel_3LPT_2_prev[1];
                            _lg->Vel_3LPT_2_prev[2] = cur_tv_g->Frag[tl_pi].Vel_3LPT_2_prev[2];
#endif /* THREE_LPT */
#endif /* TWO_LPT */
#endif /* RECOMPUTE_DISPLACEMENTS */
                            _lg->Mass = 1;
                            _lg->name = particle_name;
                            _lg->good = good_particle;
                          }

                          /* Structural fields — global groups[] with global indices.
                             Vel/Pos NOT written here; they live in LocalGroups until
                             tile_vol_flush() at tile end.                           */
                          groups[tl_my_group].t_peak   = cur_tv_g->Frag[tl_pi].Fmax;
                          groups[tl_my_group].t_appear = -1;
                          groups[tl_my_group].t_merge  = -1;
                          groups[tl_my_group].Mass     = 1;
                          groups[tl_my_group].name     = particle_name;
                          groups[tl_my_group].good     = good_particle;
                          groups[tl_my_group].point    = tl_iz;
                          groups[tl_my_group].bottom   = tl_iz;
                          groups[tl_my_group].ll       = tl_my_group;
                          groups[tl_my_group].halo_app = tl_my_group;
#ifdef PLC
                          if (tl_Fmax_pi > plc.Fstart)
                            groups[tl_my_group].Flast = plc.Fstart;
                          else
                            groups[tl_my_group].Flast = tl_Fmax_pi;
#endif
                          group_ID[tl_iz]     = tl_my_group;
                          linking_list[tl_iz] = tl_iz;

                          if (params.MinHaloMass == 1)
                            {
                              groups[tl_my_group].t_appear = tl_Fmax_pi;
                              /* keep LocalGroups in sync */
                              cur_tv_g->LocalGroups[cur_tv_g->n_local - 1].t_appear = tl_Fmax_pi;
#ifdef SNAPSHOT
                              /* zacc is written to global frag[]; Fmax read from Frag[] */
                              frag[tl_iz].zacc = tl_Fmax_pi - 1;
#endif
                            }
                        }

                      /* === CASE 2: SINGLE NEIGHBOUR GROUP === */
                      else if (tl_neigrp == 1)
                        {
                          condition_for_accretion(1, tl_ibox, tl_jbox, tl_kbox,
                                                  tl_iz, cur_tv_g->Frag[tl_pi].Fmax,
                                                  tl_neigh[0], &tl_d2, &tl_r2);
                          if (tl_d2 < tl_r2)
                            {
                              if (good_particle) tl_cnt[7]++;
                              tl_accrflag  = 1;
                              tl_to_group  = tl_neigh[0];
                              accretion(tl_to_group, tl_ibox, tl_jbox, tl_kbox,
                                        tl_iz, cur_tv_g->Frag[tl_pi].Fmax);
                            }
                          else
                            {
                              if (good_particle) tl_cnt[12]++;
                              tl_filament_delta++;
                              group_ID[tl_iz]     = FILAMENT;
                              linking_list[tl_iz] = tl_iz;
                            }
                        }

                      /* === CASE 3: MULTIPLE NEIGHBOUR GROUPS === */
                      else if (tl_neigrp > 1)
                        {
                          /* Try to accrete onto closest group */
                          tl_best_ratio = pow(10.0 * subbox.Lgwbl[_x_], 2.0);
                          tl_accgrp = -1;
                          for (tl_ig1 = 0; tl_ig1 < tl_neigrp; tl_ig1++)
                            {
                              condition_for_accretion(2, tl_ibox, tl_jbox, tl_kbox,
                                                      tl_iz, cur_tv_g->Frag[tl_pi].Fmax,
                                                      tl_neigh[tl_ig1],
                                                      &tl_d2, &tl_r2);
                              tl_ratio = tl_d2 / tl_r2;
                              if (tl_ratio < 1.0 && tl_ratio < tl_best_ratio)
                                {
                                  tl_best_ratio = tl_ratio;
                                  tl_accgrp     = tl_ig1;
                                }
                            }

                          if (tl_accgrp >= 0)
                            {
                              if (good_particle)
                                {
                                  tl_cnt[7]++;
                                  tl_cnt[8]++;
                                }
                              tl_accrflag = 1;
                              tl_to_group = tl_neigh[tl_accgrp];
                              accretion(tl_neigh[tl_accgrp],
                                        tl_ibox, tl_jbox, tl_kbox,
                                        tl_iz, cur_tv_g->Frag[tl_pi].Fmax);
                            }

                          /* Check pairwise merging */
                          tl_nmerge = 0;
                          for (tl_ig1 = 0; tl_ig1 < tl_neigrp; tl_ig1++)
                            for (tl_ig2 = 0; tl_ig2 < tl_ig1; tl_ig2++)
                              {
                                tl_merge_arr[tl_ig1][tl_ig2] = 0;
                                condition_for_merging(cur_tv_g->Frag[tl_pi].Fmax,
                                                      tl_neigh[tl_ig1],
                                                      tl_neigh[tl_ig2],
                                                      &tl_merge_flag);
                                if (tl_merge_flag)
                                  {
                                    tl_merge_arr[tl_ig1][tl_ig2] = 1;
                                    tl_nmerge++;
                                  }
                              }

                          if (tl_nmerge > 0)
                            {
                              for (tl_ig1 = 0; tl_ig1 < tl_neigrp; tl_ig1++)
                                for (tl_ig2 = 0; tl_ig2 < tl_ig1; tl_ig2++)
                                  if (tl_merge_arr[tl_ig1][tl_ig2] == 1 &&
                                      tl_neigh[tl_ig1] != tl_neigh[tl_ig2])
                                    {
                                      if (good_particle) tl_cnt[10]++;
                                      if (groups[tl_neigh[tl_ig1]].Mass >
                                          groups[tl_neigh[tl_ig2]].Mass)
                                        {
                                          merge_groups(tl_neigh[tl_ig1],
                                                       tl_neigh[tl_ig2],
                                                       cur_tv_g->Frag[tl_pi].Fmax);
                                          tl_large = tl_neigh[tl_ig1];
                                          tl_small = tl_neigh[tl_ig2];
                                        }
                                      else
                                        {
                                          merge_groups(tl_neigh[tl_ig2],
                                                       tl_neigh[tl_ig1],
                                                       cur_tv_g->Frag[tl_pi].Fmax);
                                          tl_small = tl_neigh[tl_ig1];
                                          tl_large = tl_neigh[tl_ig2];
                                        }
                                      if (tl_to_group == tl_small)
                                        tl_to_group = tl_large;
                                      for (tl_ig3 = 0; tl_ig3 < tl_neigrp; tl_ig3++)
                                        if (tl_neigh[tl_ig3] == tl_small)
                                          tl_neigh[tl_ig3] = tl_large;
                                      /* Sync local buffer: replace absorbed grp with
                                         surviving grp so future tile particles don't
                                         see stale (invalidated) group IDs.           */
                                      {
                                        int tl_bk;
                                        for (tl_bk = 0; tl_bk < LS3; tl_bk++)
                                          if (tl_local_gid[tl_bk] == tl_small)
                                            tl_local_gid[tl_bk] = tl_large;
                                      }
                                      if (groups[tl_large].Mass <
                                          5 * groups[tl_small].Mass && good_particle)
                                        tl_cnt[11]++;
                                    }
                            }

                          /* Retry accretion if not yet accreted */
                          if (tl_accgrp == -1)
                            {
                              clean_list(tl_neigh);
                              for (tl_nn = tl_neigrp = 0; tl_nn < NV; tl_nn++)
                                if (tl_neigh[tl_nn] > FILAMENT) tl_neigrp++;

                              tl_best_ratio = pow(10.0 * subbox.Lgwbl[_x_], 2.0);
                              tl_accgrp = -1;
                              for (tl_ig1 = 0; tl_ig1 < tl_neigrp; tl_ig1++)
                                {
                                  condition_for_accretion(3, tl_ibox, tl_jbox,
                                                          tl_kbox, tl_iz,
                                                          cur_tv_g->Frag[tl_pi].Fmax,
                                                          tl_neigh[tl_ig1],
                                                          &tl_d2, &tl_r2);
                                  tl_ratio = tl_d2 / tl_r2;
                                  if (tl_ratio < tl_best_ratio)
                                    {
                                      tl_best_ratio = tl_ratio;
                                      tl_accgrp     = tl_ig1;
                                    }
                                }

                              if (tl_best_ratio < 1.0)
                                {
                                  if (good_particle)
                                    {
                                      tl_cnt[7]++;
                                      tl_cnt[9]++;
                                    }
                                  tl_accrflag = 1;
                                  tl_to_group = tl_neigh[tl_accgrp];
                                  accretion(tl_neigh[tl_accgrp],
                                            tl_ibox, tl_jbox, tl_kbox,
                                            tl_iz, cur_tv_g->Frag[tl_pi].Fmax);
                                }
                              else
                                {
                                  if (good_particle) tl_cnt[12]++;
                                  tl_filament_delta++;
                                  group_ID[tl_iz]     = FILAMENT;
                                  linking_list[tl_iz] = tl_iz;
                                }
                            }
                        }

                      /* === CASE 4: FILAMENT === */
                      else
                        {
                          if (good_particle) tl_cnt[12]++;
                          tl_filament_delta++;
                          group_ID[tl_iz]     = FILAMENT;
                          linking_list[tl_iz] = tl_iz;
                        }

                      /* === FILAMENT ACCRETION POST-CHECK === */
                      if (tl_accrflag && tl_nf && !tl_skip)
                        {
                          /* First pass: tag filaments that qualify */
                          for (tl_ifil = 0; tl_ifil < tl_nf; tl_ifil++)
                            {
                              condition_for_accretion(4,
                                tl_fil_list[tl_ifil][0],
                                tl_fil_list[tl_ifil][1],
                                tl_fil_list[tl_ifil][2],
                                tl_fil_list[tl_ifil][3],
                                cur_tv_g->Frag[tl_pi].Fmax, tl_to_group,
                                &tl_d2, &tl_r2);
                              if (tl_d2 < tl_r2)
                                tl_fil_list[tl_ifil][3] *= -1; /* tag */
                            }
                          /* Second pass: accrete tagged filaments */
                          for (tl_ifil = 0; tl_ifil < tl_nf; tl_ifil++)
                            if (tl_fil_list[tl_ifil][3] < 0)
                              {
                                tl_fil_list[tl_ifil][3] *= -1; /* untag */
                                accretion(tl_to_group,
                                          tl_fil_list[tl_ifil][0],
                                          tl_fil_list[tl_ifil][1],
                                          tl_fil_list[tl_ifil][2],
                                          tl_fil_list[tl_ifil][3],
                                          cur_tv_g->Frag[tl_pi].Fmax);
                                /* Update local buffer: filament particle now belongs to tl_to_group */
                                {
                                  int tl_fli = tl_fil_list[tl_ifil][0] - (tile_sx - 1);
                                  int tl_flj = tl_fil_list[tl_ifil][1] - (tile_sy - 1);
                                  int tl_flk = tl_fil_list[tl_ifil][2] - (tile_sz - 1);
                                  if (tl_fli >= 0 && tl_fli < LS &&
                                      tl_flj >= 0 && tl_flj < LS &&
                                      tl_flk >= 0 && tl_flk < LS)
                                    tl_local_gid[LIDX(tl_fli, tl_flj, tl_flk)] = tl_to_group;
                                }
                                tl_filament_delta--;

                                if (tl_fil_list[tl_ifil][0] >= subbox.safe[_x_] &&
                                    tl_fil_list[tl_ifil][0] <
                                      subbox.Lgwbl[_x_] - subbox.safe[_x_] &&
                                    tl_fil_list[tl_ifil][1] >= subbox.safe[_y_] &&
                                    tl_fil_list[tl_ifil][1] <
                                      subbox.Lgwbl[_y_] - subbox.safe[_y_] &&
                                    tl_fil_list[tl_ifil][2] >= subbox.safe[_z_] &&
                                    tl_fil_list[tl_ifil][2] <
                                      subbox.Lgwbl[_z_] - subbox.safe[_z_])
                                  {
                                    tl_cnt[7]++;
                                    tl_cnt[13]++;
                                    tl_cnt[12]--;
                                  }
                              }
                        } /* end filament accretion */

                      /* Update local buffer with this particle's final group_ID
                         so subsequent particles in the same tile see it.        */
                      tl_local_gid[LIDX(tl_li_own, tl_lj_own, tl_lk_own)] =
                        group_ID[tl_iz];

                    } /* end particle loop within tile */

                  /* Fase A: flush LocalGroups Vel/Pos back to global groups[] and
                     clear the per-thread pointer for safety.                     */
                  tile_vol_flush(cur_tv_g);
                  cur_tv_g = NULL;

                } /* end omp for over tiles */

              /* Flush thread-local FILAMENT.Mass delta — one atomic per thread
                 per color pass instead of one per filament event.              */
#pragma omp atomic
              groups[FILAMENT].Mass += tl_filament_delta;
              tl_filament_delta = 0;

              /* Reduce thread-local counters into global after each color pass */
#pragma omp critical(counters_reduce)
              {
                int tl_ci2;
                for (tl_ci2 = 0; tl_ci2 < NCOUNTERS; tl_ci2++)
                  counters[tl_ci2] += tl_cnt[tl_ci2];
              }
            } /* end omp parallel */

            if (par_error)
              {
                printf("OH MY DEAR, TASK %d FOUND TOO MANY GROUPS, THIS SHOULD NOT HAPPEN!\n",
                       ThisTask);
                tile_free();
                return 1;
              }

          } /* end color loop (0..7) */

        /* ---- PLC post-pass: serial, using correct group_ID[] from tiling ---- */
#ifdef PLC
        {
          /* Re-run PLC logic serially over all particles in the epoch.
             group_ID[] is now fully populated so PLC checks are exact.  */
          int tl_tz;
          for (tl_tz = last_z; tl_tz <= to_z_ep; tl_tz++)
            {
              int tl_iz2 = tl_tz;
              int tl_ibox2, tl_jbox2, tl_kbox2;
              INDEX_TO_COORD(frag_pos[tl_iz2], tl_ibox2, tl_jbox2, tl_kbox2,
                             subbox.Lgwbl);

              int tl_skip2 = 0;
              if (!subbox.pbc[_x_] &&
                  (tl_ibox2 == 0 || tl_ibox2 == subbox.Lgwbl[_x_]-1)) ++tl_skip2;
              if (!subbox.pbc[_y_] &&
                  (tl_jbox2 == 0 || tl_jbox2 == subbox.Lgwbl[_y_]-1)) ++tl_skip2;
              if (!subbox.pbc[_z_] &&
                  (tl_kbox2 == 0 || tl_kbox2 == subbox.Lgwbl[_z_]-1)) ++tl_skip2;

              if (tl_skip2) continue;

              /* Rebuild neighbor list to get neigrp and neigh[] */
              int tl_neigh2[NV];
              int tl_neigrp2 = 0;
              int tl_nn2;
              for (tl_nn2 = 0; tl_nn2 < NV; tl_nn2++) tl_neigh2[tl_nn2] = 0;

              int tl_i1p, tl_j1p, tl_k1p;
              for (tl_nn2 = 0; tl_nn2 < NV; tl_nn2++)
                {
                  switch (tl_nn2)
                    {
                    case 0:
                      tl_i1p = (subbox.pbc[_x_] && tl_ibox2 == 0 ?
                                subbox.Lgwbl[_x_]-1 : tl_ibox2-1);
                      tl_j1p = tl_jbox2; tl_k1p = tl_kbox2; break;
                    case 1:
                      tl_i1p = (subbox.pbc[_x_] &&
                                tl_ibox2 == subbox.Lgwbl[_x_]-1 ?
                                0 : tl_ibox2+1);
                      tl_j1p = tl_jbox2; tl_k1p = tl_kbox2; break;
                    case 2:
                      tl_i1p = tl_ibox2;
                      tl_j1p = (subbox.pbc[_y_] && tl_jbox2 == 0 ?
                                subbox.Lgwbl[_y_]-1 : tl_jbox2-1);
                      tl_k1p = tl_kbox2; break;
                    case 3:
                      tl_i1p = tl_ibox2;
                      tl_j1p = (subbox.pbc[_y_] &&
                                tl_jbox2 == subbox.Lgwbl[_y_]-1 ?
                                0 : tl_jbox2+1);
                      tl_k1p = tl_kbox2; break;
                    case 4:
                      tl_i1p = tl_ibox2; tl_j1p = tl_jbox2;
                      tl_k1p = (subbox.pbc[_z_] && tl_kbox2 == 0 ?
                                subbox.Lgwbl[_z_]-1 : tl_kbox2-1); break;
                    case 5:
                      tl_i1p = tl_ibox2; tl_j1p = tl_jbox2;
                      tl_k1p = (subbox.pbc[_z_] &&
                                tl_kbox2 == subbox.Lgwbl[_z_]-1 ?
                                0 : tl_kbox2+1); break;
                    default:
                      tl_i1p = tl_j1p = tl_k1p = 0;
                    }
                  int tl_pos2 = find_location(tl_i1p, tl_j1p, tl_k1p);
                  if (tl_pos2 >= 0 && group_ID[tl_pos2] > FILAMENT)
                    tl_neigh2[tl_nn2] = group_ID[tl_pos2];
                }
              clean_list(tl_neigh2);
              for (tl_nn2 = 0; tl_nn2 < NV; tl_nn2++)
                if (tl_neigh2[tl_nn2] > FILAMENT) tl_neigrp2++;

              /* Periodic PLC sync check */
              if (plc_started &&
                  frag[tl_iz2].Fmax < NextF_PLC &&
                  frag[tl_iz2].Fmax >= plc.Fstop)
                {
                  int mysave2, save2;
                  mysave2 = (plc.Nmax - plc.Nstored <
                             SAFEPLC * (plc.Nstored - plc.Nstored_last));
                  MPI_Reduce(&mysave2, &save2, 1, MPI_INT, MPI_SUM, 0,
                             MPI_COMM_WORLD);
                  MPI_Bcast(&save2, 1, MPI_INT, 0, MPI_COMM_WORLD);
                  if (save2)
                    {
                      if (write_PLC(0)) { tile_free(); return 1; }
                      plc.Nstored = 0;
                    }
                  NextF_PLC *= DeltaF_PLC;
                  plc.Nstored_last = plc.Nstored;
                }

              /* PLC crossing check for each neighbouring group */
              if (frag[tl_iz2].Fmax < plc.Fstart &&
                  frag[tl_iz2].Fmax >= plc.Fstop)
                {
                  if (!plc_started)
                    {
                      plc_started = 1;
                      if (!ThisTask)
                        printf("[%s] Starting PLC reconstruction\n", fdate());
                      cputmp = MPI_Wtime();
                    }
                  int tl_storex = subbox.pbc[_x_];
                  int tl_storey = subbox.pbc[_y_];
                  int tl_storez = subbox.pbc[_z_];
                  subbox.pbc[_x_] = subbox.pbc[_y_] = subbox.pbc[_z_] = 0;

                  int tl_ig1p;
                  for (tl_ig1p = 0; tl_ig1p < tl_neigrp2; tl_ig1p++)
                    {
                      if (tl_neigh2[tl_ig1p] > FILAMENT &&
                          groups[tl_neigh2[tl_ig1p]].good &&
                          groups[tl_neigh2[tl_ig1p]].Mass >= params.MinHaloMass)
                        {
                          int tl_irep;
                          thisgroup = tl_neigh2[tl_ig1p];
                          for (tl_irep = 0; tl_irep < plc.Nreplications; tl_irep++)
                            if (!(frag[tl_iz2].Fmax > plc.repls[tl_irep].F1 ||
                                  groups[thisgroup].Flast < plc.repls[tl_irep].F2))
                              {
                                double tl_aa, tl_bb, tl_Fplc;
                                replicate[0] = plc.repls[tl_irep].i;
                                replicate[1] = plc.repls[tl_irep].j;
                                replicate[2] = plc.repls[tl_irep].k;
                                tl_bb = condition_PLC(frag[tl_iz2].Fmax);
                                if (tl_bb == 0.0)
                                  {
                                    if (store_PLC(frag[tl_iz2].Fmax))
                                      { tile_free(); return 1; }
                                  }
                                else if (tl_bb > 0.0)
                                  {
                                    tl_aa = condition_PLC(groups[thisgroup].Flast);
                                    if (tl_aa < 0.0)
                                      {
                                        tl_Fplc = find_brent(groups[thisgroup].Flast,
                                                             frag[tl_iz2].Fmax);
                                        if (tl_Fplc == -99.0)
                                          { tile_free(); return 1; }
                                        if (store_PLC(tl_Fplc))
                                          { tile_free(); return 1; }
                                      }
                                  }
                              }
                          groups[tl_neigh2[tl_ig1p]].Flast = frag[tl_iz2].Fmax;
                        }
                    }
                  subbox.pbc[_x_] = tl_storex;
                  subbox.pbc[_y_] = tl_storey;
                  subbox.pbc[_z_] = tl_storez;
                }
              else if (plc.Fstart > 0.0 && frag[tl_iz2].Fmax < plc.Fstart)
                {
                  int tl_ig1p;
                  for (tl_ig1p = 0; tl_ig1p < tl_neigrp2; tl_ig1p++)
                    if (tl_neigh2[tl_ig1p] > FILAMENT)
                      groups[tl_neigh2[tl_ig1p]].Flast = frag[tl_iz2].Fmax;
                }

              /* PLC final check at end of epoch or stop condition */
              if (plc.Fstart > 0 && !last_check_done &&
                  (tl_tz == to_z_ep || frag[tl_iz2].Fmax < plc.Fstop))
                {
                  if (write_PLC(0)) { tile_free(); return 1; }
                  plc.Nstored = 0;
                  int tl_storex = subbox.pbc[_x_];
                  int tl_storey = subbox.pbc[_y_];
                  int tl_storez = subbox.pbc[_z_];
                  subbox.pbc[_x_] = subbox.pbc[_y_] = subbox.pbc[_z_] = 0;
                  last_check_done = 1;
                  int tl_g;
                  for (tl_g = FILAMENT+1; tl_g <= ngroups; tl_g++)
                    {
                      if (groups[tl_g].point >= 0 && groups[tl_g].good &&
                          groups[tl_g].Mass >= params.MinHaloMass)
                        {
                          int tl_irep;
                          thisgroup = tl_g;
                          for (tl_irep = 0; tl_irep < plc.Nreplications; tl_irep++)
                            if (groups[tl_g].Flast > plc.repls[tl_irep].F2)
                              {
                                double tl_aa, tl_bb, tl_Fplc;
                                replicate[0] = plc.repls[tl_irep].i;
                                replicate[1] = plc.repls[tl_irep].j;
                                replicate[2] = plc.repls[tl_irep].k;
                                tl_bb = condition_PLC(plc.Fstop);
                                if (tl_bb == 0.0)
                                  {
                                    if (store_PLC(plc.Fstop))
                                      { tile_free(); return 1; }
                                  }
                                else if (tl_bb > 0.0)
                                  {
                                    tl_aa = condition_PLC(groups[tl_g].Flast);
                                    if (tl_aa < 0.0)
                                      {
                                        tl_Fplc = find_brent(groups[tl_g].Flast,
                                                             plc.Fstop);
                                        if (tl_Fplc == -99.0)
                                          { tile_free(); return 1; }
                                        if (store_PLC(tl_Fplc))
                                          { tile_free(); return 1; }
                                      }
                                  }
                              }
                        }
                    }
                  subbox.pbc[_x_] = tl_storex;
                  subbox.pbc[_y_] = tl_storey;
                  subbox.pbc[_z_] = tl_storez;
                  if (!ThisTask)
                    printf("[%s] PLC: Last check done, Task 0 stored %d halos (max:%d)\n",
                           fdate(), plc.Nstored, plc.Nmax);
                  cputime.plc += MPI_Wtime() - cputmp;
                  if (write_PLC(1)) { tile_free(); return 1; }
                  break; /* stop PLC post-pass early */
                }
            } /* end PLC post-pass loop */
        }
#endif /* PLC */

        /* ---- Progress print ---- */
        if (!ThisTask)
          printf("[%s] *** %3d%% done via tiling, F = %6.2f,  z = %6.2f\n",
                 fdate(), 100, frag[to_z_ep].Fmax, frag[to_z_ep].Fmax - 1.0);

        /* ---- Write pending outputs (serial, after all tiling) ---- */
        {
          double tl_cputmp;
          int tl_this_last = (to_z_ep == nstep - 1) ? 1 : 0;
          while (iout < outputs.n &&
                 (tl_this_last || frag[to_z_ep].Fmax < outputs.F[iout]))
            {
              tl_cputmp = MPI_Wtime();
              if (!ThisTask)
                printf("[%s] Writing output at z=%f\n", fdate(),
                       outputs.z[iout]);
              fflush(stdout);
              MPI_Barrier(MPI_COMM_WORLD);
              if (write_catalog(iout))   { tile_free(); return 1; }
              if (compute_mf(iout))      { tile_free(); return 1; }
              if (iout == outputs.n - 1)
                {
                  if (write_histories()) { tile_free(); return 1; }
                }
              cputime.io += MPI_Wtime() - tl_cputmp;
              iout++;
              if (tl_this_last) break;
            }
        }

        tile_free();

        /* ---- Handle pause / completion ---- */
        if (to_z_ep < nstep - 1)
          {
            if (!ThisTask)
              printf("[%s] Pausing fragmentation process\n", fdate());
            last_z = to_z_ep + 1;
            return 0;
          }
        last_z = nstep; /* mark done */

      } /* end if (to_z_ep >= last_z) */

    goto build_groups_statistics;
  }
#endif /* defined(_OPENMP) && !defined(CLASSIC_FRAGMENTATION) */


  /* ================================================================
     SERIAL PATH (unchanged from original)
     Active when: _OPENMP not defined, OR CLASSIC_FRAGMENTATION set.
     ================================================================ */
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
#ifdef _OPENMP
#pragma omp parallel for reduction(+:counters[14])
#endif
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
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

  /* Fase A: route through local cache when inside a tile (get_group_ptr
     returns &LocalGroups[li] for local groups, &groups[grp] otherwise). */
  group_data *g = get_group_ptr(grp);

  myobj->M=g->Mass;
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

  myobj->M=g->Mass;
  for (int i=0;i<3;i++)
    {
      myobj->q[i]=g->Pos[i];
      myobj->v[i]=g->Vel[i];
#ifdef TWO_LPT
      myobj->v2[i]=g->Vel_2LPT[i];
#ifdef THREE_LPT
      myobj->v31[i]=g->Vel_3LPT_1[i];
      myobj->v32[i]=g->Vel_3LPT_2[i];
#endif
#endif

#ifdef RECOMPUTE_DISPLACEMENTS
      myobj->v_prev[i]=g->Vel_prev[i];
#ifdef TWO_LPT
      myobj->v2_prev[i]=g->Vel_2LPT_prev[i];
#ifdef THREE_LPT
      myobj->v31_prev[i]=g->Vel_3LPT_1_prev[i];
      myobj->v32_prev[i]=g->Vel_3LPT_2_prev[i];
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

  /* Fase A: route through local cache when inside a tile.  Mass is also
     written back to the global groups[] entry immediately so that callers
     of accretion/merge_groups that read groups[grp].Mass directly (e.g.
     condition_for_accretion, condition_for_merging) always see the
     up-to-date value without waiting for tile_vol_flush().              */
  group_data *g = get_group_ptr(grp);

  g->Mass=myobj->M;
  if (g != &groups[grp])
    groups[grp].Mass = myobj->M;   /* keep global Mass in sync */

  for (int i=0;i<3;i++)
    {
      g->Pos[i]=myobj->q[i];
      g->Vel[i]=myobj->v[i];
#ifdef TWO_LPT
      g->Vel_2LPT[i]=myobj->v2[i];
#ifdef THREE_LPT
      g->Vel_3LPT_1[i]=myobj->v31[i];
      g->Vel_3LPT_2[i]=myobj->v32[i];
#endif
#endif

#ifdef RECOMPUTE_DISPLACEMENTS
      g->Vel_prev[i]=myobj->v_prev[i];
#ifdef TWO_LPT
      g->Vel_2LPT_prev[i]=myobj->v2_prev[i];
#ifdef THREE_LPT
      g->Vel_3LPT_1_prev[i]=myobj->v31_prev[i];
      g->Vel_3LPT_2_prev[i]=myobj->v32_prev[i];
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

