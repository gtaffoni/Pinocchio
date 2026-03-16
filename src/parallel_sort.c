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
   Parallel radix sort for PINOCCHIO fragmentation.

   Replaces qsort with a cache-friendly, O(N) radix sort that can be
   parallelized with OpenMP. For 100M+ particles this gives ~5-15x
   speedup over the standard qsort with indirect comparisons.

   Algorithm: LSB radix sort with 8-bit radix (256 buckets), 4 passes
   for 32-bit keys. Each pass consists of:
   1. Per-thread histogram (parallel)
   2. Global prefix sum (serial, negligible cost: P x 256 entries)
   3. Per-thread scatter (parallel)

   Double-buffering is used: input and output swap at each pass.
   With 4 passes (even), the result ends up in the original arrays.
*/

#include "pinocchio.h"
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define RADIX_BITS  8
#define RADIX_SIZE  (1 << RADIX_BITS)   /* 256 */
#define RADIX_MASK  (RADIX_SIZE - 1)    /* 0xFF */
#define NUM_PASSES  4                   /* 4 passes for 32-bit keys */

/* Maximum number of OpenMP threads for histogram arrays.
   Stack usage: MAX_SORT_THREADS x 256 x sizeof(int) x 2 arrays.
   With 256 threads: 256 x 256 x 4 x 2 = 512 KB (heap-allocated). */
#define MAX_SORT_THREADS 256


/* Convert a PRODFLOAT Fmax value to a 32-bit unsigned int that sorts
   in DESCENDING order of the original float value.

   IEEE 754 floats can be compared as unsigned integers after:
   - positive floats: flip the sign bit (0x80000000)
   - negative floats: flip all bits
   Then bitwise NOT (~) gives descending order.

   We truncate double to float for the sort key since Fmax values
   (collapse redshifts, typically 1-100) have more than enough
   precision in 32 bits for correct ordering. */
static inline unsigned int fmax_to_sortkey_desc(PRODFLOAT f)
{
  float ff = (float)f;
  unsigned int u;
  memcpy(&u, &ff, sizeof(u));
  unsigned int mask = -(u >> 31) | 0x80000000u;
  return ~(u ^ mask);
}


/* Convert a position (int) to a 32-bit unsigned key for ascending sort.
   Positions are non-negative grid indices in PINOCCHIO, but we handle
   the general signed case by shifting to unsigned range. */
static inline unsigned int position_to_sortkey_asc(int pos)
{
  return (unsigned int)pos + 0x80000000u;
}


/* Core parallel radix sort on (key, index) pairs.

   Input:  keys[N], idx[N]
   Scratch: buf_keys[N], buf_idx[N]

   After sorting, the result is in keys[N], idx[N] (even number of
   passes means the data swaps back to the original arrays). */
static void radix_sort_core(
    unsigned int *keys,
    int *idx,
    unsigned int *buf_keys,
    int *buf_idx,
    int N)
{
  unsigned int *src_keys, *dst_keys;
  int *src_idx, *dst_idx;
  int pass;
  int nthreads;

#ifdef _OPENMP
  nthreads = omp_get_max_threads();
  if (nthreads > MAX_SORT_THREADS)
    nthreads = MAX_SORT_THREADS;
#else
  nthreads = 1;
#endif

  /* Per-thread histograms and scatter offsets (heap-allocated) */
  int *hist_flat = (int *)malloc((size_t)nthreads * RADIX_SIZE * sizeof(int));
  int *offs_flat = (int *)malloc((size_t)nthreads * RADIX_SIZE * sizeof(int));

  src_keys = keys;
  src_idx  = idx;
  dst_keys = buf_keys;
  dst_idx  = buf_idx;

  for (pass = 0; pass < NUM_PASSES; pass++)
    {
      int shift = pass * RADIX_BITS;
      int t, b;

      /* Phase 1: Per-thread histogram.
         Each thread counts occurrences of each byte value in its chunk. */
      memset(hist_flat, 0, (size_t)nthreads * RADIX_SIZE * sizeof(int));

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
      {
        int tid = omp_get_thread_num();
        int *my_hist = hist_flat + tid * RADIX_SIZE;

#pragma omp for schedule(static)
        for (int i = 0; i < N; i++)
          my_hist[(src_keys[i] >> shift) & RADIX_MASK]++;
      }
#else
      {
        int *my_hist = hist_flat;
        for (int i = 0; i < N; i++)
          my_hist[(src_keys[i] >> shift) & RADIX_MASK]++;
      }
#endif

      /* Phase 2: Global prefix sum.
         For each bucket b, iterate over threads in order. This ensures
         stability: within bucket b, thread 0's elements come first,
         then thread 1's, etc. */
      {
        int running_sum = 0;
        for (b = 0; b < RADIX_SIZE; b++)
          for (t = 0; t < nthreads; t++)
            {
              offs_flat[t * RADIX_SIZE + b] = running_sum;
              running_sum += hist_flat[t * RADIX_SIZE + b];
            }
      }

      /* Phase 3: Per-thread scatter.
         Each thread writes its elements to the destination using offsets.
         schedule(static) guarantees the same index-to-thread mapping
         as Phase 1, so each thread's offset counters are correct. */
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
      {
        int tid = omp_get_thread_num();
        int *my_offs = offs_flat + tid * RADIX_SIZE;

#pragma omp for schedule(static)
        for (int i = 0; i < N; i++)
          {
            int bucket = (src_keys[i] >> shift) & RADIX_MASK;
            int dest = my_offs[bucket]++;
            dst_keys[dest] = src_keys[i];
            dst_idx[dest]  = src_idx[i];
          }
      }
#else
      {
        int *my_offs = offs_flat;
        for (int i = 0; i < N; i++)
          {
            int bucket = (src_keys[i] >> shift) & RADIX_MASK;
            int dest = my_offs[bucket]++;
            dst_keys[dest] = src_keys[i];
            dst_idx[dest]  = src_idx[i];
          }
      }
#endif

      /* Swap source and destination for the next pass */
      {
        unsigned int *tmp_k = src_keys;
        int *tmp_i = src_idx;
        src_keys = dst_keys;
        src_idx  = dst_idx;
        dst_keys = tmp_k;
        dst_idx  = tmp_i;
      }
    }

  free(hist_flat);
  free(offs_flat);

  /* After 4 passes (even), the sorted result is back in (keys, idx). */
}


void parallel_radix_sort_by_fmax_desc(
    product_data *frag,
    int *indices,
    int *buf_indices,
    unsigned int *buf_keys,
    int *buf_idx,
    int N)
{
  /* Extract sort keys from frag[].Fmax into indices (reinterpreted
     as unsigned int), and initialize buf_idx as identity permutation.
     Then sort with double-buffering between the two pairs of arrays.

     After sorting:
     - keys end up in (uint*)indices (we don't need the keys)
     - sorted index permutation ends up in buf_idx

     We then copy the result to indices. */

  unsigned int *key_arr = (unsigned int *)indices;
  int i;

  /* Extract keys and initialize index array */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (i = 0; i < N; i++)
    {
      key_arr[i] = fmax_to_sortkey_desc(frag[i].Fmax);
      buf_idx[i] = i;
    }

  /* Radix sort: (key_arr, buf_idx) with scratch (buf_keys, buf_indices)
     After 4 passes, result is in (key_arr, buf_idx). */
  radix_sort_core(key_arr, buf_idx, buf_keys, buf_indices, N);

  /* Copy sorted indices to the output array.
     key_arr aliases indices, so we must copy from buf_idx. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (i = 0; i < N; i++)
    indices[i] = buf_idx[i];
}


void parallel_radix_sort_by_position_asc(
    int *positions,
    int *indices,
    int *buf_indices,
    unsigned int *buf_keys,
    int *buf_idx,
    int N)
{
  unsigned int *key_arr = (unsigned int *)indices;
  int i;

  /* Extract keys and initialize index array */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (i = 0; i < N; i++)
    {
      key_arr[i] = position_to_sortkey_asc(positions[i]);
      buf_idx[i] = i;
    }

  /* Radix sort */
  radix_sort_core(key_arr, buf_idx, buf_keys, buf_indices, N);

  /* Copy sorted indices to output */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (i = 0; i < N; i++)
    indices[i] = buf_idx[i];
}
