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

#ifndef PARALLEL_SORT_H
#define PARALLEL_SORT_H

/* Parallel radix sort for PINOCCHIO fragmentation.

   These functions sort an index array according to keys extracted
   from the data arrays. They use 8-bit radix (256 buckets), 4 passes
   for 32-bit keys. With OpenMP enabled, histogram and scatter phases
   are parallelized across threads.

   Buffer requirements (3 caller-supplied arrays of N ints/uints):
   - indices:     output sorted permutation [N]
   - buf_indices: scratch for indices during double-buffering [N]
   - buf_keys:    scratch for keys [N]

   The fourth scratch array (buf_idx) may be passed as NULL, in which case
   each function allocates a temporary buffer internally and frees it before
   returning.  This avoids any impact on the caller's memory budget (Nalloc).
*/

/* Sort indices by frag[].Fmax in descending order (identity input permutation).
   Tie-break: stable on input order (0,1,...,N-1), i.e. ascending array index.
   Use parallel_radix_sort_fmax_desc_stable_by_pos for the index_compare_F
   equivalent (tie-break by ascending Lagrangian position).
   buf_idx may be NULL (scratch allocated internally). */
void parallel_radix_sort_by_fmax_desc(
    product_data *frag,       /* data array (read-only, keys extracted from Fmax) */
    int *indices,             /* output: sorted permutation [N] */
    int *buf_indices,         /* scratch buffer during sort [N] */
    unsigned int *buf_keys,   /* scratch buffer for keys [N] */
    int *buf_idx,             /* scratch buffer for indices [N], or NULL */
    int N                     /* number of elements to sort */
);

/* Two-pass stable sort: Fmax descending, ties broken by positions[] ascending.
   Exactly equivalent to index_compare_F / qsort(index_compare_F).
   Pass 1: sort by positions[i] ascending.
   Pass 2: sort the Pass-1 permutation by frag[indices[j]].Fmax descending,
           stably, preserving pass-1 order for equal-Fmax elements.
   Because both passes are stable, the composition = Fmax desc + pos asc tie-break.
   WARNING: indices is used as scratch by pass 2 internally; it is repopulated
   with the final sorted permutation before the function returns.
   buf_idx may be NULL (scratch allocated internally). */
void parallel_radix_sort_fmax_desc_stable_by_pos(
    product_data *frag,       /* data array (Fmax keys read indirectly) */
    int *positions,           /* Lagrangian positions for tie-break [N] */
    int *indices,             /* output: sorted permutation [N] */
    int *buf_indices,         /* scratch buffer [N] */
    unsigned int *buf_keys,   /* scratch buffer for keys [N] */
    int *buf_idx,             /* scratch buffer for indices [N], or NULL */
    int N                     /* number of elements to sort */
);

/* Sort indices by integer positions in ascending order.
   After the call, indices[0] points to the particle with the smallest position.
   Positions must be unique (no ties); result is exactly index_compare_P.
   buf_idx may be NULL (scratch allocated internally). */
void parallel_radix_sort_by_position_asc(
    int *positions,           /* key array (e.g. frag_pos values) [N] */
    int *indices,             /* output: sorted permutation [N] */
    int *buf_indices,         /* scratch buffer during sort [N] */
    unsigned int *buf_keys,   /* scratch buffer for keys [N] */
    int *buf_idx,             /* scratch buffer for indices [N], or NULL */
    int N                     /* number of elements to sort */
);

#endif /* PARALLEL_SORT_H */
