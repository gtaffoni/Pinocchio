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

   Buffer requirements (4 arrays of N ints/uints total):
   - indices:     output sorted permutation [N]
   - buf_indices: scratch for indices during double-buffering [N]
   - buf_keys:    scratch for keys [N]
   - buf_idx:     scratch for indices [N] (e.g. group_ID before it is used)
*/

/* Sort indices by frag[].Fmax in descending order.
   After the call, indices[0] points to the particle with the highest Fmax. */
void parallel_radix_sort_by_fmax_desc(
    product_data *frag,       /* data array (read-only, keys extracted from Fmax) */
    int *indices,             /* output: sorted permutation [N] */
    int *buf_indices,         /* scratch buffer for keys during sort [N] */
    unsigned int *buf_keys,   /* scratch buffer for keys [N] */
    int *buf_idx,             /* scratch buffer for indices [N] */
    int N                     /* number of elements to sort */
);

/* Sort indices by integer positions in ascending order.
   After the call, indices[0] points to the particle with the smallest position. */
void parallel_radix_sort_by_position_asc(
    int *positions,           /* key array (e.g. frag_pos values) [N] */
    int *indices,             /* output: sorted permutation [N] */
    int *buf_indices,         /* scratch buffer for keys during sort [N] */
    unsigned int *buf_keys,   /* scratch buffer for keys [N] */
    int *buf_idx,             /* scratch buffer for indices [N] */
    int N                     /* number of elements to sort */
);

#endif /* PARALLEL_SORT_H */
