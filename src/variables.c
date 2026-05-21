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


#include "pinocchio.h"

int ThisTask,NTasks;

//int            pfft_flags_c2r, pfft_flags_r2c;
MPI_Comm        FFT_Comm;
internal_data   internal;

char *main_memory, *wheretoplace_mycat;
product_data *products, *frag;
unsigned int **seedtable;  // QUESTO RIMANE?
unsigned int   *cubes_ordering;
double **kdensity;
double **density;
double ***first_derivatives;
double ***second_derivatives;
#ifdef TWO_LPT
double *kvector_2LPT;
double *source_2LPT;
#ifdef THREE_LPT
double *kvector_3LPT_1,*kvector_3LPT_2;
double *source_3LPT_1,*source_3LPT_2;
#endif
#endif

double Rsmooth;
smoothing_data Smoothing;

grid_data *MyGrids;
int Ngrids;

/**
 * @brief FFT arrays and HeFFTe configuration (backend-dependent)
 *
 * When USE_HEFFTE is enabled:
 * - cvector_fft: Fourier-space array using struct my_double_complex (no pfft dependency)
 * - cvector_size: total complex elements in local Fourier-space grid
 * - inbox_low/high, outbox_low/high: pencil-slab boundaries for MPI redistribution
 * - options_fft: HeFFTe plan creation options
 *
 * When USE_HEFFTE is not enabled:
 * - cvector_fft: Fourier-space array using pfft_complex
 *
 * Both backends use the same rvector_fft for real-space data.
 *
 * @see pinocchio.h (declarations)
 * @see set_one_grid() (initialization in fmax-heffte.c)
 */
#ifdef USE_HEFFTE
struct my_double_complex **cvector_fft;      /**< Fourier-space arrays (HeFFTe backend) */
long int cvector_size;                       /**< Size of local Fourier-space grid per rank */
int inbox_low[3], inbox_high[3];             /**< Input pencil boundaries for FFT decomposition */
int outbox_low[3], outbox_high[3];           /**< Output pencil boundaries after transform */
heffte_plan_options options_fft;             /**< HeFFTe plan creation options */
#else
pfft_complex **cvector_fft;                  /**< Fourier-space arrays (PFFT backend) */
#endif
double **rvector_fft;                        /**< Real-space arrays (both backends) */

param_data params={0};
output_data outputs;
subbox_data subbox;
#ifdef PLC
plc_data plc;
plcgroup_data *plcgroups;
#endif

cputime_data cputime={0.0};

int WindowFunctionType;

group_data *groups;

char date_string[25];

int *frag_pos,*indices,*indicesY,*sorted_pos,*group_ID,*linking_list;
unsigned int *frag_map, *frag_map_update;
int map_to_be_used;
double f_m, f_rm, espo, f_a, f_ra, f_200, sigmaD0;

gsl_integration_workspace * workspace;
gsl_rng *random_generator;

mf_data mf;

gsl_spline **SPLINE;
gsl_interp_accel **ACCEL;
#if defined(SCALE_DEPENDENT) && defined(ELL_CLASSIC)
gsl_spline **SPLINE_INVGROW;
gsl_interp_accel **ACCEL_INVGROW;
#endif

#ifdef MOD_GRAV_FR
double H_over_c;
#endif

memory_data memory;

int ngroups;
extern pos_data obj, obj1, obj2;
ScaleDep_data ScaleDep;
