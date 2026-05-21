/*****************************************************************
 *                        PINOCCHIO  V5.1                        *
 *  (PINpointing Orbit-Crossing Collapsed HIerarchical Objects)  *
 *****************************************************************

 This code was written by
 Pierluigi Monaco
 Copyright (C) 2016, extended for HeFFTe backend in 2025

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

/**
 * @file fmax-heffte.c
 * @brief HeFFTe-based FFT backend for PINOCCHIO collapse time computation
 *
 * @details
 * This module provides multi-backend FFT support via HeFFTe (Highly Efficient FFT library),
 * which can target FFTW (CPU), CUFFT (NVIDIA GPU), or ROCFFT (AMD GPU) at compile time.
 * It replaces the PFFT backend used in the default CPU-only build.
 *
 * **Algorithm:**
 *
 * The PINOCCHIO collapse time calculation (Stage 2 of the pipeline) requires computing
 * spatial derivatives of the gravitational potential on a 3D grid. This is done via FFT:
 *
 * 1. Density field is interpolated onto the grid
 * 2. Forward FFT (real-to-complex) via heffte_forward()
 * 3. Multiply in Fourier space by Green's function for each smoothing radius
 * 4. Inverse FFT (complex-to-real) via heffte_backward()
 * 5. Extract derivatives on particles via interpolation
 *
 * The HeFFTe library handles the pencil-slab MPI decomposition and backend selection
 * transparently. All operations are wrapped to maintain compatibility with the pfft API.
 *
 * **Parallelism:**
 *
 * - MPI decomposition: 1D pencil in x-direction (distributed across NTasks)
 * - OpenMP threading: Optional, via HeFFTe's FFTW backend (-DOMP flag)
 * - GPU offload: Full offload with GPU_OMP_FULL + heFFTe backend CUFFT
 * - No explicit GPU kernels: computation delegated to heFFTe library
 *
 * **HeFFTe specifics:**
 *
 * - Complex number format: struct my_double_complex (real, imag) instead of pfft_complex
 * - Backend selection at compile time: CUFFT (GPU) or FFTW (CPU)
 * - No transpose overhead in plan (heFFTe handles redistribution implicitly)
 *
 * **References:**
 *
 * - HeFFTe documentation: https://icl.utk.edu/fft/
 * - PINOCCHIO collapse time theory: Monaco et al. 2002, MNRAS 331, 587
 *
 * @author Pierluigi Monaco, David Goz, extended for HeFFTe
 * @date 2025
 * @version 5.1
 * @note Requires compilation with -DUSE_HEFFTE and HeFFTe library installed
 *
 * @see fmax-pfft.c (CPU-only PFFT alternative)
 * @see collapse_times.c (calls these FFT functions)
 * @see fmax.c (host code for collapse time loop)
 */

#include "pinocchio.h"


// -------------------------
// defines and variables
// -------------------------

//#define DEBUG

#ifdef DEBUG
  FILE *results;
  char filename[300];
#endif

#define GRID MyGrids[ThisGrid]

typedef ptrdiff_t point[3];
point *starts;

// -------------------------
// functions' prototypes
// -------------------------


double greens_function                  (double *, double, int, int);
int    cubes_order                      (const void *, const void *);

// -------------------------
// code segment
// -------------------------


int cubes_order(const void *A, const void *B)
{
  float a = *(starts[*(int*)A]);
  float b = *(starts[*(int*)B]);

  return (a - b);

  /*
  if( a > b)
    return 1;
  if(a < b)
    return -1;
  return 0;
  */
}




/**
 * @brief Initialize HeFFTe grid decomposition for a given FFT grid
 *
 * @details
 * Sets up MPI pencil decomposition for the current FFT grid and initializes
 * HeFFTe-specific box descriptors (inbox_low/high, outbox_low/high).
 * This is called once per grid before any FFT operations.
 *
 * **MPI Decomposition:**
 *
 * Decomposes the global grid (GSglobal[x] x GSglobal[y] x GSglobal[z])
 * into pencils distributed along the x-direction across NTasks MPI ranks.
 * Each rank gets approximately GSglobal[x]/NTasks slices, with remainder
 * distributed to the first ranks.
 *
 * @param[in] ThisGrid  Index of the FFT grid (0 to Ngrids-1)
 *
 * @return 0 on success, non-zero on error
 *
 * @warning Must be called before compute_fft_plans() for the same grid.
 * @warning This function modifies global variables: inbox_low/high, outbox_low/high, cvector_size.
 *
 * @note Side effects:
 * - Sets GRID.GSlocal[3], GRID.GSstart[3], GRID.GSlocal_k[3], GRID.GSstart_k[3]
 * - Computes cvector_size = GSlocal_k[x] * GSlocal_k[y] * GSlocal_k[z]
 * - Fills inbox/outbox boxes for HeFFTe
 *
 * @see compute_fft_plans()
 * @see finalize_fft()
 */
int set_one_grid(int ThisGrid)
{

  GRID.CellSize = (double)GRID.BoxSize / GRID.GSglobal[_x_];

  //int NTasks;
  //MPI_Comm_size(MPI_COMM_WORLD, &NTasks);
  
  /* Set decomposition for heffte */

  /* Remember that _x_ and _z_ axes must be reversed
     to ensure correctness of results with PFFT */

  long int local_x = MyGrids[0].GSglobal[_x_] / NTasks;
  long int rem     = MyGrids[0].GSglobal[_x_] % NTasks;
  
  long int local_x_start = ThisTask * local_x + (ThisTask < rem ? ThisTask : rem);
  long int local_x_count = local_x + (ThisTask < rem ? 1 : 0);
  long int local_x_end   = local_x_start + local_x_count - 1;
 
  
  /* Construct inboxes and outboxes for each task 
     For a L*M*N matrix, indices must go up to L-1, M-1, N-1 */
  inbox_low[_x_]   = (int)local_x_start;
  inbox_low[_y_]   = 0;
  inbox_low[_z_]   = 0;
  inbox_high[_x_]  = (int)local_x_end;
  inbox_high[_y_]  = (int)MyGrids[0].GSglobal[_y_]-1;
  inbox_high[_z_]  = (int)MyGrids[0].GSglobal[_z_]-1;
  outbox_low[_x_]  = (int)local_x_start;
  outbox_low[_y_]  = 0;
  outbox_low[_z_]  = 0;
  outbox_high[_x_] = (int)local_x_end;
  outbox_high[_y_] = (int)MyGrids[0].GSglobal[_y_]-1;
  outbox_high[_z_] = (int)MyGrids[0].GSglobal[_z_]/2;
      
  /* Try to adapt Heffte to PFFT */
  
  GRID.GSlocal[_x_] = local_x_count;
  GRID.GSlocal[_y_] = MyGrids[0].GSglobal[_y_];
  GRID.GSlocal[_z_] = MyGrids[0].GSglobal[_z_];

  GRID.GSlocal_k[_x_] = local_x_count;
  GRID.GSlocal_k[_y_] = MyGrids[0].GSglobal[_y_];
  GRID.GSlocal_k[_z_] = MyGrids[0].GSglobal[_z_]/2 + 1;
  
  GRID.GSstart[_x_] = local_x_start;
  GRID.GSstart[_y_] = 0;
  GRID.GSstart[_z_] = 0;
  
  GRID.GSstart_k[_x_] = local_x_start;
  GRID.GSstart_k[_y_] = 0;
  GRID.GSstart_k[_z_] = 0;

  //GRID.total_local_size_fft = GRID.GSlocal[_x_] * GRID.GSlocal[_y_] * GRID.GSlocal[_z_];
  GRID.total_local_size     = GRID.GSlocal[_x_] * GRID.GSlocal[_y_] * GRID.GSlocal[_z_];  

  cvector_size = GRID.GSlocal_k[_x_] * GRID.GSlocal_k[_y_] * GRID.GSlocal_k[_z_];

  GRID.total_local_size_fft = 2 * cvector_size;
  // printf("Rank %d: total local size fft %ld; total local size fft / 2 %ld, size complex %ld\n",
	//  ThisTask,
	//  GRID.total_local_size_fft,
	//  GRID.total_local_size_fft/2,
	//  GRID.GSlocal_k[_x_] * GRID.GSlocal_k[_y_] * GRID.GSlocal_k[_z_]);
  
  dprintf(VDBG, ThisTask, "[set grid %02d] task %d %ld "
	  "i: %ld %ld %ld - i start: %ld %ld %ld - "
	  "o: %ld %ld %ld - o start: %ld %ld %ld\n",
	  ThisGrid, ThisTask, GRID.total_local_size,
	  GRID.GSlocal[_x_], GRID.GSlocal[_y_], GRID.GSlocal[_z_],
	  GRID.GSstart[_x_], GRID.GSstart[_y_], GRID.GSstart[_z_],
	  GRID.GSlocal_k[_x_], GRID.GSlocal_k[_y_], GRID.GSlocal_k[_z_],
	  GRID.GSstart_k[_x_], GRID.GSstart_k[_y_], GRID.GSstart_k[_z_]);
  
  return 0;
}
  



/**
 * @brief Create HeFFTe FFT plan for all grids
 *
 * @details
 * Initializes the HeFFTe FFT plan (heffte_plan) for pencil-slab decomposed
 * 3D complex-to-real and real-to-complex transforms. The plan is created
 * with options specified in options_fft and uses the backend selected
 * at compile time (BACKEND macro: CUFFT for GPU, FFTW for CPU).
 *
 * This is a collective MPI operation: all ranks must call it with the
 * same grid configuration.
 *
 * @par MPI
 * Collective call on MPI_COMM_WORLD. All ranks participate.
 *
 * @return 0 on success, non-zero on HeFFTe error
 *
 * @warning Must be called after set_one_grid() for all grids.
 * @warning Must be called before forward_transform() or reverse_transform().
 *
 * @note The plan is stored in GRID.plan and reused for all smoothing radii.
 *       No transpose is needed between forward and reverse transforms.
 *
 * @see set_one_grid()
 * @see forward_transform()
 * @see reverse_transform()
 * @see finalize_fft()
 */
int compute_fft_plans()
{
  ptrdiff_t DIM[3];
  //int       pfft_flags;
  int       ThisGrid;
  
  dprintf(VMSG, 0, "[%s] Computing fft plans\n",fdate());

  int order[3] = {2,1,0};
  
  for (ThisGrid = 0; ThisGrid < Ngrids; ThisGrid++)
    { 
      /* Check that the plan is created correctly */
      int heffte_err;
      
      // printf("Task %d: Inbox low: %d, %d, %d\n, Inbox high: %d, %d, %d\n",
	    //  ThisTask,
	    //  inbox_low[0], inbox_low[1], inbox_low[2],
	    //  inbox_high[0], inbox_high[1], inbox_high[2]);
      
      heffte_err = heffte_plan_create_r2c(BACKEND, inbox_low, inbox_high, order,
					  outbox_low, outbox_high, order,
					  2, MPI_COMM_WORLD, &options_fft, &GRID.plan);
      
 
      //printf("Rank %d --> Device %d\n", ThisTask, devID);
      
      if (heffte_err != Heffte_SUCCESS)
	{
	  printf("Heffte error in plan create %d", heffte_err);
	  return 1;
	}

      /* Define inbox size, outbox size and workspace */
      int inbox_size  = heffte_size_inbox(GRID.plan);
      int outbox_size = heffte_size_outbox(GRID.plan);
    
      // printf("Task %d: inbox size %d; outbox size %d\n",
	    //  ThisTask, inbox_size, outbox_size);
    }

  /* NO REVERSE PLAN IS NEEDED FOR HEFFTE */

  dprintf(VMSG, 0, "[%s] fft plans done\n",fdate());

  return 0;
}


/**
 * @brief Execute forward real-to-complex FFT via HeFFTe
 *
 * @details
 * Performs a real-to-complex fast Fourier transform of the density field
 * stored in rvector_fft[ThisGrid]. The input array is expected to contain
 * the density field interpolated to the current grid. Output is written
 * to cvector_fft[ThisGrid] in Fourier space.
 *
 * This is part of the PINOCCHIO collapse time calculation pipeline:
 * density \f$ \rho(\mathbf{r}) \f$ \f$ \to \f$ FFT \f$ \to \f$ \f$ \hat{\rho}(\mathbf{k}) \f$
 *
 * @param[in] ThisGrid  Index of the FFT grid (0 to Ngrids-1)
 *
 * @return Time spent in HeFFTe forward transform (in seconds)
 *
 * @par MPI
 * Collective operation within the FFT_Comm communicator. All ranks exchange
 * data to complete the pencil redistribution.
 *
 * @warning The input array rvector_fft[ThisGrid] is destroyed during the transform.
 * @warning Must be called after set_one_grid() and compute_fft_plans().
 *
 * @see reverse_transform()
 * @see compute_derivative()
 */
double forward_transform(int ThisGrid)
{
  double time;

  struct my_double_complex * cvector_fft1 = cvector_fft[ThisGrid];
  double * rvector_fft1 = rvector_fft[ThisGrid];

  time=MPI_Wtime();

 #ifdef GPU_OMP_FULL
 #pragma omp target update to(rvector_fft[ThisGrid][0:GRID.total_local_size]) device(devID)
 #endif //GPU_OMP_FULL

  double time_comp = MPI_Wtime();

 #ifdef GPU_OMP_FULL
#pragma omp target data use_device_addr(cvector_fft1, rvector_fft1) device(devID)
 #endif //GPU_OMP_FULL
  heffte_forward_d2z(GRID.plan, rvector_fft1, cvector_fft1, Heffte_SCALE_NONE);

  cputime.fft_compute += MPI_Wtime() - time_comp;

 #ifdef GPU_OMP_FULL
 #pragma omp target update from(cvector_fft[ThisGrid][0:cvector_size]) device(devID)
 #endif //GPU_OMP_FULL
  
  return MPI_Wtime()-time;
}


/**
 * @brief Execute inverse complex-to-real FFT via HeFFTe
 *
 * @details
 * Performs a complex-to-real inverse Fourier transform of the Fourier-space
 * data in cvector_fft[ThisGrid] (typically after multiplication by Green's function
 * in Fourier space). Output is written to rvector_fft[ThisGrid] in real space.
 *
 * In the PINOCCHIO collapse time calculation:
 * \f$ \hat{\phi}(\mathbf{k}) \to \text{IFFT} \to \phi(\mathbf{r}) \f$
 *
 * @param[in] ThisGrid  Index of the FFT grid (0 to Ngrids-1)
 *
 * @return Time spent in HeFFTe reverse (inverse) transform (in seconds)
 *
 * @par MPI
 * Collective operation within the FFT_Comm communicator.
 *
 * @warning The input array cvector_fft[ThisGrid] is destroyed during the transform.
 * @warning Must be called after set_one_grid() and compute_fft_plans().
 *
 * @see forward_transform()
 * @see compute_derivative()
 */
double reverse_transform(int ThisGrid)
{
  double time;

  struct my_double_complex * cvector_fft1 = cvector_fft[ThisGrid];
  double * rvector_fft1 = rvector_fft[ThisGrid];

  time=MPI_Wtime();

 #ifdef GPU_OMP_FULL
 #pragma omp target update to(cvector_fft[ThisGrid][0:cvector_size]) device(devID)
 #endif //GPU_OMP_FULL

  double time_comp = MPI_Wtime();

 #ifdef GPU_OMP_FULL
#pragma omp target data use_device_addr(cvector_fft1, rvector_fft1) device(devID)
 #endif //GPU_OMP_FULL
  
  /* Heffte_SCALE_FULL accounts for the inverse transform normalization */
  heffte_backward_z2d(GRID.plan, cvector_fft1, rvector_fft1, Heffte_SCALE_FULL);

  cputime.fft_compute += MPI_Wtime() - time_comp;

 #ifdef GPU_OMP_FULL
 #pragma omp target update from(rvector_fft[ThisGrid][0:GRID.total_local_size]) device(devID)
 #endif //GPU_OMP_FULL
  
  return MPI_Wtime() - time;
}


/**
 * @brief Destroy HeFFTe FFT plan and deallocate FFT arrays
 *
 * @details
 * Cleans up all HeFFTe resources and deallocates the FFT working arrays
 * (cvector_fft, rvector_fft) for all grids. This should be called at the
 * end of the collapse time computation, before program exit.
 *
 * @return 0 on success, 1 on error (failure to deallocate)
 *
 * @warning This is a collective operation (each rank destroys its own plan).
 * @warning After this call, forward_transform() and reverse_transform() cannot be called.
 *
 * @see compute_fft_plans()
 * @see deallocate_fft_vectors()
 */
int finalize_fft()
{
#ifndef RECOMPUTE_DISPLACEMENTS
  int ThisGrid;

  for (ThisGrid = Ngrids-1; ThisGrid >= 0; ThisGrid--)
    {
      heffte_plan_destroy(GRID.plan);
      if (deallocate_fft_vectors(ThisGrid))
	return 1;
      
      //#ifdef GPU_OMP_FULL
      //#pragma omp target exit data map(delete: rvector_fft[ThisGrid], cvector_fft[ThisGrid])
      //#endif //GPU_OMP_FULL
    }

 #if defined(_OPENMP) && !defined(GPU_OMP_FULL)
  fftw_cleanup_threads();
 #endif //threads
  
  //MPI_Comm_free(&MPI_COMM_WORLD);
#endif

  return 0;
}


/**
 * @brief Compute spatial derivatives of potential via Fourier-space multiplication
 *
 * @details
 * Multiplies the Fourier-space density field (in cvector_fft[ThisGrid]) by the
 * Green's function kernel \f$ \frac{1}{k^2} \f$ (and derivatives thereof) to compute
 * gravitational potential or its spatial derivatives. The process is:
 *
 * 1. For each Fourier mode, apply Green's function multiplication
 * 2. Handle Laplacian: \f$ \nabla^2 \phi = -\rho \f$ by multiplying by \f$ 1/k^2 \f$
 * 3. Compute derivatives: \f$ \frac{\partial}{\partial x} = -i k_x \hat{\phi}(k) \f$
 * 4. Inverse FFT to real space
 * 5. Store results in products (via write_from_rvector())
 *
 * @param[in]  ThisGrid            Index of the FFT grid (0 to Ngrids-1)
 * @param[in]  first_derivative    Which component: 0=x, 1=y, 2=z; or -1 for potential
 * @param[in]  second_derivative   0 to compute only first_derivative; 1 to compute mixed/second derivatives
 *
 * @return 0 on success, non-zero on error
 *
 * @par Physics
 * Implements the Zel'dovich approximation: \f$ \vec{x}(\mathbf{q}, z) = \mathbf{q} - D(z) \nabla \phi(\mathbf{q}) / a(z) \f$
 * where \f$ \phi \f$ is the Newtonian potential and \f$ D(z) \f$ is the growth factor.
 *
 * @warning Must be called between forward_transform() and reverse_transform() for the same grid.
 * @warning Overwrites cvector_fft[ThisGrid] with derivative data.
 *
 * @see forward_transform()
 * @see reverse_transform()
 * @see greens_function()
 */
int compute_derivative(int ThisGrid, int first_derivative, int second_derivative)
{
  int    swap, local[3], start[3], C[3], N[3], Nhalf[3];
  double knorm[3];
#ifdef SCALE_DEPENDENT
  double k_module;
#endif

#ifdef DEBUG
  sprintf(filename,"results.%d-%d.%d",first_derivative,second_derivative,ThisTask);
  results=fopen(filename,"w");
#endif
  
  // note: for PFFT the transposed order is different when
  // the subdivision is done either in 1 or higher dimensions.
  // In 1D the memory order is [ y, x, z ], while in 2D and 3D
  // it is [ y, z, x].

  /* GINOCCHIO_CHECK */
  //
  
  if(!params.use_transposed_fft)
    // this is the non-transposed order x, y, z    
    for(int i = 0; i < 3; i++)
    C[i] = i;
  else
    {
      if(internal.tasks_subdivision_dim > 1)
	C[0] = _y_, C[1] = _z_, C[2] = _x_;
      else
	C[0] = _y_, C[1] = _x_, C[2] = _z_;
    }
  
  for(int i = 0; i < 3; i++)
    {
      // grid number
      N[C[i]]     = GRID.GSglobal[C[i]];
      // Nyquist frequencies
      Nhalf[C[i]] = N[i] / 2;
      // k vectors
      knorm[C[i]] = 2.*PI / (double)GRID.GSglobal[ C[i] ];
      
      local[C[i]] = GRID.GSlocal_k[C[i]];
      start[C[i]] = GRID.GSstart_k[C[i]];

    }

  // for first derivatives the real and imaginary parts must be swapped
  swap=((first_derivative==0 && second_derivative> 0) ||
	(first_derivative> 0 && second_derivative==0));

  double growth_rate=1.0;

  // loop over k-space indices

  int idx;
  for (idx = 0; idx < local[_x_]; idx++)
    {
      int    ii[3];
      
      ii[_x_] = idx + start[_x_]; // this is the kx index
      
      if (ii[_x_] > Nhalf[_x_])   // kx-N if kx > Nx/2
	ii[_x_] -= N[_x_];
      
      double k_x  = knorm[_x_] * ii[_x_]; 
      double k2_0 = k_x * k_x;

      int idy;
      for (idy = 0; idy < local[_y_]; idy++)
	{
	  ii[_y_] = idy + start[_y_]; // this is the ky index
	  
	  if (ii[_y_] > Nhalf[_y_])
	    ii[_y_] -= N[_y_];        // ky-N if ky > Ny/2
	  
	  double k_y  = knorm[_y_] * ii[_y_];
	  double k2_1 = k2_0 + k_y * k_y;

	  int idz;
	  for (idz = 0; idz < local[_z_]; idz++)
	    {
	      ii[_z_] = idz + start[_z_]; // this is the kz index
	      
	      if (ii[_z_] > Nhalf[_z_])
		ii[_z_] -= N[_z_];        // kz-N if ky > Nz/2
	      
	      double k_z  = knorm[_z_] * ii[_z_];

              double k_squared  = k2_1 + k_z * k_z;

	      //#ifdef SCALE_DEPENDENT
	      double k_module = sqrt(k_squared);

	      /* In the scale-dependent case the delta(k) must be multiplied 
		 by the relevant growth rate */
	      switch (ScaleDep.order)
		{
		case 0:
		  growth_rate = 1.0;
		  break;
		case 1:
		  growth_rate = GrowingMode(ScaleDep.redshift,k_module);
		  break;
		case 2:
		  growth_rate = GrowingMode_2LPT(ScaleDep.redshift,k_module);
		  break;
		case 3:
		  growth_rate = GrowingMode_3LPT_1(ScaleDep.redshift,k_module);
		  break;
		case 4:
		  growth_rate = GrowingMode_3LPT_2(ScaleDep.redshift,k_module);
		  break;
		default:
		  growth_rate = 1.0;
		  break;
		}
	      //#endif

	      int index = 2*(( idx * local[_y_] + idy ) * local[_z_] + idz);
	      
              if (k_squared != 0.)
		{

		  // Gaussian smoothing window
		  double smoothing = exp(-0.5 * k_squared * Rsmooth * Rsmooth);
		  
		  double diff_comp[4];
		  
		  // the components are stored in the vectors that control the differentiation
      
		  diff_comp[0] = 1.0;
		  diff_comp[1] = k_x;
		  diff_comp[2] = k_y;
		  diff_comp[3] = k_z;

		  double green = greens_function(diff_comp, k_squared, first_derivative, second_derivative);

		  cvector_fft[ThisGrid][index/2].real *=  green * smoothing * growth_rate;
		  cvector_fft[ThisGrid][index/2].imag *=  green * smoothing * growth_rate;
		}

              if (swap)
		{
		  double tmp                          = cvector_fft[ThisGrid][index/2].imag;
		  cvector_fft[ThisGrid][index/2].imag = cvector_fft[ThisGrid][index/2].real;
		  cvector_fft[ThisGrid][index/2].real = -tmp;
		}
	    }
	}
    }

  /* int idy, idz, index; */
  /* for (idx = 0; idx < local[_x_]; idx++) */
  /*   for (idy = 0; idy < local[_y_]; idy++) */
  /*     for (idz = 0; idz < local[_z_]; idz++) */
  /* 	{ */
  /* 	  index = 2*(( idx * local[_y_] + idy ) * local[_z_] + idz); */
  /* 	  printf(" %3d %3d %3d  %6d  %8f %8f\n", */
  /* 		 idx,idy,idz,index,cvector_fft[ThisGrid][index/2][0],cvector_fft[ThisGrid][index/2][1]); */
  /* 	}   */

  /* char name[60]; */
  /* char coord[4] = {' ', 'x', 'y', 'z'}; */
  /* sprintf(name, "c_before_derivative.1.%c", coord[first_derivative]); */
  /* dump_cvector((double*)cvector_fft[ThisGrid], params.use_transposed_fft*internal.tasks_subdivision_dim, */
  /* 	       MyGrids[ThisGrid].GSglobal[_x_], */
  /* 	       MyGrids[ThisGrid].GSlocal_k, */
  /* 	       MyGrids[ThisGrid].GSstart_k, name, 0); */
  

  if (!ThisTask)
    dprintf(VMSG, 0, "[%s] compute_derivative: starting fft\n",fdate());

  {
    
    double time = reverse_transform(ThisGrid);
    
    
    if (!ThisTask)
      dprintf(VMSG, 0, "[%s] compute_derivative: done fft, cpu time = %f\n",fdate(),time);
    
    cputime.fft+=time;
  }

  /*
  for (int local_x = 0; local_x < MyGrids[ThisGrid].GSlocal[_x_]; local_x++) 
    for (int local_y = 0; local_y < MyGrids[ThisGrid].GSlocal[_y_]; local_y++) 
      for (int local_z = 0; local_z < MyGrids[ThisGrid].GSlocal[_z_]; local_z++)
  	{ 
   	  int index=local_x + MyGrids[ThisGrid].GSlocal[_x_] * (local_y + local_z * MyGrids[ThisGrid].GSlocal[_y_]); 
	  printf(" %3d %3d %3d  %6d  %8f\n",local_x,local_y,local_z,index,rvector_fft[ThisGrid][index]); 
	} 
  */
  fflush(stdout);

  return 0;
}


double greens_function(double *diff_comp, double k_squared, int first_derivative, int second_derivative)
{

  if (first_derivative == -1 && second_derivative == -1)
    /* in this case the greens_function is simply 1 */
    return 1.0;

  if (first_derivative==0 && second_derivative==0)
    return -diff_comp[first_derivative]*diff_comp[second_derivative]/k_squared;
  else
    return diff_comp[first_derivative]*diff_comp[second_derivative]/k_squared;

}


void write_in_cvector(int ThisGrid, double * restrict vector)
{
  /*
#pragma omp target use_device_addr(vector, cvector_fft[ThisGrid])
#pragma omp target teams distribute parallel for
  */
  // non-vector code   
  for ( long int i = 0; i < GRID.total_local_size_fft; i++ )
    {
      /*
      cvector_fft[ThisGrid][i].real = vector[i];
      cvector_fft[ThisGrid][i].imag = vector[i+1];
      */
      *((double*)cvector_fft[ThisGrid] + i) = *(vector+i);
      //printf("index %ld <-- %f, %f\n", i, cvector_fft[ThisGrid][i].real, cvector_fft[ThisGrid][i].imag);
    }
  /*
  for ( long int i = 0; i < cvector_size; i++)
    printf("index %ld <-- %f, %f\n", i, cvector_fft[ThisGrid][i].real, cvector_fft[ThisGrid][i].imag);
  */
}


void write_from_cvector(int ThisGrid, double * restrict vector)
{
  // non-vector code   
  for ( long int i = 0; i < GRID.total_local_size_fft; i++ )
    {
      *(vector + i) = *((double*)cvector_fft[ThisGrid] + i);
      /*
      vector[i]   = cvector_fft[ThisGrid][i].real;
      vector[i+1] = cvector_fft[ThisGrid][i].imag;
      */
    }
}



void write_in_rvector(int ThisGrid, double * restrict vector)
{
  dvec * restrict target = (dvec*)rvector_fft[ThisGrid];
  dvec * restrict source = (dvec*)vector;
  int mysize = GRID.total_local_size / DVEC_SIZE;
  
#if !defined(_OPENMP)
#pragma GCC ivdep
#endif  
#ifdef _OPENMP
#pragma omp for simd schedule(static)  
#endif
  for( int i = 0; i < mysize; i++ )
    *(target + i) = *(source + i);
  
  for( int i = mysize*DVEC_SIZE ; i < GRID.total_local_size; i++ )
    *((double*)rvector_fft[ThisGrid] + i) = *(vector + i);

  // non-vector code   
  /* for(i = 0; i < GRID.total_local_size; i++) */
  /*   *(rvector_fft[ThisGrid] + i) = *(vector + i); */

}


void write_from_rvector(int ThisGrid, double * restrict vector)
{
  dvec * restrict source = (dvec*)rvector_fft[ThisGrid];
  dvec * restrict target = (dvec*)vector;
  int mysize = GRID.total_local_size / DVEC_SIZE;

#if !defined(_OPENMP)
#pragma GCC ivdep
#endif
#ifdef _OPENMP
#pragma omp for simd schedule(static)  
#endif
  for( int i = 0; i < mysize; i++ )
    *(target + i) = *(source + i);

  for( int i = mysize*DVEC_SIZE; i < GRID.total_local_size; i++ )
    *(vector + i) = *((double*)rvector_fft[ThisGrid] + i);

  // non-vector code 
  /* for( i = 0 ; i < GRID.total_local_size; i++) */
  /*   *(vector + i) = *(rvector_fft[ThisGrid] + i); */

}

void write_from_rvector_to_products(int ThisGrid, int ia, int order)
{
  /* writes the results of a (first) derivative to the corresponding products */
  switch (order)
    {
    case 1:
#ifdef _OPENMP
#pragma omp parallel
      {
#pragma omp for nowait
#endif
	for (int index = 0; index < MyGrids[ThisGrid].total_local_size; index++)
	  products[index].Vel[ia]=*((double*)rvector_fft[ThisGrid] + index);
#ifdef _OPENMP
      }
#endif
      break;

#ifdef TWO_LPT

    case 2:
#ifdef _OPENMP
#pragma omp parallel
      {
#pragma omp for nowait
#endif
	for (int index = 0; index < MyGrids[ThisGrid].total_local_size; index++)
	  products[index].Vel_2LPT[ia]=*((double*)rvector_fft[ThisGrid] + index);
#ifdef _OPENMP
      }
#endif
      break;

#ifdef THREE_LPT

    case 3:
#ifdef _OPENMP
#pragma omp parallel
      {
#pragma omp for nowait
#endif
	for (int index = 0; index < MyGrids[ThisGrid].total_local_size; index++)
	  products[index].Vel_3LPT_1[ia]=*((double*)rvector_fft[ThisGrid] + index);
#ifdef _OPENMP
      }
#endif
      break;

    case 4:
#ifdef _OPENMP
#pragma omp parallel
      {
#pragma omp for nowait
#endif
	for (int index = 0; index < MyGrids[ThisGrid].total_local_size; index++)
	  products[index].Vel_3LPT_2[ia]=*((double*)rvector_fft[ThisGrid] + index);
#ifdef _OPENMP
      }
#endif
      break;

#endif
#endif

    default:
      break;
    }

}


int store_velocities()
{
  /* LUCA: vettorializziamo e ompizziamo? */

  /* loop on all particles */
  for (int index = 0; index < MyGrids[0].total_local_size; index++)
    for (int i = 0; i < 3; i++)
      products[index].Vel[i]=first_derivatives[0][i][index];

  return 0;
}


void dump_cvector(double * restrict vector, int T, int Nmesh, ptrdiff_t * restrict local_n, ptrdiff_t * restrict local_start,  char *filename, int ThisGrid)
// this routine dump in a unique files the 3d complex-space cube
// the quantity of each "particle" (i.e. cell) is written with its unique ordinal number in x,y,z
//   order
// - T          : flags whether the memory order is transposed or not, as follows:
//                   0: non transposed
//                   1: transposed as in 1D pfft order
//                 2-3: transposed as in 2d/3D pfft order
// - NMesh_2    : Ngrid/2
// - local_n    : the extension of this tasks domain in the 3 coordinates
// - local_start: the starting point of this task domain in the 3 coordinates
//
{

  int ii, i, j, k, addr, addr_;
  int Nmesh_2 = Nmesh / 2;
  FILE *file;
  char myfilename[300];
  
  sprintf(myfilename, "partial_%d", ThisTask);

  for(ii = 0; ii < NTasks; ii++)
    {
      if( (ThisTask == ii) &&
	  (local_start[_z_] < Nmesh_2-1) )	
	{
	  file = fopen(myfilename, "w");

	  if(T == 1)
	    // transposed order, 1D decomposition
	    {
	      for(j = 0; j < local_n[_y_]; j++)
	  	for(i = 0; i < local_n[_x_]; i++)
	  	  for(k = 0; k < (local_n[_z_]) && (k + local_start[_z_] <= Nmesh_2); k++)
	  	    {
		      if(internal.dump_kdensity == 1)
			// write in x,y order of the non-transposed case
			addr_   = ((i+local_start[_x_])*Nmesh + j+local_start[_y_])*(Nmesh_2+1) + k+local_start[_z_];
		      else
			// write in natural order
			addr_   = ((j+local_start[_y_])*Nmesh + i+local_start[_x_])*(Nmesh_2+1) + k+local_start[_z_];
	  	      addr    = 2*((j * local_n[_x_] + i) * local_n[_z_] + k);
	  	      fprintf(file, "%d - %g %g\n", addr_,
	  		      vector[addr], vector[addr+1]);
	  	    }
	      
	    }
	  else if(T > 1)
	    // transposed order, 2D or 3D decomposition
	    {
	      for(j = 0; j < local_n[_y_]; j++)
	  	for(k = 0; k < (local_n[_z_]) && (k + local_start[_z_] <= Nmesh_2); k++)
	  	  for(i = 0; i < local_n[_x_]; i++)
	  	    {
		      if(internal.dump_kdensity == 1)
			// write in x,y order of the non-transposed case
			addr_   = ((i+local_start[_x_])*Nmesh + j+local_start[_y_])*(Nmesh_2+1) + k+local_start[_z_];
		      else
			// write in natural order
			addr_   = ((j+local_start[_y_])*Nmesh + k+local_start[_z_])*(Nmesh_2+1) + i+local_start[_x_];
	  	      addr    = 2*((j * local_n[_z_] + k) * local_n[_x_] + i);
	  	      fprintf(file, "%d - %g %g\n", addr_,
	  		      vector[addr], vector[addr+1]);
	  	    }
	      
	    }
	  else
	    // not transposed
	    {
	      for(i = 0; i < local_n[_x_]; i++)
	  	for(j = 0; j < local_n[_y_]; j++)
	  	  for(k = 0; k < (local_n[_z_]) && (k + local_start[_z_] <= Nmesh_2); k++)
	  	    {
		      if(internal.dump_kdensity == 2)
			// write in x,y order of the transposed case (1-D decomposition)
			addr_   = ((j+local_start[_y_])*Nmesh + i+local_start[_x_])*(Nmesh_2+1) + k+local_start[_z_];
		      else
			// write in natural order
			addr_   = ((i+local_start[_x_])*Nmesh + j+local_start[_y_])*(Nmesh_2+1) + k+local_start[_z_];
	  	      addr    = 2*( (i*local_n[_y_] + j)*local_n[_z_] + k);
	  	      fprintf(file, "%d - %g %g\n", addr_,
	  		      vector[addr], vector[addr+1]);
	  	    }
	      
	    }

	  fflush(file);
	  fclose(file);
	}

      MPI_Barrier(MPI_COMM_WORLD);
    }
  
  if(ThisTask == 0)
    {
      int err;
      sprintf(myfilename, "cat partial_* | sort -k1 -n > %s", filename);
      err = system(myfilename);
      err = system("rm -f partial_*");
      if(err != 0)
	dprintf(VXX, 0, "something got wrong while writing cvector dump file\n");
    }
     
  
  return;

}


void dump_rvector(double * restrict vector, int Nmesh, ptrdiff_t * restrict local_n, ptrdiff_t * restrict local_start,  char *filename, int ThisGrid)
// this routine dump in a unique files the 3d complex-space cube
// the quantity of each "particle" (i.e. cell) is written with its unique ordinal number in x,y,z
//   order
// - NMesh_2    : Ngrid/2
// - local_n    : the extension of this tasks domain in the 3 coordinates
// - local_start: the starting point of this task domain in the 3 coordinates
//
{

  int ii, i, j, k, addr, addr_;
  //int Nmesh_2 = Nmesh / 2;
  FILE *file;
  char myfilename[300];
  
  sprintf(myfilename, "partial_%d", ThisTask);

  for(ii = 0; ii < NTasks; ii++)
    {
      if( (ThisTask == ii) )
	{
	  file = fopen(myfilename, "w");
	  
	  for(i = 0; i < local_n[_x_]; i++)
	    for(j = 0; j < local_n[_y_]; j++)
	      for(k = 0; k < local_n[_z_]; k++)
		{
		  /* if (params.use_transposed_fft) */
		  /*   { */
		  /*     addr_   = ((j+local_start[_y_])*Nmesh + k+local_start[_z_])*Nmesh + i+local_start[_x_]; */
		  /*     addr    = ( (k*local_n[_y_] + j)*local_n[_x_] + i); */
		  /*   } */
		  /* else */
		    { // OK, AGGIUDICATO
		      addr_   = ((i+local_start[_x_])*Nmesh + j+local_start[_y_])*Nmesh + k+local_start[_z_];
		      addr    = ( (i*local_n[_y_] + j)*local_n[_z_] + k);
		    }

		  fprintf(file, "%d  %d %d %d  %g\n", addr_, 
			  (int)(i + local_start[_x_]), (int)(j + local_start[_y_]), (int)(k + local_start[_z_]),
			  vector[addr]);
		}

	  fflush(file);
	  fclose(file);
	}

      MPI_Barrier(MPI_COMM_WORLD);
    }
  
  if(ThisTask == 0)
    {
      int err;
      sprintf(myfilename, "cat partial_* | sort -k1 -n > %s", filename);
      err = system(myfilename);
      err = system("rm -f partial_*");
      if(err != 0)
	dprintf(VXX, 0, "something got wrong while writing cvector dump file\n");
    }
     
  
  return;

}
