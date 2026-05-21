/**
 * @file cubic_spline_interpolation.h
 * @brief Custom cubic spline interpolation for GPU-accelerated collapse time computation
 *
 * @details
 * Provides a GPU-friendly cubic spline interpolation structure and functions
 * that can be executed on both host and device via OpenMP target offload.
 *
 * This is required for GPU_OMP and GPU_OMP_FULL builds because:
 * - GSL splines are not GPU-compatible (dynamic allocation, complex state)
 * - Custom spline with coefficients pre-computed fits GPU execution model
 * - Coefficients can be copied to device via omp declare target
 *
 * The spline uses natural boundary conditions and pre-computes cubic polynomial
 * coefficients for each interval to enable fast evaluation on GPU.
 *
 * @author Leonardo collaboration, extended for GPU offload 2025
 * @see cubic_spline_interpolation.c (implementation)
 * @see collapse_times_GPU.c (GPU consumer)
 */

#ifndef CUBIC_SPLINE
#define CUBIC_SPLINE

/**
 * @brief GPU-optimized cubic spline structure
 *
 * Stores interpolation data and pre-computed cubic polynomial coefficients.
 * All arrays are contiguous (no indirect indexing) for GPU access efficiency.
 *
 * For each interval [x[i], x[i+1]], the spline value is computed as:
 * \f$ S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3 \f$
 *
 * @see custom_cubic_spline_init()
 * @see custom_cubic_spline_eval()
 */
typedef struct GPU_CubicSpline
{
  int size;           /**< Number of data points (length of x, y, d2y_data, coeff_*) */
  double *x;          /**< X-coordinates of data points (strictly increasing) */
  double *y;          /**< Y-coordinates of data points */
  double *d2y_data;   /**< Second derivatives at data points (computed from linear system) */
  double *coeff_a;    /**< Cubic spline coefficient a_i (y-value at x_i) */
  double *coeff_b;    /**< Cubic spline coefficient b_i (slope coefficient) */
  double *coeff_c;    /**< Cubic spline coefficient c_i (curvature coefficient) */
  double *coeff_d;    /**< Cubic spline coefficient d_i (cubic term) */
} CubicSpline;

extern CubicSpline *host_spline;  /**< Host-side spline (used in CPU/GPU_OMP_FULL modes) */

/**
 * @brief Evaluate cubic spline at a point (GPU-friendly)
 *
 * @details
 * Evaluates the spline \f$ S(x) \f$ at a given x-coordinate using O(log n)
 * binary search to find the interval, then Horner's method for cubic evaluation.
 *
 * For GPU_OMP_FULL: requires size parameter (spline structure not accessible on device).
 * For GPU_OMP: uses device spline structure; size is implicit.
 *
 * @param[in] spline  Pointer to CubicSpline structure
 * @param[in] x       Evaluation point (must be in [x[0], x[size-1]])
 * @param[in] size    (GPU_OMP_FULL only) Number of spline points
 *
 * @return Interpolated value at x
 *
 * @warning x must be within the spline domain; behavior is undefined outside.
 * @note GPU_OMP version uses implicit size from gpu_spline.size.
 *
 * @see custom_cubic_spline_init()
 */
#if defined(GPU_OMP_FULL)
// FULL GPU interpolation. It requires size as an extra argument
double custom_cubic_spline_eval(CubicSpline *const spline, const double x, int size);
#else
// Half GPU version: only the eval is done on GPU in this case
double custom_cubic_spline_eval(CubicSpline *const spline, const double x);
#endif

#if defined(GPU_OMP)
extern CubicSpline gpu_spline;                     /**< GPU spline object (GPU_OMP mode) */
#pragma omp declare target(gpu_spline, custom_cubic_spline_eval)
#elif defined(GPU_OMP_FULL) || defined(CUSTOM_INTERPOLATION)
extern CubicSpline *gpu_spline;                    /**< Pointer to GPU spline (GPU_OMP_FULL mode) */
extern CubicSpline **CT_Spline;                    /**< Array of collapse time splines (tabulated) */
#pragma omp declare target(gpu_spline)
#pragma omp declare target(CT_Spline, custom_cubic_spline_eval)
#endif // GPU_OMP

/**
 * @brief Initialize spline coefficients from data points
 *
 * @details
 * Computes natural cubic spline coefficients from (x_data, y_data) pairs.
 * This must be called after custom_cubic_spline_alloc() and before eval.
 *
 * Solves the tridiagonal system for second derivatives, then computes
 * cubic polynomial coefficients for each interval.
 *
 * @param[in,out] spline    Allocated CubicSpline structure (must have x, y, size set)
 * @param[in]  x_data       X-coordinates of input points (must be strictly increasing)
 * @param[in]  y_data       Y-coordinates of input points
 * @param[in]  size         Number of points
 *
 * @note This is a wrapper calling calculate_second_derivatives() then cubic_spline_coefficients().
 *
 * @see custom_cubic_spline_alloc()
 * @see calculate_second_derivatives()
 * @see cubic_spline_coefficients()
 */
void custom_cubic_spline_init( CubicSpline *const restrict spline,
			      const double *const restrict x_data,
			      const double *const restrict y_data,
			      const int                    size);

/**
 * @brief Allocate and initialize CubicSpline structure
 *
 * @param[in] size  Number of data points
 *
 * @return Pointer to allocated CubicSpline, or NULL on failure
 *
 * @note Caller must call custom_cubic_spline_free() to deallocate.
 *
 * @see custom_cubic_spline_free()
 */
CubicSpline* custom_cubic_spline_alloc(const int size);

/**
 * @brief Compute second derivatives of spline (tridiagonal solve)
 *
 * @param[in,out] spline  Structure with x, y arrays populated; outputs d2y_data
 *
 * @details
 * Solves the tridiagonal system derived from natural spline boundary conditions
 * to find second derivatives at each data point.
 *
 * @see cubic_spline_coefficients()
 */
void calculate_second_derivatives(CubicSpline *const spline);

/**
 * @brief Compute cubic polynomial coefficients for each interval
 *
 * @param[in,out] spline  Structure with d2y_data populated; outputs coeff_a/b/c/d
 *
 * @note This must be called after calculate_second_derivatives().
 *
 * @see calculate_second_derivatives()
 */
void cubic_spline_coefficients(CubicSpline *const spline);

/**
 * @brief Free allocated CubicSpline structure
 *
 * @param[in] spline  Pointer to CubicSpline (must have been allocated via custom_cubic_spline_alloc)
 *
 * @note Safe to call with NULL.
 *
 * @see custom_cubic_spline_alloc()
 */
void custom_cubic_spline_free(CubicSpline *spline);

#endif // CUBIC_SPLINE
