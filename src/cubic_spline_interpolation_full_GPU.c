#ifdef GPU_OMP_FULL

#include "cubic_spline_interpolation.h"
#include "pinocchio.h"
#include <omp.h>
#include <assert.h>
#include <math.h>
#include <time.h> 
#include <stdlib.h>

CubicSpline *custom_cubic_spline_alloc(const int size)
{
  CubicSpline *spline = (CubicSpline *)malloc(sizeof(CubicSpline));
  spline->size     = size;
  spline->x        = NULL;
  spline->y        = NULL;
  spline->d2y_data = malloc(size * sizeof(double));
  spline->coeff_a  = malloc((size - 1) * sizeof(double));
  spline->coeff_b  = malloc((size - 1) * sizeof(double));
  spline->coeff_c  = malloc((size - 1) * sizeof(double));
  spline->coeff_d  = malloc((size - 1) * sizeof(double));

  return spline;
}

void solve_natural_spline_tridiagonal_system(const double *x, const double *y, double *d2y, int n) {
    double *a   = malloc((n - 2) * sizeof(double));
    double *b   = malloc((n - 2) * sizeof(double));
    double *c   = malloc((n - 2) * sizeof(double));
    double *rhs = malloc((n - 2) * sizeof(double));

    // Step sizes
    for (int i = 1; i < n - 1; i++) {
        double h1  = x[i] - x[i - 1];
        double h2  = x[i + 1] - x[i];
        a[i - 1]   = h1 / 6.0;
        b[i - 1]   = (h1 + h2) / 3.0;
        c[i - 1]   = h2 / 6.0;
        rhs[i - 1] = (y[i + 1] - y[i]) / h2 - (y[i] - y[i - 1]) / h1;
    }

    // Forward elimination
    for (int i = 1; i < n - 2; i++) {
        double  m = a[i] / b[i - 1];
        b[i]   -= m * c[i - 1];
        rhs[i] -= m * rhs[i - 1];
    }

    // Back substitution
    d2y[0]     = 0.0;            // Natural boundary
    d2y[n - 1] = 0.0;            // Natural boundary
    d2y[n - 2] = rhs[n - 3] / b[n - 3];

    for (int i = n - 4; i >= 0; i--) {
        d2y[i + 1] = (rhs[i] - c[i] * d2y[i + 2]) / b[i];
    }

    free(a);
    free(b);
    free(c);
    free(rhs);
}

void custom_cubic_spline_init(CubicSpline  *const restrict spline, 
                              const double *const restrict x_data, 
                              const double *const restrict y_data, 
                              const int size)
{    
    // Copy data
    #pragma omp target device(devID)
    {
      spline->x                  = x_data;
      spline->y                  = y_data;
      spline->size               = size;
      spline->d2y_data[0]        = 0.0;  // Boundary condition
      spline->d2y_data[size - 1] = 0.0;  // Boundary condition
    }
    
    // Compute second derivatives on CPU
    solve_natural_spline_tridiagonal_system(x_data, y_data, spline->d2y_data, size);

    // Compute second derivatives
    // #pragma omp target teams distribute parallel for device(devID)
    // for (int i = 1; i < size - 1; i++)
    //   {
    //     const double h1  = (spline->x[i + 0] - spline->x[i - 1]);
    //     const double h2  = (spline->x[i + 1] - spline->x[i - 0]);
    //     const double dy1 = (spline->y[i + 0] - spline->y[i - 1]);
    //     const double dy2 = (spline->y[i + 1] - spline->y[i - 0]);

    //     spline->d2y_data[i] = 6.0 / (h1 + h2) * ((dy2 / h2) - (dy1 / h1));
    //   }

    // Compute coefficients
    #pragma omp target teams distribute parallel for device(devID)
    for (int i = 0; i < size - 1; i++)
      {
        double h           = spline->x[i + 1] - spline->x[i];
        spline->coeff_a[i] = spline->y[i];
        spline->coeff_b[i] = (spline->y[i + 1] - spline->y[i]) / h - h / 6.0 * (spline->d2y_data[i + 1] + 2.0 * spline->d2y_data[i]);
        spline->coeff_c[i] = spline->d2y_data[i] * 0.5;
        spline->coeff_d[i] = (spline->d2y_data[i + 1] - spline->d2y_data[i]) / (6.0 * h);
      }

    return;
}

double custom_cubic_spline_eval(CubicSpline *const spline, const double x, int size) {
   
    // Handle extrapolation
    if (x < spline->x[0]) {
        return spline->coeff_a[0] + spline->coeff_b[0] * (x - spline->x[0]);
    } else if (x > spline->x[size - 1]) {
        return spline->coeff_a[size - 2] + spline->coeff_b[size - 2] * (x - spline->x[size - 1]);
    }
    
    // Find the interval
    int k = 0;
    while (k < size - 1 && x > spline->x[k + 1]) {
        k++;
    }

    //  Step 2: Binary search for the interval index 
    // int k = 0;
    // int step = size / 2;
    // while (step > 0) {
    //     int mid = k + step;
    //     if (mid < size - 1 && x > spline->x[mid]) {
    //         k = mid;
    //     }
    //     step /= 2;
    // }

    // Evaluate spline
    double dx = x - spline->x[k];
    return spline->coeff_a[k] + spline->coeff_b[k] * dx + spline->coeff_c[k] * dx * dx + spline->coeff_d[k] * dx * dx * dx;
}

void custom_cubic_spline_free(CubicSpline *spline)
{
  if (spline->x != NULL)
    free(spline->x);

  if (spline->y != NULL)
    free(spline->y);

  if (spline->d2y_data != NULL)
    free(spline->d2y_data);

  if (spline->coeff_a != NULL)
    free(spline->coeff_a);

  if (spline->coeff_b != NULL)
    free(spline->coeff_b);

  if (spline->coeff_c != NULL)
    free(spline->coeff_c);

  if (spline->coeff_d != NULL)
    free(spline->coeff_d);

  if (spline != NULL)
    free(spline);
}

#endif
