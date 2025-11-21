// Minimal C GridInterpolator for regular-grid linear interpolation.
// Depends on HDF5 (C API). Assumes results are stored as in gridutils.py:
// group "gridvals" with datasets "0","1",..., and "results/<dataset>" with
// shape (*grid_shape, n_out).

#ifndef GRIDUTILS_C_H
#define GRIDUTILS_C_H

#include <stddef.h>

typedef struct {
    size_t ndim;        // number of grid dimensions
    size_t n_out;       // length of output vector per grid point
    size_t *shape;      // length ndim
    double **gridvals;  // gridvals[d][i]
    double *data;       // flattened results array
} GridInterpolator;

// Load a GridInterpolator from HDF5.
// dataset_name should match a dataset under group "results".
// Returns NULL on failure.
GridInterpolator *grid_interpolator_load(const char *filename, const char *dataset_name);

// Free all memory associated with a GridInterpolator.
void grid_interpolator_free(GridInterpolator *gi);

// Multilinear interpolation at point x (length ndim).
// out must have capacity n_out. Returns 0 on success, non-zero on failure.
int grid_interpolate(const GridInterpolator *gi, const double *x, double *out);

#endif // GRIDUTILS_C_H
