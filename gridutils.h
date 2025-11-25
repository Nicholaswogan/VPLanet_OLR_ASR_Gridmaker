// Minimal C GridInterpolator for regular-grid linear interpolation.
// Depends on HDF5 (C API). Assumes results are stored as in gridutils.py:
// group "gridvals" with datasets "0","1",..., and "results/<dataset>" with
// shape (*grid_shape, n_out).

#ifndef GRIDUTILS_H
#define GRIDUTILS_H

#include <stddef.h>

typedef struct {
    size_t ndim;        // number of grid dimensions
    size_t n_out;       // length of output vector per grid point
    size_t *shape;      // length ndim
    double **gridvals;  // gridvals[d][i]
    double *data;       // flattened results array
} GridInterpolator;

/*
 * Load a GridInterpolator from an HDF5 file.
 *
 * filename      Path to the HDF5 grid file.
 * dataset_name  Dataset under group "results" to load (e.g., "ASR_OLR").
 *
 * Returns: allocated GridInterpolator on success, NULL on failure.
 */
GridInterpolator *grid_interpolator_load(const char *filename, const char *dataset_name);

/*
 * Free all memory associated with a GridInterpolator (safe on NULL).
 *
 * gi  Pointer returned by grid_interpolator_load.
 */
void grid_interpolator_free(GridInterpolator *gi);

/*
 * Multilinear interpolation at a point.
 *
 * gi   Interpolator handle.
 * x    Input coordinates (length ndim).
 * out  Output array (length n_out) for the interpolated values.
 *
 * Returns 0 on success, non-zero on failure.
 */
int grid_interpolate(const GridInterpolator *gi, const double *x, double *out);

#endif // GRIDUTILS_H
