#include "gridutils_c.h"
#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void free_gridvals(double **gridvals, size_t ndim) {
    if (!gridvals) return;
    for (size_t i = 0; i < ndim; ++i) {
        free(gridvals[i]);
    }
    free(gridvals);
}

void grid_interpolator_free(GridInterpolator *gi) {
    if (!gi) return;
    free_gridvals(gi->gridvals, gi->ndim);
    free(gi->shape);
    free(gi->data);
    free(gi);
}

static int read_gridvals(hid_t file_id, size_t ndim, double ***gridvals_out, size_t **shape_out) {
    hid_t g_grid = H5Gopen(file_id, "gridvals", H5P_DEFAULT);
    if (g_grid < 0) return -1;

    double **gridvals = (double **)calloc(ndim, sizeof(double *));
    size_t *shape = (size_t *)calloc(ndim, sizeof(size_t));
    if (!gridvals || !shape) {
        H5Gclose(g_grid);
        free(gridvals);
        free(shape);
        return -1;
    }

    for (size_t d = 0; d < ndim; ++d) {
        char name[32];
        snprintf(name, sizeof(name), "%zu", d);
        hid_t dset = H5Dopen(g_grid, name, H5P_DEFAULT);
        if (dset < 0) {
            H5Gclose(g_grid);
            free_gridvals(gridvals, d);
            free(shape);
            return -1;
        }
        hid_t space = H5Dget_space(dset);
        int nd = H5Sget_simple_extent_ndims(space);
        hsize_t dims[8];
        H5Sget_simple_extent_dims(space, dims, NULL);
        if (nd != 1) {
            H5Sclose(space);
            H5Dclose(dset);
            H5Gclose(g_grid);
            free_gridvals(gridvals, d);
            free(shape);
            return -1;
        }
        shape[d] = (size_t)dims[0];
        gridvals[d] = (double *)malloc(shape[d] * sizeof(double));
        if (!gridvals[d]) {
            H5Sclose(space);
            H5Dclose(dset);
            H5Gclose(g_grid);
            free_gridvals(gridvals, d);
            free(shape);
            return -1;
        }
        if (H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, gridvals[d]) < 0) {
            H5Sclose(space);
            H5Dclose(dset);
            H5Gclose(g_grid);
            free_gridvals(gridvals, d + 1);
            free(shape);
            return -1;
        }
        H5Sclose(space);
        H5Dclose(dset);
    }

    H5Gclose(g_grid);
    *gridvals_out = gridvals;
    *shape_out = shape;
    return 0;
}

GridInterpolator *grid_interpolator_load(const char *filename, const char *dataset_name) {
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) return NULL;

    // Determine rank of dataset
    char ds_path[256];
    snprintf(ds_path, sizeof(ds_path), "results/%s", dataset_name);
    hid_t dset = H5Dopen(file_id, ds_path, H5P_DEFAULT);
    if (dset < 0) {
        H5Fclose(file_id);
        return NULL;
    }
    hid_t space = H5Dget_space(dset);
    int rank = H5Sget_simple_extent_ndims(space);
    if (rank < 1) {
        H5Sclose(space);
        H5Dclose(dset);
        H5Fclose(file_id);
        return NULL;
    }
    hsize_t dims[16];
    H5Sget_simple_extent_dims(space, dims, NULL);
    size_t ndim = (rank > 1) ? (size_t)(rank - 1) : 1;
    size_t n_out = (rank > 1) ? (size_t)dims[rank - 1] : 1;

    double **gridvals = NULL;
    size_t *shape = NULL;
    if (read_gridvals(file_id, ndim, &gridvals, &shape) != 0) {
        H5Sclose(space);
        H5Dclose(dset);
        H5Fclose(file_id);
        return NULL;
    }

    // Read data
    size_t total = 1;
    for (size_t i = 0; i < (size_t)rank; ++i) total *= (size_t)dims[i];
    double *data = (double *)malloc(total * sizeof(double));
    if (!data) {
        free_gridvals(gridvals, ndim);
        free(shape);
        H5Sclose(space);
        H5Dclose(dset);
        H5Fclose(file_id);
        return NULL;
    }
    if (H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
        free(data);
        free_gridvals(gridvals, ndim);
        free(shape);
        H5Sclose(space);
        H5Dclose(dset);
        H5Fclose(file_id);
        return NULL;
    }

    H5Sclose(space);
    H5Dclose(dset);
    H5Fclose(file_id);

    GridInterpolator *gi = (GridInterpolator *)calloc(1, sizeof(GridInterpolator));
    if (!gi) {
        free(data);
        free_gridvals(gridvals, ndim);
        free(shape);
        return NULL;
    }
    gi->ndim = ndim;
    gi->n_out = n_out;
    gi->shape = shape;
    gi->gridvals = gridvals;
    gi->data = data;
    return gi;
}

int grid_interpolate(const GridInterpolator *gi, const double *x, double *out) {
    if (!gi || !x || !out) return -1;
    size_t ndim = gi->ndim;
    size_t n_out = gi->n_out;

    // Find bounding indices and weights in each dimension
    size_t *idx_lo = (size_t *)malloc(ndim * sizeof(size_t));
    size_t *idx_hi = (size_t *)malloc(ndim * sizeof(size_t));
    double *t = (double *)malloc(ndim * sizeof(double));
    if (!idx_lo || !idx_hi || !t) {
        free(idx_lo); free(idx_hi); free(t);
        return -1;
    }

    for (size_t d = 0; d < ndim; ++d) {
        const double *gv = gi->gridvals[d];
        size_t n = gi->shape[d];
        double xv = x[d];
        if (xv <= gv[0]) {
            idx_lo[d] = idx_hi[d] = 0;
            t[d] = 0.0;
        } else if (xv >= gv[n - 1]) {
            idx_lo[d] = idx_hi[d] = n - 1;
            t[d] = 0.0;
        } else {
            size_t i = 0;
            while (i + 1 < n && gv[i + 1] < xv) ++i;
            idx_lo[d] = i;
            idx_hi[d] = i + 1;
            double denom = gv[i + 1] - gv[i];
            t[d] = (denom > 0.0) ? (xv - gv[i]) / denom : 0.0;
        }
    }

    for (size_t k = 0; k < n_out; ++k) out[k] = 0.0;

    size_t corners = 1u << ndim;
    for (size_t mask = 0; mask < corners; ++mask) {
        double weight = 1.0;
        size_t flat_index = 0;
        for (size_t d = 0; d < ndim; ++d) {
            size_t idx = (mask & (1u << d)) ? idx_hi[d] : idx_lo[d];
            weight *= (mask & (1u << d)) ? t[d] : (1.0 - t[d]);
            flat_index = flat_index * gi->shape[d] + idx;
        }
        flat_index *= n_out;
        const double *base = gi->data + flat_index;
        for (size_t k = 0; k < n_out; ++k) {
            out[k] += weight * base[k];
        }
    }

    free(idx_lo);
    free(idx_hi);
    free(t);
    return 0;
}
