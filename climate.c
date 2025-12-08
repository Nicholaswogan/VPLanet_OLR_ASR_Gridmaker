#include "climate.h"

#include <float.h>
#include <hdf5.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

// --- GridInterpolator implementation (formerly in gridutils.c) ---

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

    // Read gridvals to know the grid dimensionality
    hid_t g_grid = H5Gopen(file_id, "gridvals", H5P_DEFAULT);
    if (g_grid < 0) {
        H5Sclose(space);
        H5Dclose(dset);
        H5Fclose(file_id);
        return NULL;
    }
    H5G_info_t ginfo;
    if (H5Gget_info(g_grid, &ginfo) < 0) {
        H5Gclose(g_grid);
        H5Sclose(space);
        H5Dclose(dset);
        H5Fclose(file_id);
        return NULL;
    }
    size_t grid_ndim = (size_t)ginfo.nlinks;
    H5Gclose(g_grid);

    size_t ndim;
    size_t n_out;
    if ((size_t)rank == grid_ndim) {
        // Dataset matches grid shape -> scalar output
        ndim = (size_t)rank;
        n_out = 1;
    } else {
        // Assume last axis is output
        ndim = (size_t)(rank - 1);
        n_out = (size_t)dims[rank - 1];
    }

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

    // strides[d] = product of shape[d+1..ndim-1] * n_out (C-order flattening)
    size_t *strides = (size_t *)malloc(ndim * sizeof(size_t));
    if (!strides) return -1;
    size_t acc = n_out;
    for (ssize_t d = (ssize_t)ndim - 1; d >= 0; --d) {
        strides[d] = acc;
        acc *= gi->shape[d];
    }

    // Find bounding indices and weights in each dimension
    size_t *idx_lo = (size_t *)malloc(ndim * sizeof(size_t));
    size_t *idx_hi = (size_t *)malloc(ndim * sizeof(size_t));
    double *t = (double *)malloc(ndim * sizeof(double));
    if (!idx_lo || !idx_hi || !t) {
        free(idx_lo); free(idx_hi); free(t); free(strides);
        return -1;
    }

    for (size_t d = 0; d < ndim; ++d) {
        const double *gv = gi->gridvals[d];
        size_t n = gi->shape[d];
        double xv = x[d];
        if (n < 2) {
            idx_lo[d] = idx_hi[d] = 0;
            t[d] = 0.0;
            continue;
        }

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
            int use_hi = (mask & (1u << d)) != 0;
            size_t idx = use_hi ? idx_hi[d] : idx_lo[d];
            weight *= use_hi ? t[d] : (1.0 - t[d]);
            flat_index += idx * strides[d];
        }
        const double *base = gi->data + flat_index;
        for (size_t k = 0; k < n_out; ++k) {
            out[k] += weight * base[k];
        }
    }

    free(strides);
    free(idx_lo);
    free(idx_hi);
    free(t);
    return 0;
}

// --- ClimateModel implementation ---

typedef struct {
    const ClimateModel *cm;
    double P_CO2;
    double stellar_flux;
    double surface_albedo;
} NetFluxCtx;

static int net_flux(const NetFluxCtx *ctx, double T_surf, double *out_flux) {
    double ASR = 0.0, OLR = 0.0;
    if (climate_model_toa_fluxes(ctx->cm, T_surf, ctx->P_CO2, ctx->stellar_flux,
                                 ctx->surface_albedo, &ASR, &OLR) != 0) {
        return -1;
    }
    *out_flux = ASR - OLR;
    return 0;
}

static int is_stable(const NetFluxCtx *ctx, double T_eq, const double T_bounds[2], double eps) {
    double low = T_eq - eps;
    double high = T_eq + eps;
    if (low < T_bounds[0]) low = T_bounds[0];
    if (high > T_bounds[1]) high = T_bounds[1];
    if (high <= low) return 0;

    double f_low = 0.0, f_high = 0.0;
    if (net_flux(ctx, low, &f_low) != 0 || net_flux(ctx, high, &f_high) != 0) return 0;
    double deriv = (f_high - f_low) / (high - low);
    return deriv < 0.0;
}

// Brent-Dekker (brentq) root finder on a sign-changing bracket.
static int brentq_solve(const NetFluxCtx *ctx, double a, double b, double fa, double fb,
                        double tol, size_t max_iter, double *root_out) {
    if (fa * fb > 0.0) return -1;
    double c = a, fc = fa;
    double d = b - a, e = d;
    for (size_t iter = 0; iter < max_iter; ++iter) {
        if (fb * fc > 0.0) {
            c = a;
            fc = fa;
            d = e = b - a;
        }
        if (fabs(fc) < fabs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }

        double tol1 = 2.0 * DBL_EPSILON * fabs(b) + 0.5 * tol;
        double xm = 0.5 * (c - b);
        if (fabs(xm) <= tol1 || fb == 0.0) {
            *root_out = b;
            return 0;
        }

        if (fabs(e) >= tol1 && fabs(fa) > fabs(fb)) {
            double s = fb / fa;
            double p, q;
            if (a == c) {
                // Secant method
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                // Inverse quadratic interpolation
                double q1 = fa / fc;
                double r = fb / fc;
                p = s * (2.0 * xm * q1 * (q1 - r) - (b - a) * (r - 1.0));
                q = (q1 - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            p = fabs(p);

            if (2.0 * p < fmin(3.0 * xm * q - fabs(tol1 * q), fabs(e * q))) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }

        a = b;
        fa = fb;
        if (fabs(d) > tol1) {
            b += d;
        } else {
            b += (xm > 0.0 ? tol1 : -tol1);
        }
        if (net_flux(ctx, b, &fb) != 0) return -1;
    }
    *root_out = b;
    return 0;
}

ClimateModel *climate_model_load(const char *filename) {
    ClimateModel *cm = (ClimateModel *)calloc(1, sizeof(ClimateModel));
    if (!cm) return NULL;

    cm->rad_interp = grid_interpolator_load(filename, "ASR_OLR");
    if (!cm->rad_interp) {
        free(cm);
        return NULL;
    }
    return cm;
}

void climate_model_free(ClimateModel *cm) {
    if (!cm) return;
    grid_interpolator_free(cm->rad_interp);
    free(cm);
}

int climate_model_toa_fluxes(const ClimateModel *cm,
                             double T_surf,
                             double P_CO2,
                             double stellar_flux,
                             double surface_albedo,
                             double *ASR,
                             double *OLR) {
    if (!cm || !cm->rad_interp || !ASR || !OLR) return -1;
    if (!isfinite(P_CO2) || P_CO2 <= 0.0) return -1;
    double x[4];
    x[0] = T_surf;
    x[1] = log10(P_CO2);
    x[2] = stellar_flux;
    x[3] = surface_albedo;

    double out[2] = {0.0, 0.0};
    if (grid_interpolate(cm->rad_interp, x, out) != 0) return -1;
    *ASR = out[0];
    *OLR = out[1];
    return 0;
}

int climate_model_surface_temperature(const ClimateModel *cm,
                                      double P_CO2,
                                      double stellar_flux,
                                      double surface_albedo,
                                      const double T_bounds[2],
                                      double T_surf_guess,
                                      size_t n_intervals,
                                      double tol,
                                      double *T_out) {
    if (!cm || !T_bounds || !T_out || n_intervals < 1) return -1;
    double t_min = T_bounds[0];
    double t_max = T_bounds[1];
    if (t_max <= t_min) return -1;
    NetFluxCtx ctx = {cm, P_CO2, stellar_flux, surface_albedo};

    double guess = T_surf_guess;
    if (!isfinite(guess)) guess = 0.5 * (t_min + t_max);
    if (guess < t_min) guess = t_min;
    if (guess > t_max) guess = t_max;

    double step = (t_max - t_min) / (double)n_intervals;
    size_t max_brackets = n_intervals + 1;
    double *roots = (double *)malloc(max_brackets * sizeof(double));
    size_t root_count = 0;

    double prev_T = t_min;
    double prev_f = 0.0;
    if (net_flux(&ctx, prev_T, &prev_f) != 0) {
        free(roots);
        return -1;
    }

    for (size_t i = 0; i < n_intervals; ++i) {
        double curr_T = (i == n_intervals - 1) ? t_max : (t_min + (i + 1) * step);
        double curr_f = 0.0;
        if (net_flux(&ctx, curr_T, &curr_f) != 0) {
            free(roots);
            return -1;
        }

        int sign_change = (prev_f == 0.0) || (curr_f == 0.0) || (prev_f * curr_f < 0.0);
        if (sign_change) {
            double root_val = curr_T;
            if (prev_f == 0.0) {
                root_val = prev_T;
            } else if (curr_f == 0.0) {
                root_val = curr_T;
            } else {
                if (brentq_solve(&ctx, prev_T, curr_T, prev_f, curr_f, tol, 100, &root_val) != 0) {
                    free(roots);
                    return -1;
                }
            }

            if (is_stable(&ctx, root_val, T_bounds, 0.5)) {
                if (root_count < max_brackets) {
                    roots[root_count++] = root_val;
                }
            }
        }

        prev_T = curr_T;
        prev_f = curr_f;
    }

    if (root_count == 0) {
        free(roots);
        return -1;
    }

    double best_root = roots[0];
    double best_dist = fabs(best_root - guess);
    for (size_t i = 1; i < root_count; ++i) {
        double dist = fabs(roots[i] - guess);
        if (dist < best_dist) {
            best_dist = dist;
            best_root = roots[i];
        }
    }

    free(roots);
    *T_out = best_root;
    return 0;
}
