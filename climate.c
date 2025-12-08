#include "climate.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

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

// Simple bisection solver on a sign-changing bracket.
static int bisection_solve(const NetFluxCtx *ctx, double a, double b, double fa, double fb,
                           double tol, size_t max_iter, double *root_out) {
    if (fa * fb > 0.0) return -1;
    double left = a;
    double right = b;
    double f_left = fa;
    double f_right = fb;
    for (size_t iter = 0; iter < max_iter; ++iter) {
        double mid = 0.5 * (left + right);
        double f_mid = 0.0;
        if (net_flux(ctx, mid, &f_mid) != 0) return -1;
        if (fabs(f_mid) < tol || fabs(right - left) < tol) {
            *root_out = mid;
            return 0;
        }
        if (f_left * f_mid <= 0.0) {
            right = mid;
            f_right = f_mid;
        } else {
            left = mid;
            f_left = f_mid;
        }
    }
    *root_out = 0.5 * (left + right);
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
                if (bisection_solve(&ctx, prev_T, curr_T, prev_f, curr_f, tol, 100, &root_val) != 0) {
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
