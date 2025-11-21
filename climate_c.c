#include "climate_c.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

ClimateModelC *climate_model_create(const char *filename, const char *dataset_name) {
    ClimateModelC *cm = (ClimateModelC *)calloc(1, sizeof(ClimateModelC));
    if (!cm) return NULL;
    cm->rad_interp = grid_interpolator_load(filename, dataset_name);
    if (!cm->rad_interp) {
        free(cm);
        return NULL;
    }
    return cm;
}

void climate_model_free(ClimateModelC *cm) {
    if (!cm) return;
    grid_interpolator_free(cm->rad_interp);
    free(cm);
}

int climate_model_toa_fluxes(const ClimateModelC *cm,
                             double T_surf, double N_H2O, double N_CO2,
                             double stellar_flux, double surface_albedo,
                             double *ASR, double *OLR) {
    if (!cm || !ASR || !OLR) return -1;
    // Grid uses log10 for H2O and CO2 columns per gridnames in ClimateGrid.
    double logN_H2O = log10(N_H2O);
    double logN_CO2 = log10(N_CO2);
    double x[5] = {T_surf, logN_H2O, logN_CO2, stellar_flux, surface_albedo};
    double out[2] = {0.0, 0.0};
    int rc = grid_interpolate(cm->rad_interp, x, out);
    if (rc != 0) return rc;
    *ASR = out[0];
    *OLR = out[1];
    return 0;
}

int climate_model_surface_temperature(const ClimateModelC *cm,
                                      double N_H2O, double N_CO2,
                                      double stellar_flux, double surface_albedo,
                                      double T_lo, double T_hi, double T_guess,
                                      double tol,
                                      int max_iter, double *T_out) {
    if (!cm || !T_out) return -1;
    // Delegate to the stable finder with default scan/slope settings.
    double scan_step = fmax(0.5, (T_hi - T_lo) / 50.0); // coarse scan
    double slope_delta = 0.25; // K for derivative estimate
    return climate_model_surface_temperature_stable(cm, N_H2O, N_CO2,
                                                    stellar_flux, surface_albedo,
                                                    T_lo, T_hi, T_guess,
                                                    scan_step, slope_delta,
                                                    tol, max_iter, T_out);
}

// Helper: compute net flux ASR-OLR
static int net_flux(const ClimateModelC *cm, double T, double N_H2O, double N_CO2,
                    double stellar_flux, double surface_albedo, double *out) {
    double ASR, OLR;
    int rc = climate_model_toa_fluxes(cm, T, N_H2O, N_CO2, stellar_flux, surface_albedo, &ASR, &OLR);
    if (rc != 0) return rc;
    *out = ASR - OLR;
    return 0;
}

// Brent root on a bracket [a,b]
static int brent_root(const ClimateModelC *cm, double a, double b,
                      double fa, double fb,
                      double N_H2O, double N_CO2,
                      double stellar_flux, double surface_albedo,
                      double tol, int max_iter, double *root) {
    double c = a, fc = fa;
    double d = b - a, e = d;
    for (int iter = 0; iter < max_iter; ++iter) {
        if (fabs(fc) < fabs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }
        double tol_act = 2.0 * DBL_EPSILON * fabs(b) + 0.5 * tol;
        double m = 0.5 * (c - b);
        if (fabs(m) <= tol_act || fb == 0.0) {
            *root = b;
            return 0;
        }
        double p, q;
        if (fabs(e) < tol_act || fabs(fa) <= fabs(fb)) {
            d = m;
            e = m;
        } else {
            double s = fb / fa;
            if (a == c) {
                p = 2.0 * m * s;
                q = 1.0 - s;
            } else {
                double r = fc / fa;
                double t = fb / fc;
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            p = fabs(p);
            if (2.0 * p < fmin(3.0 * m * q - fabs(tol_act * q), fabs(e * q))) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        }
        a = b;
        fa = fb;
        if (fabs(d) > tol_act)
            b += d;
        else
            b += (m > 0 ? tol_act : -tol_act);
        if (net_flux(cm, b, N_H2O, N_CO2, stellar_flux, surface_albedo, &fb) != 0)
            return -1;
        if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
    }
    *root = b;
    return 1; // max iterations reached
}

int climate_model_surface_temperature_stable(const ClimateModelC *cm,
                                             double N_H2O, double N_CO2,
                                             double stellar_flux, double surface_albedo,
                                             double T_lo, double T_hi, double T_guess,
                                             double scan_step, double slope_delta,
                                             double tol, int max_iter, double *T_out) {
    if (!cm || !T_out) return -1;
    double ASR_lo, OLR_lo, ASR_hi, OLR_hi;
    if (climate_model_toa_fluxes(cm, T_lo, N_H2O, N_CO2, stellar_flux, surface_albedo, &ASR_lo, &OLR_lo) != 0)
        return -1;
    if (climate_model_toa_fluxes(cm, T_hi, N_H2O, N_CO2, stellar_flux, surface_albedo, &ASR_hi, &OLR_hi) != 0)
        return -1;

    // Scan for sign changes to identify brackets
    const int max_brackets = 64;
    double brackets_lo[max_brackets];
    double brackets_hi[max_brackets];
    double f_lo_arr[max_brackets];
    double f_hi_arr[max_brackets];
    int num_brackets = 0;

    double t_prev = T_lo;
    double f_prev = ASR_lo - OLR_lo;
    for (double t = T_lo + scan_step; t <= T_hi + 1e-9; t += scan_step) {
        double t_curr = (t > T_hi) ? T_hi : t;
        double f_curr;
        if (net_flux(cm, t_curr, N_H2O, N_CO2, stellar_flux, surface_albedo, &f_curr) != 0)
            return -1;
        if (f_prev == 0.0) {
            // Exact root at t_prev
            if (num_brackets < max_brackets) {
                brackets_lo[num_brackets] = t_prev;
                brackets_hi[num_brackets] = t_prev;
                f_lo_arr[num_brackets] = f_prev;
                f_hi_arr[num_brackets] = f_prev;
                num_brackets++;
            }
        } else if (f_prev * f_curr <= 0.0) {
            if (num_brackets < max_brackets) {
                brackets_lo[num_brackets] = t_prev;
                brackets_hi[num_brackets] = t_curr;
                f_lo_arr[num_brackets] = f_prev;
                f_hi_arr[num_brackets] = f_curr;
                num_brackets++;
            }
        }
        t_prev = t_curr;
        f_prev = f_curr;
        if (t_curr >= T_hi) break;
    }

    if (num_brackets == 0) return -2; // no roots found

    // Solve each bracket, evaluate stability, pick closest stable to T_guess
    double best_root = 0.0;
    double best_dist = 1e300;
    double fallback_root = 0.0;
    double fallback_dist = 1e300;

    for (int i = 0; i < num_brackets; ++i) {
        double r = 0.0;
        double fa = f_lo_arr[i];
        double fb = f_hi_arr[i];
        if (brackets_lo[i] == brackets_hi[i]) {
            r = brackets_lo[i];
        } else {
            if (brent_root(cm, brackets_lo[i], brackets_hi[i], fa, fb,
                           N_H2O, N_CO2, stellar_flux, surface_albedo,
                           tol, max_iter, &r) != 0) {
                continue;
            }
        }

        // Estimate slope at root
        double delta = slope_delta;
        double t_minus = fmax(T_lo, r - delta);
        double t_plus = fmin(T_hi, r + delta);
        double f_minus, f_plus;
        if (net_flux(cm, t_minus, N_H2O, N_CO2, stellar_flux, surface_albedo, &f_minus) != 0) continue;
        if (net_flux(cm, t_plus, N_H2O, N_CO2, stellar_flux, surface_albedo, &f_plus) != 0) continue;
        double slope = (f_plus - f_minus) / (t_plus - t_minus);

        double dist = fabs(r - T_guess);
        if (slope < 0.0) {
            if (dist < best_dist) {
                best_dist = dist;
                best_root = r;
            }
        }
        // Track closest root regardless of stability as fallback
        if (dist < fallback_dist) {
            fallback_dist = dist;
            fallback_root = r;
        }
    }

    if (best_dist < 1e299) {
        *T_out = best_root;
        return 0;
    }
    if (fallback_dist < 1e299) {
        *T_out = fallback_root;
        return 1; // returned unstable root
    }
    return -3;
}
