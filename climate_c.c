#include "climate_c.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

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
                                      double T_lo, double T_hi, double tol,
                                      int max_iter, double *T_out) {
    if (!cm || !T_out) return -1;
    double ASR_lo, OLR_lo, ASR_hi, OLR_hi;
    if (climate_model_toa_fluxes(cm, T_lo, N_H2O, N_CO2, stellar_flux, surface_albedo, &ASR_lo, &OLR_lo) != 0)
        return -1;
    if (climate_model_toa_fluxes(cm, T_hi, N_H2O, N_CO2, stellar_flux, surface_albedo, &ASR_hi, &OLR_hi) != 0)
        return -1;
    double a = T_lo, fa = ASR_lo - OLR_lo;
    double b = T_hi, fb = ASR_hi - OLR_hi;
    if (fa * fb > 0.0) return -2; // no sign change

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
            *T_out = b;
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
        if (climate_model_toa_fluxes(cm, b, N_H2O, N_CO2, stellar_flux, surface_albedo, &ASR_hi, &OLR_hi) != 0)
            return -1;
        fb = ASR_hi - OLR_hi;
        if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
    }
    *T_out = b;
    return 1; // max iterations reached
}
