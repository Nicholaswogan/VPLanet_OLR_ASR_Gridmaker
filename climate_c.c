#include "climate_c.h"
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
    double x[5] = {T_surf, N_H2O, N_CO2, stellar_flux, surface_albedo};
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
    double f_lo = ASR_lo - OLR_lo;
    double f_hi = ASR_hi - OLR_hi;
    if (f_lo * f_hi > 0.0) {
        return -2; // no sign change
    }

    double a = T_lo;
    double b = T_hi;
    for (int iter = 0; iter < max_iter; ++iter) {
        double m = 0.5 * (a + b);
        double ASR_m, OLR_m;
        if (climate_model_toa_fluxes(cm, m, N_H2O, N_CO2, stellar_flux, surface_albedo, &ASR_m, &OLR_m) != 0)
            return -1;
        double f_m = ASR_m - OLR_m;
        if (fabs(f_m) < tol || 0.5 * fabs(b - a) < tol) {
            *T_out = m;
            return 0;
        }
        if (f_lo * f_m <= 0.0) {
            b = m;
            f_hi = f_m;
        } else {
            a = m;
            f_lo = f_m;
        }
    }
    *T_out = 0.5 * (a + b);
    return 1; // max iterations reached
}
