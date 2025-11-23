#include "climate_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void) {
    const char *filename = "../ClimateGrid.h5";
    const char *dataset = "ASR_OLR";

    ClimateModelC *cm = climate_model_create(filename, dataset);
    if (!cm) {
        fprintf(stderr, "Failed to load climate model from %s\n", filename);
        return EXIT_FAILURE;
    }

    // Modern Earth-like inputs (total inventories atmosphere + ocean; tune as needed)
    double N_H2O = pow(10.0, 4.1); // mol/cm^2
    double N_CO2 = 3.0;            // mol/cm^2
    double stellar_flux = 1361.0;  // W/m^2
    double surface_albedo = 0.3;

    double T_out = 0.0;
    int rc = climate_model_surface_temperature(cm, N_H2O, N_CO2, stellar_flux, surface_albedo,
                                               200.0, 400.0, 300.0, 1e-6, 100, &T_out);
    if (rc != 0) {
        fprintf(stderr, "surface_temperature failed (rc=%d)\n", rc);
        climate_model_free(cm);
        return EXIT_FAILURE;
    }

    // Time multiple calls to surface_temperature
    const int n_runs = 10;
    double total_us = 0.0;
    double T_out_timed = 0.0;
    for (int i = 0; i < n_runs; ++i) {
        clock_t start = clock();
        rc = climate_model_surface_temperature(cm, N_H2O, N_CO2, stellar_flux, surface_albedo,
                                               200.0, 400.0, 300.0, 1e-6, 100, &T_out_timed);
        clock_t end = clock();
        if (rc != 0) {
            fprintf(stderr, "timed surface_temperature failed (rc=%d)\n", rc);
            climate_model_free(cm);
            return EXIT_FAILURE;
        }
        double elapsed_us = 1e6 * (double)(end - start) / CLOCKS_PER_SEC;
        total_us += elapsed_us;
    }

    double avg_us = total_us / n_runs;

    printf("Surface temperature: %.6f K\n", T_out);
    printf("Timed run (avg of %d): %.6f K in %.3f microseconds\n", n_runs, T_out_timed, avg_us);

    // Compute P_CO2 from grid (P_surf * f_surf_CO2)
    GridInterpolator *gi_Psurf = grid_interpolator_load(filename, "P_surf");
    GridInterpolator *gi_fCO2 = grid_interpolator_load(filename, "f_surf_CO2");
    if (gi_Psurf && gi_fCO2) {
        double x_interp[5] = {T_out, log10(N_H2O), log10(N_CO2), stellar_flux, surface_albedo};
        double Psurf_out[1] = {0.0};
        double fCO2_out[1] = {0.0};
        int rcP = grid_interpolate(gi_Psurf, x_interp, Psurf_out);
        int rcF = grid_interpolate(gi_fCO2, x_interp, fCO2_out);
        if (rcP == 0 && rcF == 0) {
            double P_CO2_bar = fCO2_out[0] * Psurf_out[0] * 1e-6;
            printf("Interpolated P_CO2: %.6e bar (P_surf=%.3e dyn/cm^2, f_CO2=%.3e)\n",
                   P_CO2_bar, Psurf_out[0], fCO2_out[0]);
        } else {
            fprintf(stderr, "Interpolation for P_surf/f_surf_CO2 failed\n");
        }
    } else {
        fprintf(stderr, "Could not load P_surf/f_surf_CO2 interpolators\n");
    }
    grid_interpolator_free(gi_Psurf);
    grid_interpolator_free(gi_fCO2);
    climate_model_free(cm);
    return EXIT_SUCCESS;
}
