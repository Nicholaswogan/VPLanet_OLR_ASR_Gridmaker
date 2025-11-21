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

    double N_H2O = pow(10.0, 4.1); // 1e4.1 mol/cm^2
    double N_CO2 = 1.0;
    double stellar_flux = 1300.0;
    double surface_albedo = 0.2;

    double T_out = 0.0;
    int rc = climate_model_surface_temperature(cm, N_H2O, N_CO2, stellar_flux, surface_albedo,
                                               200.0, 400.0, 1e-6, 100, &T_out);
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
                                               200.0, 400.0, 1e-6, 100, &T_out_timed);
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
    climate_model_free(cm);
    return EXIT_SUCCESS;
}
