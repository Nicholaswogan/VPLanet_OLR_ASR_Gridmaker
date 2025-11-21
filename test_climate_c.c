#include "climate_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
                                               150.0, 450.0, 1e-6, 100, &T_out);
    if (rc != 0) {
        fprintf(stderr, "surface_temperature failed (rc=%d)\n", rc);
        climate_model_free(cm);
        return EXIT_FAILURE;
    }

    printf("Surface temperature: %.6f K\n", T_out);
    climate_model_free(cm);
    return EXIT_SUCCESS;
}
