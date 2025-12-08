#include <stdio.h>
#include <time.h>

#include "climate.h"

int main(void) {
    const char *grid_file = "../ClimateGrid.h5";
    ClimateModel *cm = climate_model_load(grid_file);
    if (!cm) {
        fprintf(stderr, "C Modern-Earth benchmark: failed to load %s\n", grid_file);
        return 1;
    }

    double P_CO2 = 400e-6;         // bar (Modern Earth ~400 ppmv)
    double stellar_flux = 1370.0;  // W/m^2
    double surface_albedo = 0.2;
    double T_bounds[2] = {100.0, 600.0};
    double T_surf = 0.0;

    // Benchmark: run multiple times to estimate average compute time.
    const int n_runs = 100;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int rc = 0;
    for (int i = 0; i < n_runs; ++i) {
        rc = climate_model_surface_temperature(
            cm,
            P_CO2,
            stellar_flux,
            surface_albedo,
            T_bounds,
            300.0,   // T_surf_guess
            20,      // n_intervals
            1e-4,    // tol
            &T_surf
        );
        if (rc != 0) break;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    if (rc != 0) {
        fprintf(stderr, "C Modern-Earth benchmark: surface_temperature solve failed (rc=%d)\n", rc);
        climate_model_free(cm);
        return 1;
    }

    double elapsed_sec = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    double avg_us = (elapsed_sec / n_runs) * 1e6;

    printf(
        "C Modern-Earth benchmark: P_CO2=%.4e bar, stellar_flux=%.1f W/m^2, "
        "surface_albedo=%.3f, T_surf=%.3f K, avg_time=%.3f µs over %d runs\n",
        P_CO2, stellar_flux, surface_albedo, T_surf, avg_us, n_runs
    );

    climate_model_free(cm);
    return 0;
}
