#include <stdio.h>
#include <time.h>
#include "climate.h"

typedef struct {
    int success;        // 0 on success, non-zero on failure
    double T_surf;       // Surface temperature [K]
    double f_H2O_surf;   // Surface mixing ratio of H2O [unitless]
    double f_N2_surf;    // Surface mixing ratio of N2 [unitless]
    double f_CO2_surf;   // Surface mixing ratio of CO2 [unitless]
    double P_surf;       // Surface pressure [dyne/cm^2]
} ClimateResult;

// Compute a climate state from CO2 partial pressure and forcing, returning surface properties.
//
// Inputs:
//   cm             Initialized ClimateModel pointer
//   pco2_pa        CO2 partial pressure [Pa]
//   stellar_flux   Bolometric stellar flux at planet [W/m^2]
//   surface_albedo Surface albedo [unitless]
//   T_surf_guess   Initial guess for surface temperature [K]
//
// Output:
//   ClimateResult populated on success; success != 0 indicates failure.
ClimateResult vplanet_climate(
    ClimateModel *cm,
    double pco2_pa,
    double stellar_flux,
    double surface_albedo,
    double T_surf_guess
){
    double P_CO2 = pco2_pa/1.0e5; // Pa to bar.
    double T_bounds[2] = {100.0, 600.0};
    double T_surf = 0.0;

    int rc = climate_model_surface_temperature(
        cm,
        P_CO2,
        stellar_flux,
        surface_albedo,
        T_bounds,
        T_surf_guess,   // T_surf_guess
        20,      // n_intervals
        1.0e-4,    // tol
        &T_surf
    );

    ClimateResult result = {rc, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (result.success != 0){
        return result;
    }

    // If successful we save results directly interpolated from the grid.
    double P_surf = 0.0, f_H2O = 0.0, f_N2 = 0.0, f_CO2 = 0.0;
    int rc_state = climate_model_surface_state(
        cm,
        T_surf,
        P_CO2,
        stellar_flux,
        surface_albedo,
        &P_surf,
        &f_H2O,
        &f_N2,
        &f_CO2
    );
    if (rc_state != 0) {
        result.success = rc_state;
        return result;
    }

    result.T_surf = T_surf;
    result.f_H2O_surf = (double)f_H2O;
    result.f_N2_surf = (double)f_N2;
    result.f_CO2_surf = (double)f_CO2;
    result.P_surf = (double)P_surf;

    return result;
}

int main(void) {
    const char *grid_file = "ClimateGrid.h5";
    ClimateModel *cm = climate_model_load(grid_file);
    if (!cm) {
        fprintf(stderr, "Failed to load %s\n", grid_file);
        return 1;
    }

    const int n_runs = 1000;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    ClimateResult res = {0};
    for (int i = 0; i < n_runs; ++i) {
        res = vplanet_climate(
            cm,
            40.0,     // pco2_pa
            1370.0,   // stellar_flux
            0.2,      // surface_albedo
            300.0     // T_surf_guess
        );
        if (res.success != 0) {
            break;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    if (res.success != 0) {
        fprintf(stderr, "vplanet_climate failed (rc=%d)\n", res.success);
        climate_model_free(cm);
        return 1;
    }

    double elapsed_sec = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    double avg_us = (elapsed_sec / n_runs) * 1e6;

    printf("T_surf=%.3f K, P_surf=%.6e dyne/cm^2, f_H2O=%.6e, f_N2=%.6e, f_CO2=%.6e\n",
           res.T_surf, res.P_surf, res.f_H2O_surf, res.f_N2_surf, res.f_CO2_surf);
    printf("Average runtime over %d runs: %.3f µs\n", n_runs, avg_us);

    climate_model_free(cm);
    return 0;
}
