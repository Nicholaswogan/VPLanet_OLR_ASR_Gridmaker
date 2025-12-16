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
    double surface_albedo;
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
    double ground_albedo,
    double opacity_scale,
    double scattering_gamma,
    double beta_cloud,
    double albedo_cloud,
    double T_surf_guess
){
    double P_CO2 = pco2_pa/1.0e5; // Pa to bar.
    double T_bounds[2] = {100.0, 600.0};

    ClimateAlbedoOptions cloud_opts = climate_albedo_options_cloud(
        ground_albedo, opacity_scale, scattering_gamma, beta_cloud, albedo_cloud);

    double T_surf = 0.0;
    double surface_albedo = 0.0;
    int rc = climate_model_surface_temperature_with_albedo(
        cm,
        P_CO2,
        stellar_flux,
        &cloud_opts,
        T_bounds,
        T_surf_guess,   // T_surf_guess
        20,      // n_intervals
        1.0e-4,    // tol
        &T_surf,
        &surface_albedo
    );

    ClimateResult result = {rc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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
    result.f_H2O_surf = f_H2O;
    result.f_N2_surf = f_N2;
    result.f_CO2_surf = f_CO2;
    result.P_surf = P_surf;
    result.surface_albedo = surface_albedo;

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
            40.0,        // pco2_pa
            1370.0,      // stellar_flux
            0.1,         // ground_albedo
            3000.0,      // opacity_scale,
            0.998,       // scattering_gamma,
            0.000471169, // beta_cloud,
            0.89,        // albedo_cloud,
            300.0        // T_surf_guess
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

    printf("T_surf=%.3f K, P_surf=%.6e dyne/cm^2, f_H2O=%.6e, f_N2=%.6e, f_CO2=%.6e, albedo=%.2f\n",
           res.T_surf, res.P_surf, res.f_H2O_surf, res.f_N2_surf, res.f_CO2_surf, res.surface_albedo);
    printf("Average runtime over %d runs: %.3f µs\n", n_runs, avg_us);

    climate_model_free(cm);
    return 0;
}
