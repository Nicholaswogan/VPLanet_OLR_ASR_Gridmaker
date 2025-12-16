#include <math.h>
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

    double P_surf = 0.0, f_H2O = 0.0, f_N2 = 0.0, f_CO2 = 0.0;
    rc = climate_model_surface_state(
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
    if (rc != 0) {
        fprintf(stderr, "Failed to interpolate surface state (rc=%d)\n", rc);
        climate_model_free(cm);
        return 1;
    }

    printf(
        "Surface state: P_surf=%.6e dyne/cm^2, f_H2O=%.6e, f_N2=%.6e, f_CO2=%.6e\n",
        P_surf, f_H2O, f_N2, f_CO2
    );

    // Cloud albedo conversion test: verify that using Pa for pH2O produces
    // consistent temperatures with a fixed-albedo solve using the same effective albedo.
    double ground_albedo = 0.1;
    double opacity_scale = 3000.0;           // Pa; tune as needed (Driscoll & Bercovici ~1785 Pa)
    double scattering_gamma = 0.998;
    double beta_cloud = 0.000471169;
    double albedo_cloud = 0.89;

    ClimateAlbedoOptions cloud_opts = climate_albedo_options_cloud(
        ground_albedo, opacity_scale, scattering_gamma, beta_cloud, albedo_cloud);

    double T_cloud = 0.0;
    double albedo_cloud_eff = 0.0;
    rc = climate_model_surface_temperature_with_albedo(
        cm,
        P_CO2,
        stellar_flux,
        &cloud_opts,
        T_bounds,
        300.0,
        20,
        1e-4,
        &T_cloud,
        &albedo_cloud_eff);
    if (rc != 0) {
        fprintf(stderr, "Cloud-albedo temperature solve failed (rc=%d)\n", rc);
        climate_model_free(cm);
        return 1;
    }

    // Recompute pH2O at the cloud-equilibrium temperature to keep the fixed-albedo
    // comparison consistent with the state used in the cloud run.
    double P_surf_cloud = 0.0;
    double f_H2O_cloud = 0.0;
    rc = climate_model_surface_state(
        cm,
        T_cloud,
        P_CO2,
        stellar_flux,
        albedo_cloud_eff,
        &P_surf_cloud,
        &f_H2O_cloud,
        NULL,
        NULL
    );
    if (rc != 0) {
        fprintf(stderr, "Failed to interpolate surface state at cloud equilibrium (rc=%d)\n", rc);
        climate_model_free(cm);
        return 1;
    }
    double pH2O_pa_cloud = f_H2O_cloud * P_surf_cloud * 0.1;

    double albedo_expected = cloud_albedo(
        T_cloud, pH2O_pa_cloud, ground_albedo, opacity_scale, scattering_gamma, beta_cloud,
        albedo_cloud);
    if (fabs(albedo_expected - albedo_cloud_eff) > 1e-6) {
        fprintf(stderr,
                "Cloud albedo output mismatch: effective=%.9f, expected=%.9f\n",
                albedo_cloud_eff, albedo_expected);
        climate_model_free(cm);
        return 1;
    }
    ClimateAlbedoOptions fixed_opts = climate_albedo_options_fixed(albedo_expected);

    double T_fixed = 0.0;
    rc = climate_model_surface_temperature_with_albedo(
        cm,
        P_CO2,
        stellar_flux,
        &fixed_opts,
        T_bounds,
        300.0,
        20,
        1e-4,
        &T_fixed,
        NULL);
    if (rc != 0) {
        fprintf(stderr, "Fixed-albedo temperature solve failed (rc=%d)\n", rc);
        climate_model_free(cm);
        return 1;
    }

    double delta_T = fabs(T_cloud - T_fixed);
    if (delta_T > 1.0) {
        fprintf(stderr,
                "Cloud albedo conversion test failed: T_cloud=%.3f K, T_fixed=%.3f K, |delta|=%.3f K\n",
                T_cloud, T_fixed, delta_T);
        climate_model_free(cm);
        return 1;
    }
    printf("Cloud albedo conversion test passed: T_cloud=%.3f K, T_fixed=%.3f K, |delta|=%.3f K\n",
           T_cloud, T_fixed, delta_T);
    printf("Cloud albedos: effective=%.6f, expected=%.6f\n",
           albedo_cloud_eff, albedo_expected);

    climate_model_free(cm);
    return 0;
}
