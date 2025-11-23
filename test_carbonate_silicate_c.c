#include "climate_c.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Simple parameterizations for a toy carbonate–silicate cycle integration
static double stellar_flux_at_time(double t_Gyr) {
    // Linear brightening: ~70% at -4 Gyr to 100% today (toy model)
    double f = 0.7 + 0.3 * (1.0 + t_Gyr / 4.0); // t_Gyr in [-4,0]
    double S0 = 1361.0; // present solar constant W/m^2
    return f * S0;
}

static double volcanic_degassing(double t_Gyr) {
    // Constant degassing flux (toy): mol/cm^2/Gyr
    return 1e3;
}

static double silicate_weathering(double P_CO2_bar, double T_surf_K) {
    // Weathering flux scaling with CO2 and temperature (toy)
    // Berger & Kasting-style: F_w ~ (P/P0)^beta * exp[k (T - T0)]
    const double P0 = 3.3e-4; // bar (modern CO2)
    const double T0 = 288.0;  // K
    const double beta = 0.3;
    const double kT = 0.04;   // per K (dampen temperature sensitivity)
    double scale = pow(P_CO2_bar / P0, beta) * exp(kT * (T_surf_K - T0));
    double F0 = 1e3; // mol/cm^2/Gyr reference weathering to balance degassing at modern
    return F0 * scale;
}

int main(void) {
    const char *filename = "../ClimateGrid.h5";
    const char *dataset = "ASR_OLR";

    ClimateModelC *cm = climate_model_create(filename, dataset);
    if (!cm) {
        fprintf(stderr, "Failed to load climate model from %s\n", filename);
        return EXIT_FAILURE;
    }
    GridInterpolator *gi_Psurf = grid_interpolator_load(filename, "P_surf");
    GridInterpolator *gi_fCO2 = grid_interpolator_load(filename, "f_surf_CO2");
    if (!gi_Psurf || !gi_fCO2) {
        fprintf(stderr, "Failed to load P_surf or f_surf_CO2 from %s\n", filename);
        grid_interpolator_free(gi_Psurf);
        grid_interpolator_free(gi_fCO2);
        climate_model_free(cm);
        return EXIT_FAILURE;
    }

    // Fixed water column
    double N_H2O = pow(10.0, 4.1); // mol/cm^2

    // Initial CO2 column at 4 Gyr ago (within grid span)
    double N_CO2 = 50.0; // mol/cm^2

    double surface_albedo = 0.2;

    // Time integration settings
    double t_start = -4.0; // Gyr
    double t_end = 0.0;    // Gyr
    double dt = 0.001;     // Gyr (~1 Myr)
    int steps = (int)((t_end - t_start) / dt);

    // Limit fractional change per step to reduce oscillations
    const double max_frac_change = 0.02; // 2% per step cap

    double t = t_start;
    for (int i = 0; i <= steps; ++i) {
        double stellar_flux = stellar_flux_at_time(t);

        // Compute surface temperature using current CO2 and water
        double T_surf = 0.0;
        int rc = climate_model_surface_temperature(cm, N_H2O, N_CO2, stellar_flux, surface_albedo,
                                                   200.0, 400.0, 280.0, 1e-6, 200, &T_surf);
        if (rc != 0) {
            fprintf(stderr, "surface_temperature failed at t=%.3f Gyr (rc=%d)\n", t, rc);
            break;
        }

        // Interpolate surface pressure and CO2 mixing ratio to get P_CO2
        double x_interp[5] = {T_surf, log10(N_H2O), log10(N_CO2), stellar_flux, surface_albedo};
        double Psurf_out[1] = {0.0};
        double fCO2_out[1] = {0.0};
        int rcP = grid_interpolate(gi_Psurf, x_interp, Psurf_out);
        int rcF = grid_interpolate(gi_fCO2, x_interp, fCO2_out);
        if (rcP != 0 || rcF != 0) {
            fprintf(stderr, "Interpolation failed for P_surf/f_surf_CO2 at t=%.3f Gyr\n", t);
            break;
        }
        double P_CO2_bar = fCO2_out[0] * Psurf_out[0] * 1e-6; // dyn/cm^2 to bar

        double F_degas = volcanic_degassing(t);              // mol/cm^2/Gyr
        double F_weather = silicate_weathering(P_CO2_bar, T_surf); // mol/cm^2/Gyr

        double dNdt = F_degas - F_weather;
        double dN = dNdt * dt;
        // Cap fractional change to avoid large oscillations
        double max_dN = max_frac_change * N_CO2;
        if (dN > max_dN) dN = max_dN;
        if (dN < -max_dN) dN = -max_dN;

        N_CO2 += dN;
        if (N_CO2 < 1e-12) N_CO2 = 1e-12;

        if (i % 50 == 0 || i == steps) {
            printf("t=%.2f Gyr, N_CO2=%.3e mol/cm^2, P_CO2=%.3e bar (interp), T=%.2f K, F_degas=%.2e, F_weather=%.2e\n",
                   t, N_CO2, P_CO2_bar, T_surf, F_degas, F_weather);
        }

        t += dt;
    }

    grid_interpolator_free(gi_Psurf);
    grid_interpolator_free(gi_fCO2);
    climate_model_free(cm);
    return EXIT_SUCCESS;
}
