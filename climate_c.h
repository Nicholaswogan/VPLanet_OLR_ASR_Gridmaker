// C analogue of ClimateModel using GridInterpolator.
#ifndef CLIMATE_C_H
#define CLIMATE_C_H

#include "gridutils_c.h"

typedef struct {
    GridInterpolator *rad_interp;
} ClimateModelC;

// Load the model using the given HDF5 file and dataset (e.g., "ASR_OLR").
ClimateModelC *climate_model_create(const char *filename, const char *dataset_name);
void climate_model_free(ClimateModelC *cm);

// Compute TOA fluxes; outputs ASR and OLR.
int climate_model_toa_fluxes(const ClimateModelC *cm,
                             double T_surf, double N_H2O, double N_CO2,
                             double stellar_flux, double surface_albedo,
                             double *ASR, double *OLR);

// Solve for surface temperature where ASR == OLR using bisection.
// Returns 0 on success; T_out receives the root.
int climate_model_surface_temperature(const ClimateModelC *cm,
                                      double N_H2O, double N_CO2,
                                      double stellar_flux, double surface_albedo,
                                      double T_lo, double T_hi, double T_guess,
                                      double tol,
                                      int max_iter, double *T_out);

// Stable equilibrium near a guess: scans for all roots, filters for negative slope, chooses closest to T_guess.
int climate_model_surface_temperature_stable(const ClimateModelC *cm,
                                             double N_H2O, double N_CO2,
                                             double stellar_flux, double surface_albedo,
                                             double T_lo, double T_hi, double T_guess,
                                             double scan_step, double slope_delta,
                                             double tol, int max_iter, double *T_out);

#endif // CLIMATE_C_H
