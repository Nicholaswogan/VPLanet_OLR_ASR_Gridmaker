// C analogue of ClimateModel using GridInterpolator.
#ifndef CLIMATE_C_H
#define CLIMATE_C_H

#include "gridutils_c.h"

typedef struct {
    GridInterpolator *rad_interp;
} ClimateModelC;

/*
 * Load a climate interpolation model from an HDF5 grid.
 *
 * filename      Path to the HDF5 grid file (e.g., "ClimateGrid.h5").
 * dataset_name  Dataset under /results to use (e.g., "ASR_OLR").
 *
 * Returns: allocated ClimateModelC on success, NULL on failure.
 */
ClimateModelC *climate_model_create(const char *filename, const char *dataset_name);

/*
 * Free all resources owned by a ClimateModelC. Safe to call with NULL.
 *
 * cm  Model pointer created by climate_model_create.
 */
void climate_model_free(ClimateModelC *cm);

/*
 * Compute TOA fluxes for a given state.
 *
 * cm              Model handle.
 * T_surf          Surface temperature [K].
 * N_H2O           Total H2O column [mol/cm^2].
 * N_CO2           Total CO2 column [mol/cm^2].
 * stellar_flux    Bolometric stellar flux at planet [W/m^2].
 * surface_albedo  Surface albedo [unitless].
 * ASR             Output absorbed solar radiation [W/m^2].
 * OLR             Output outgoing longwave radiation [W/m^2].
 *
 * Returns 0 on success, non-zero on interpolation failure.
 */
int climate_model_toa_fluxes(const ClimateModelC *cm,
                             double T_surf, double N_H2O, double N_CO2,
                             double stellar_flux, double surface_albedo,
                             double *ASR, double *OLR);

/*
 * Solve for surface temperature where ASR == OLR near a guess.
 *
 * cm              Model handle.
 * N_H2O           Total H2O column [mol/cm^2].
 * N_CO2           Total CO2 column [mol/cm^2].
 * stellar_flux    Bolometric stellar flux at planet [W/m^2].
 * surface_albedo  Surface albedo [unitless].
 * T_lo, T_hi      Temperature bracket [K] for scanning/root finding.
 * T_guess         Preferred temperature [K] near which to choose a root.
 * tol             Absolute tolerance for the root and flux residual.
 * max_iter        Maximum iterations for the root finder.
 * T_out           Output root temperature [K].
 *
 * Returns 0 on success, non-zero on failure.
 */
int climate_model_surface_temperature(const ClimateModelC *cm,
                                      double N_H2O, double N_CO2,
                                      double stellar_flux, double surface_albedo,
                                      double T_lo, double T_hi, double T_guess,
                                      double tol,
                                      int max_iter, double *T_out);

/*
 * Find a (stable) equilibrium near a guess. Scans for all roots, filters by negative slope, picks closest to T_guess.
 *
 * cm              Model handle.
 * N_H2O           Total H2O column [mol/cm^2].
 * N_CO2           Total CO2 column [mol/cm^2].
 * stellar_flux    Bolometric stellar flux at planet [W/m^2].
 * surface_albedo  Surface albedo [unitless].
 * T_lo, T_hi      Temperature bounds [K] for scanning.
 * T_guess         Target temperature [K] to select the nearest root.
 * scan_step       Step size [K] for initial sign-change scan.
 * slope_delta     Temperature perturbation [K] for finite-difference slope check.
 * tol             Absolute tolerance for root solving.
 * max_iter        Maximum iterations per root solve.
 * T_out           Output root temperature [K] (stable if available).
 *
 * Returns 0 for stable root, 1 if only an unstable root was returned, negative on failure.
 */
int climate_model_surface_temperature_stable(const ClimateModelC *cm,
                                             double N_H2O, double N_CO2,
                                             double stellar_flux, double surface_albedo,
                                             double T_lo, double T_hi, double T_guess,
                                             double scan_step, double slope_delta,
                                             double tol, int max_iter, double *T_out);

#endif // CLIMATE_C_H
