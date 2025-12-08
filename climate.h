// C interface for the ClimateModel previously implemented in utils.py.
// Provides ASR/OLR interpolation and a stable surface-temperature solver.

#ifndef CLIMATE_H
#define CLIMATE_H

#include <stddef.h>

#include "gridutils.h"

typedef struct {
    GridInterpolator *rad_interp;
} ClimateModel;

/*
 * Load the climate model using the HDF5 grid file produced by makegrid.py.
 *
 * filename: path to the HDF5 file (e.g., "ClimateGrid.h5").
 *
 * Returns: allocated ClimateModel on success, NULL on failure.
 */
ClimateModel *climate_model_load(const char *filename);

/*
 * Free a ClimateModel (safe on NULL).
 */
void climate_model_free(ClimateModel *cm);

/*
 * Interpolate top-of-atmosphere fluxes.
 *
 * Inputs mirror the Python version:
 *   T_surf         Surface temperature [K]
 *   P_CO2          CO2 partial pressure [bar]
 *   stellar_flux   Bolometric stellar flux at planet [W/m^2]
 *   surface_albedo Surface albedo [unitless]
 *
 * Outputs:
 *   ASR, OLR       Absorbed solar / outgoing longwave [W/m^2]
 *
 * Returns 0 on success, non-zero on failure.
 */
int climate_model_toa_fluxes(const ClimateModel *cm,
                             double T_surf,
                             double P_CO2,
                             double stellar_flux,
                             double surface_albedo,
                             double *ASR,
                             double *OLR);

/*
 * Solve for the stable surface temperature closest to T_surf_guess.
 *
 * Parameters:
 *   T_bounds[2]   Search bounds (K), e.g., {200.0, 400.0}
 *   T_surf_guess  Preferred equilibrium temperature (K)
 *   n_intervals   Number of intervals for scanning sign changes
 *   tol           Convergence tolerance for root finder (abs flux)
 *
 * Output:
 *   T_out         Stable temperature on success
 *
 * Returns 0 on success, non-zero on failure (no stable root or interpolation error).
 */
int climate_model_surface_temperature(const ClimateModel *cm,
                                      double P_CO2,
                                      double stellar_flux,
                                      double surface_albedo,
                                      const double T_bounds[2],
                                      double T_surf_guess,
                                      size_t n_intervals,
                                      double tol,
                                      double *T_out);

#endif  // CLIMATE_H
