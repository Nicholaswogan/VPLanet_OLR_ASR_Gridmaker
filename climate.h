// C interface for the ClimateModel previously implemented in utils.py.
// Provides ASR/OLR interpolation and a stable surface-temperature solver.

#ifndef CLIMATE_H
#define CLIMATE_H

#include <stddef.h>

typedef struct {
    size_t ndim;        // number of grid dimensions
    size_t n_out;       // length of output vector per grid point
    size_t *shape;      // length ndim
    double **gridvals;  // gridvals[d][i]
    double *data;       // flattened results array
} GridInterpolator;

/*
 * Load a GridInterpolator from an HDF5 file.
 *
 * filename      Path to the HDF5 grid file.
 * dataset_name  Dataset under group "results" to load (e.g., "ASR_OLR").
 *
 * Returns: allocated GridInterpolator on success, NULL on failure.
 */
GridInterpolator *grid_interpolator_load(const char *filename, const char *dataset_name);

/*
 * Free all memory associated with a GridInterpolator (safe on NULL).
 *
 * gi  Pointer returned by grid_interpolator_load.
 */
void grid_interpolator_free(GridInterpolator *gi);

/*
 * Multilinear interpolation at a point.
 *
 * gi   Interpolator handle.
 * x    Input coordinates (length ndim).
 * out  Output array (length n_out) for the interpolated values.
 *
 * Returns 0 on success, non-zero on failure.
 */
int grid_interpolate(const GridInterpolator *gi, const double *x, double *out);

typedef struct {
    GridInterpolator *rad_interp;
    GridInterpolator *p_surf_interp;
    GridInterpolator *f_h2o_interp;
    GridInterpolator *f_n2_interp;
    GridInterpolator *f_co2_interp;
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
 * Interpolate surface state (pressure and mixing ratios) from the grid.
 *
 * Inputs mirror the Python version:
 *   T_surf         Surface temperature [K]
 *   P_CO2          CO2 partial pressure [bar]
 *   stellar_flux   Bolometric stellar flux at planet [W/m^2]
 *   surface_albedo Surface albedo [unitless]
 *
 * Outputs (all optional if NULL):
 *   P_surf         Surface pressure [dyne/cm^2]
 *   f_H2O          Surface mixing ratio of H2O
 *   f_N2           Surface mixing ratio of N2
 *   f_CO2          Surface mixing ratio of CO2
 *
 * Returns 0 on success, non-zero on failure.
 */
int climate_model_surface_state(const ClimateModel *cm,
                                double T_surf,
                                double P_CO2,
                                double stellar_flux,
                                double surface_albedo,
                                double *P_surf,
                                double *f_H2O,
                                double *f_N2,
                                double *f_CO2);

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
