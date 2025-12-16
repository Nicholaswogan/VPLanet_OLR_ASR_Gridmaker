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

typedef enum {
    CLIMATE_ALBEDO_FIXED = 0,
    CLIMATE_ALBEDO_CLOUD = 1
} ClimateAlbedoMode;

typedef struct {
    ClimateAlbedoMode mode;
    double ground_albedo;
    double opacity_press_h2o_cloud;
    double scattering_gamma;
    double beta_cloud;
    double albedo_cloud;
} ClimateAlbedoOptions;

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
 *   albedo_out    Effective planetary albedo at the equilibrium temperature (optional)
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

/*
 * Extended surface-temperature solver that allows configurable albedo modes.
 *
 * albedo_opts  Configuration for how to compute the albedo. If NULL, defaults
 *              to fixed albedo of 0.0 (use the fixed-wrapper to set a value).
 */
int climate_model_surface_temperature_with_albedo(const ClimateModel *cm,
                                                  double P_CO2,
                                                  double stellar_flux,
                                                  const ClimateAlbedoOptions *albedo_opts,
                                                  const double T_bounds[2],
                                                  double T_surf_guess,
                                                  size_t n_intervals,
                                                  double tol,
                                                  double *T_out,
                                                  double *albedo_out);

/*
 * Helpers to create common albedo option presets.
 */
ClimateAlbedoOptions climate_albedo_options_fixed(double surface_albedo);
ClimateAlbedoOptions climate_albedo_options_cloud(double ground_albedo,
                                                  double opacity_press_h2o_cloud,
                                                  double scattering_gamma,
                                                  double beta_cloud,
                                                  double albedo_cloud);

/*
 * Compute total (cloud-modified) planetary albedo.
 *
 * This routine combines a simple cloud opacity parameterization with
 * a two-stream-like reflectance formula to compute the effective
 * planetary albedo due to partial cloud cover.
 *
 * Inputs:
 *   T_surf                  Surface temperature [K]
 *                           (included for extensibility; not used directly)
 *   pH2O                    H2O partial pressure [units consistent with
 *                           opacity_press_h2o_cloud]
 *   albedo_ground            Clear-sky / ground albedo [unitless]
 *   opacity_press_h2o_cloud  H2O partial pressure scale for cloud opacity
 *   scattering_gamma         Cloud single-scattering parameter [unitless]
 *   beta_cloud               Cloud extinction efficiency parameter [unitless]
 *   albedo_cloud             Cloud albedo in the optically thick limit
 *
 * Returns:
 *   Total (cloud-modified) planetary albedo [unitless]
 */
double cloud_albedo(double T_surf,
                    double pH2O,
                    double albedo_ground,
                    double opacity_press_h2o_cloud,
                    double scattering_gamma,
                    double beta_cloud,
                    double albedo_cloud);

#endif  // CLIMATE_H
