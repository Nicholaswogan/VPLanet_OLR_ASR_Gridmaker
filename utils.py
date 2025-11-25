import numpy as np
import yaml
from tempfile import NamedTemporaryFile
from astropy import constants
from photochem.clima import AdiabatClimate
from photochem.utils import stars
from photochem.utils import species_dict_for_climate, settings_dict_for_climate
import gridutils
from scipy import optimize

def skin_temperature(stellar_flux, bond_albedo):
    return stars.equilibrium_temperature(stellar_flux, bond_albedo)*(1/2)**(1/4)

def blackbody_spectrum_at_planet(stellar_flux, Teff, nw):

    # Blackbody
    wv_planet = np.logspace(np.log10(0.1), np.log10(100), nw)*1e3 # nm
    F_planet = stars.blackbody(Teff, wv_planet)*np.pi

    # Rescale so that it has the proper stellar flux for the planet
    factor = stellar_flux/stars.energy_in_spectrum(wv_planet, F_planet)
    F_planet *= factor

    return wv_planet, F_planet

class AdiabatClimateVPL(AdiabatClimate):

    def __init__(self, M_planet=1.0, R_planet=1.0, stellar_flux=1370, Teff=5780.0, nz=50, number_of_zeniths=4,
                 species_file=None, star_file=None, 
                 data_dir=None, double_radiative_grid=False):
        """Initializes the code. 

        Parameters
        ----------
        M_planet : float
            Mass of the planet in Earth masses.
        R_planet : float
            Radius of the planet in Earth radii.
        stellar_flux : float
            Bolometric flux at planet in W/m^2.
        Teff : float, optional
            Stellar effective temperature in K.
        nz : int, optional
            Number of vertical layers in the climate model, by default 50
        number_of_zeniths : int, optional
            Number of zenith angles in the radiative transfer calculation, by default 1
        species_file : str, optional
            Path to a settings file. If None, then a default file is used.
        m_Ca : float, optional
            Ocean Ca2+ concentration in mol/kg
        data_dir : str, optional
            Path to where climate model data is stored. If None, then installed data is used.
        double_radiative_grid : bool, optional
            If True, then this doubles the radiative grid which is needed for full RCE calculations, by default False
        
        """ 
        
        # Species file
        if species_file is None:
            species_dict = species_dict_for_climate(
                species=['H2O','CO2','N2'], 
                condensates=['H2O','CO2'], 
                particles=None
            )
        else:
            with open(species_file,'r') as f:
                species_dict = yaml.load(f, Loader=yaml.Loader)

        settings_dict = settings_dict_for_climate(
            planet_mass=float(M_planet*constants.M_earth.to('g').value), 
            planet_radius=float(R_planet*constants.R_earth.to('cm').value), 
            surface_albedo=0.0, 
            number_of_layers=int(nz), 
            number_of_zenith_angles=int(number_of_zeniths), 
            photon_scale_factor=1.0
        )

        if star_file is None:
            wv_planet, F_planet = blackbody_spectrum_at_planet(stellar_flux, Teff, nw=5000)
        else:
            wv_planet, F_planet = np.loadtxt(star_file, skiprows=1).T
        # Load the flux at the planet to a string
        flux_str = stars.photochem_spectrum_string(wv_planet, F_planet, scale_to_planet=False)

        with NamedTemporaryFile('w') as f_species:
            # Write species file
            yaml.safe_dump(species_dict, f_species)
            with NamedTemporaryFile('w') as f_settings:
                # Write settings file
                yaml.safe_dump(settings_dict, f_settings)
                with NamedTemporaryFile('w') as f_flux:
                    # Write stellar flux file
                    f_flux.write(flux_str)
                    f_flux.flush()
                    
                    # Initialize AdiabatClimate
                    super().__init__(
                        f_species.name, 
                        f_settings.name, 
                        f_flux.name,
                        data_dir=data_dir,
                        double_radiative_grid=double_radiative_grid
                    )

        # Ensure the stellar flux is right
        self.rad.set_bolometric_flux(stellar_flux)

        # Change default parameters
        self.max_rc_iters = 30 # Lots of iterations
        self.P_top = 10.0 # 10 dynes/cm^2 top, or 1e-5 bars.

    def _custom_setup(self, T_surf, stellar_flux, surface_albedo, RH, bond_albedo):
        # Set surface albedo
        self.rad.surface_albedo = np.ones_like(self.rad.surface_albedo)*surface_albedo

        # Set relative humidity
        self.RH = np.ones_like(self.RH)*RH

        # Compute tropopause temperature
        self.T_trop = skin_temperature(stellar_flux, bond_albedo)
        self.T_trop = np.minimum(T_surf-1.0e-5, self.T_trop) # Ensure T_trop < T_surf

        # Set the bolometric stellar flux
        self.rad.set_bolometric_flux(stellar_flux)
    
    def TOA_fluxes_column_custom(self, T_surf, N_i, stellar_flux, surface_albedo, RH, bond_albedo=0.3):
        """
        Compute top-of-atmosphere fluxes using column inventories (mol/cm^2). Volatiles are allowed to
        go into the atmosphere or condense onto the surface (i.e. a H2O ocean). Volatiles do not
        dissolve into a surface ocean.

        Parameters
        ----------
        T_surf : float
            Surface temperature [K].
        N_i : ndarray
            Column abundances for each species [mol/cm^2]
        stellar_flux : float
            Bolometric stellar flux at the planet [W/m^2].
        surface_albedo : float
            Surface albedo [unitless].
        RH : float
            Relative humidity [0-1] used to set the atmospheric profile.
        bond_albedo : float, optional
            Bond albedo for the tropopause temperature estimate, by default 0.3.

        Returns
        -------
        ASR : float
            Absorbed solar radiation [W/m^2].
        OLR : float
            Outgoing longwave radiation [W/m^2].
        """

        # Setup
        self._custom_setup(T_surf, stellar_flux, surface_albedo, RH, bond_albedo)

        # Compute radiative transfer
        ASR, OLR = self.TOA_fluxes_column(T_surf, N_i)

        # Convert to W/m^2
        OLR /= 1e3
        ASR /= 1e3

        return ASR, OLR
    
    def surface_temperature_column_custom(self, N_i, stellar_flux, surface_albedo, RH, bond_albedo=0.3,
                                          T_bounds=(200.0, 400.0), T_surf_guess=None, n_intervals=20):
        """
        Find the stable surface temperature closest to a target guess.

        Parameters
        ----------
        N_i : ndarray
            Column abundances for each species [mol/cm^2].
        stellar_flux : float
            Bolometric stellar flux at the planet [W/m^2].
        surface_albedo : float
            Surface albedo [unitless].
        RH : float
            Relative humidity [0-1] used to set the atmospheric profile.
        bond_albedo : float, optional
            Bond albedo for the tropopause temperature estimate, by default 0.3.
        T_bounds : tuple, optional
            Temperature bracket to search over (K), by default (200.0, 400.0).
        T_surf_guess : float, optional
            Preferred equilibrium temperature. The stable root closest to this
            value is returned.
        n_intervals : int, optional
            Number of intervals used when scanning for sign changes, by default 20.

        Returns
        -------
        float
            Stable surface temperature (K) closest to `T_surf_guess`.
        """

        def net_flux(T_surf):
            ASR, OLR = self.TOA_fluxes_column_custom(T_surf, N_i, stellar_flux, surface_albedo, RH, bond_albedo)
            return ASR - OLR

        def is_stable(T_eq, eps=0.5):
            """Stability: d(net_flux)/dT < 0 implies restoring tendency."""
            delta_low = max(T_bounds[0], T_eq - eps)
            delta_high = min(T_bounds[1], T_eq + eps)
            if delta_high == delta_low:
                return False
            f_low = net_flux(delta_low)
            f_high = net_flux(delta_high)
            deriv = (f_high - f_low) / (delta_high - delta_low)
            return deriv < 0.0

        guess = T_surf_guess if T_surf_guess is not None else 0.5 * (T_bounds[0] + T_bounds[1])

        # Scan for all sign changes to capture multiple equilibria
        scan_T = np.linspace(T_bounds[0], T_bounds[1], int(n_intervals) + 1)
        scan_flux = [net_flux(T) for T in scan_T]

        candidate_brackets = []
        for i in range(len(scan_T) - 1):
            f0, f1 = scan_flux[i], scan_flux[i + 1]
            if f0 == 0.0:
                # Exact zero on grid; treat as tiny bracket
                candidate_brackets.append((scan_T[i], scan_T[i]))
            elif f0 * f1 < 0:
                candidate_brackets.append((scan_T[i], scan_T[i + 1]))

        roots = []
        for a, b in candidate_brackets:
            if a == b:
                root = a
                converged = True
            else:
                sol = optimize.root_scalar(net_flux, bracket=(a, b), method='brentq')
                root = sol.root
                converged = sol.converged
            if converged and T_bounds[0] <= root <= T_bounds[1] and is_stable(root):
                roots.append(root)

        if not roots:
            raise RuntimeError(f"surface_temperature did not find a stable root within {T_bounds}")

        # Return the stable equilibrium closest to the requested temperature
        return min(roots, key=lambda r: abs(r - guess))


class ClimateModel:

    def __init__(self, filename):
        self.g = gridutils.GridInterpolator(filename)
        self.rad_interp = self.g.make_interpolator('ASR_OLR')

    def TOA_fluxes(self, T_surf, N_CO2, stellar_flux, surface_albedo):
        x = np.array([T_surf, np.log10(N_CO2), stellar_flux, surface_albedo])
        ASR, OLR = self.rad_interp(x)
        return ASR, OLR
    
    def surface_temperature(self, N_CO2, stellar_flux, surface_albedo,
                           T_bounds=(200.0, 400.0), T_surf_guess=None, n_intervals=20):
        """
        Solve for surface temperature where TOA absorbed solar equals outgoing longwave,
        returning the stable equilibrium closest to a target guess.

        Parameters
        ----------
        N_CO2 : float
            Total CO2 column (mol/cm^2).
        stellar_flux : float
            Bolometric stellar flux at the planet (W/m^2).
        surface_albedo : float
            Surface albedo.
        T_bounds : tuple, optional
            Temperature bracket to search over (K), by default (200.0, 400.0).
        T_surf_guess : float, optional
            Preferred equilibrium temperature. The stable root closest to this
            value is returned.
        n_intervals : int, optional
            Number of intervals used when scanning for sign changes, by default 20.

        Returns
        -------
        float
            Stable surface temperature (K) closest to `T_surf_guess`.
        """

        def net_flux(T_surf):
            ASR, OLR = self.TOA_fluxes(T_surf, N_CO2, stellar_flux, surface_albedo)
            return ASR - OLR

        def is_stable(T_eq, eps=0.5):
            """Stability: d(net_flux)/dT < 0 implies restoring tendency."""
            delta_low = max(T_bounds[0], T_eq - eps)
            delta_high = min(T_bounds[1], T_eq + eps)
            if delta_high == delta_low:
                return False
            f_low = net_flux(delta_low)
            f_high = net_flux(delta_high)
            deriv = (f_high - f_low) / (delta_high - delta_low)
            return deriv < 0.0

        guess = T_surf_guess if T_surf_guess is not None else 0.5 * (T_bounds[0] + T_bounds[1])

        # Scan for all sign changes to capture multiple equilibria
        scan_T = np.linspace(T_bounds[0], T_bounds[1], int(n_intervals) + 1)
        scan_flux = [net_flux(T) for T in scan_T]

        candidate_brackets = []
        for i in range(len(scan_T) - 1):
            f0, f1 = scan_flux[i], scan_flux[i + 1]
            if f0 == 0.0:
                candidate_brackets.append((scan_T[i], scan_T[i]))
            elif f0 * f1 < 0:
                candidate_brackets.append((scan_T[i], scan_T[i + 1]))

        roots = []
        for a, b in candidate_brackets:
            if a == b:
                root = a
                converged = True
            else:
                sol = optimize.root_scalar(net_flux, bracket=(a, b), method='brentq')
                root = sol.root
                converged = sol.converged
            if converged and T_bounds[0] <= root <= T_bounds[1] and is_stable(root):
                roots.append(root)

        if not roots:
            raise RuntimeError(f"surface_temperature did not find a stable root within {T_bounds}")

        return min(roots, key=lambda r: abs(r - guess))

def test_ClimateGrid():
    c = ClimateModel('ClimateGrid.h5')
    N_CO2 = 23*400e-6
    stellar_flux = 1370
    surface_albedo = 0.2

    T_surf = c.surface_temperature(
        N_CO2=N_CO2,
        stellar_flux=stellar_flux,
        surface_albedo=surface_albedo
    )
    print(T_surf)

if __name__ == '__main__':
    test_ClimateGrid()
