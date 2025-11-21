import numpy as np
import yaml
from tempfile import NamedTemporaryFile
from astropy import constants
from photochem.clima import AdiabatClimate
from photochem.utils import stars
from photochem.utils import species_dict_for_climate, settings_dict_for_climate
import numba as nb
from copy import deepcopy

@nb.njit()
def CO2_henrys_constant(T):
    # from NIST
    alpha = 0.035*np.exp(2400.0*((1/T) - 1/(298.15)))
    return alpha

@nb.njit()
def K2_eq_constant(T):
    # CO2(aq) + H2O <=> H+ + HCO3-
    # p. 123 in Pilson: An Introduction to the Chemistry of the Sea 
    pK2 = 17.788 - .073104 *T - .0051087*35 + 1.1463*10**-4*T**2
    K2 = 10.0**(-pK2)
    return K2

@nb.njit()
def K3_eq_constant(T):
    # HCO3- <=> H+ + CO32-
    # p. 123 in Pilson: An Introduction to the Chemistry of the Sea 
    pK3 = 20.919 - .064209 *T - .011887*35 + 8.7313*10**-5*T**2
    K3 = 10.0**(-pK3)
    return K3

@nb.njit()
def Kspcal_eq_constant(T):
    B0 = -0.77712;
    B1 = 0.0028426;
    B2 = 178.34;
    C0 = -0.07711;
    D0 = 0.0041249;

    T_ocean = T
    SALINITY = 35.0

    logK0 = -171.9065 - (0.077993 * T_ocean) + (2839.319/T_ocean) + (71.595 * np.log10(T_ocean))
    logK = logK0 + (B0 + (B1*T_ocean) + B2/(T_ocean)) * np.sqrt(SALINITY) + C0 * SALINITY + D0 * pow(SALINITY,1.5)

    dSolProd = 10.0**logK

    return dSolProd # mol/kg/atm

@nb.njit()
def aqueous_carbon_chemistry_saturation(T, P_CO2, m_Ca):

    # P_CO2 in bar
    # m_Ca in mol/kg

    # Assume there is always some ocean, and that it is 273 K
    T_ocean = np.maximum(T, 273.0)

    # Compute CO_3^{2-} via Eq. 1 in Schwieterman+2019
    Kspcal = Kspcal_eq_constant(T_ocean)
    Omega_cal = 1.0
    m_CO3 = (Omega_cal*Kspcal)/m_Ca

    # Simple Henry's law
    m_CO2 = P_CO2*CO2_henrys_constant(T_ocean)

    K2 = K2_eq_constant(T_ocean)
    K3 = K3_eq_constant(T_ocean)
    m_HCO3 = np.sqrt((K2*m_CO2*m_CO3)/K3)
    m_H = (K3*m_HCO3)/m_CO3

    return m_CO2, m_HCO3, m_CO3, m_H

@nb.njit()
def aqueous_carbon_chemistry_mass_balance(T, P_CO2, m_Ca, N_H2O, N_CO2):
    """
    Solve the carbonate system by conserving total carbon instead of enforcing calcite saturation.

    Parameters
    ----------
    T : float
        Temperature (K).
    P_CO2 : float
        Atmospheric CO2 partial pressure (bar).
    m_Ca : float
        Ocean Ca2+ concentration (mol/kg), only used to report Omega_cal.
    N_H2O : float
        Column of H2O (mol/cm^2).
    N_CO2 : float
        Total column of CO2 (mol/cm^2) in the atmosphere + ocean.
    """

    # Guard against degenerate cases
    if N_H2O <= 0.0 or N_CO2 <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    T_ocean = np.maximum(T, 273.0)

    # Henry's law for the dissolved CO2(aq)
    m_CO2 = P_CO2*CO2_henrys_constant(T_ocean)

    # Convert global column inventories into mol/kg of ocean
    mu_H2O = 1.00797*2 + 15.9994 # g/mol
    C_total = N_CO2*1e3/(N_H2O*mu_H2O) # mol/kg

    # If there is less carbon than CO2(aq) implied by Henry's law, park everything as CO2(aq)
    if C_total <= m_CO2:
        return C_total, 0.0, 0.0, 0.0, 0.0

    K2 = K2_eq_constant(T_ocean)
    K3 = K3_eq_constant(T_ocean)

    # Solve for m_CO3 that satisfies carbon mass balance:
    # m_CO2 + sqrt((K2*m_CO2*m_CO3)/K3) + m_CO3 = C_total
    x_lo = 0.0
    x_hi = C_total
    m_CO3 = 0.0
    for _ in range(100):
        m_CO3 = 0.5*(x_lo + x_hi)
        m_HCO3 = np.sqrt((K2*m_CO2*m_CO3)/K3)
        f_val = m_CO2 + m_HCO3 + m_CO3 - C_total
        if np.abs(f_val) < 1e-14:
            break
        if f_val > 0.0:
            x_hi = m_CO3
        else:
            x_lo = m_CO3

    m_HCO3 = np.sqrt((K2*m_CO2*m_CO3)/K3)
    m_H = 0.0
    if m_CO3 > 0.0:
        m_H = (K3*m_HCO3)/m_CO3

    # Report the resulting saturation state (<=1 when carbon is scarce)
    Kspcal = Kspcal_eq_constant(T_ocean)
    Omega_cal = 0.0
    if Kspcal > 0.0:
        Omega_cal = (m_CO3*m_Ca)/Kspcal

    return m_CO2, m_HCO3, m_CO3, m_H, Omega_cal

@nb.njit()
def compute_N_CO2(m_CO2, m_HCO3, m_CO3, N_H2O):
    mu_H2O = 1.00797*2 + 15.9994 # g/mol
    N_CO2 = (m_CO2 + m_HCO3 + m_CO3)*N_H2O*(mu_H2O/1e3)
    return N_CO2

@nb.njit()
def aqueous_carbon_chemistry(T, P_CO2, m_Ca, N_H2O, N_CO2):

    rel_tol = 0
    abs_tol = 1e-30
    # Require a small cushion above the inventory before declaring saturation feasible
    N_CO2_thresh = N_CO2 #+ max(rel_tol * N_CO2, abs_tol)

    m_CO2, m_HCO3, m_CO3, m_H = aqueous_carbon_chemistry_saturation(T, P_CO2, m_Ca)

    # Check to see if solution can satisfy mass balance
    N_CO2_1 = compute_N_CO2(m_CO2, m_HCO3, m_CO3, N_H2O)
    if N_CO2_1 < N_CO2_thresh:
        return m_CO2, m_HCO3, m_CO3, m_H, 1.0
    
    # If we are here, then mass balance could not be satisfied
    return aqueous_carbon_chemistry_mass_balance(T, P_CO2, m_Ca, N_H2O, N_CO2)

@nb.cfunc(nb.void(nb.double,nb.int32,nb.types.CPointer(nb.double),nb.types.CPointer(nb.double), nb.types.CPointer(nb.double)))
def water_ocean_solubility_fcn(T_surf, ng, P_i, m_i, p):
    # P_i in bar
    # m_i in mol/kg

    ind_CO2 = int(p[0])
    m_Ca = p[1]
    N_H2O = p[2]
    N_CO2 = p[3]
    
    # zero-out everything
    for i in range(ng):
        m_i[i] = 0.0

    # CO2
    P_CO2 = P_i[ind_CO2]
    m_CO2, m_HCO3, m_CO3, m_H, Omega_cal = aqueous_carbon_chemistry(T_surf, P_CO2, m_Ca, N_H2O, N_CO2)
    m_i[ind_CO2] = m_CO2 + m_HCO3 + m_CO3 # total dissolved carbon

class AdiabatClimateVPL(AdiabatClimate):

    def __init__(self, M_planet=1.0, R_planet=1.0, stellar_flux=1370, Teff=5780.0, nz=50, number_of_zeniths=4,
                 species_file=None, star_file=None, m_Ca=2.45e-03, ocean_chemistry=True, 
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
        self.tol_make_column = 1e-5 # Loosen tolerance to avoid errors

        # Deal with ocean solubility
        self.m_Ca = m_Ca
        if ocean_chemistry:
            self.set_ocean_solubility_fcn('H2O', water_ocean_solubility_fcn)

    def _custom_setup(self, T_surf, stellar_flux, surface_albedo, RH, bond_albedo):
        # Set surface albedo
        self.rad.surface_albedo = np.ones_like(self.rad.surface_albedo)*surface_albedo

        # Set relative humidity
        self.RH = np.ones_like(self.RH)*RH

        # Compute tropopause temperature
        self.T_trop = skin_temperature(stellar_flux, bond_albedo)
        self.T_trop = np.minimum(T_surf-1.0e-5,self.T_trop) # Ensure T_trop < T_surf

        # Set the bolometric stellar flux
        self.rad.set_bolometric_flux(stellar_flux)
    
    def TOA_fluxes_column_custom(self, T_surf, N_i, stellar_flux, surface_albedo, RH, bond_albedo=0.3):
        "Similar to `TOA_fluxes_custom`, but input N_i are total mol/cm^2 in the atmosphere-ocean system."

        # Setup
        self._custom_setup(T_surf, stellar_flux, surface_albedo, RH, bond_albedo)

        # Pass in some important inputs
        indCO2 = self.species_names.index('CO2')
        indH2O = self.species_names.index('H2O')
        self.ocean_args = np.array([indCO2+ 1e-8, self.m_Ca, N_i[indH2O], N_i[indCO2]])
        self.ocean_args_p = self.ocean_args.ctypes.data

        # Compute radiative transfer
        ASR, OLR = self.TOA_fluxes_column(T_surf, N_i)

        # Convert to W/m^2
        OLR /= 1e3
        ASR /= 1e3

        return ASR, OLR

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
