import warnings
warnings.filterwarnings('ignore')

from gridutils import make_grid
import numpy as np
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

import utils
from photochem.clima import ClimaException

# Initialize the climate model
CLIMATE_MODEL = utils.AdiabatClimateVPL()

def model(x):

    T_surf, log10N_H2O, log10N_CO2, stellar_flux, surface_albedo = x
    c = CLIMATE_MODEL

    N_i = np.ones(len(c.species_names))*1e-15
    N_i[c.species_names.index('H2O')] = 10.0**log10N_H2O
    N_i[c.species_names.index('N2')] = 36.0 # mol/cm^2. About 1 bar N2
    N_i[c.species_names.index('CO2')] = 10.0**log10N_CO2

    try:
        ASR, OLR = c.TOA_fluxes_column_custom(
            T_surf=T_surf, 
            N_i=N_i, 
            stellar_flux=stellar_flux, 
            surface_albedo=surface_albedo, 
            RH=1.0, 
            bond_albedo=0.3
        )
    except ClimaException as e:
        print('FAILED',list(x))
        raise ClimaException(e)

    res = make_result(c, x, ASR, OLR, 10.0**log10N_H2O, 10.0**log10N_CO2)

    return res

def make_result(c, x, ASR, OLR, N_H2O, N_CO2):

    # Ocean chemistry
    indCO2 = c.species_names.index('CO2')
    P_CO2 = (c.P_surf*c.f_i_surf[indCO2])/1e6
    m_CO2, m_HCO3, m_CO3, m_H, Omega_cal = utils.aqueous_carbon_chemistry(c.T_surf, P_CO2, c.m_Ca, N_H2O, N_CO2)
    pH = np.log10(m_H) # pH
    DIC = m_CO2 + m_HCO3 + m_CO3 # mol/kg

    # Atmospheric properties as a function of altitude
    P = np.append(c.P_surf,c.P)
    T = np.append(c.T_surf,c.T)
    z = np.append(0, c.z)
    f_i = np.concatenate((c.f_i_surf.reshape((1,len(c.f_i_surf))),c.f_i))

    # Reservoirs
    N_atmos = c.N_atmos
    N_surface = c.N_surface
    N_ocean_CO2 = c.N_ocean[indCO2,c.species_names.index('H2O')] # mol/cm^2 in ocean

    # Save results as 32 bit floats
    result = {}
    # Inputs
    result['x'] = x.astype(np.float32)
    # Profiles
    result['P'] = P.astype(np.float32)
    result['z'] = z.astype(np.float32)
    result['T'] = T.astype(np.float32)
    for i,sp in enumerate(c.species_names):
        result['f_'+sp] = f_i[:,i].astype(np.float32)
    # Surface Properties
    result['P_surf'] = np.array(c.P_surf,np.float32)
    result['T_surf'] = np.array(c.T_surf,np.float32)
    for i,sp in enumerate(c.species_names):
        result['f_surf_'+sp] = np.array(c.f_i_surf[i],np.float32)
    # Reservoirs
    for i,sp in enumerate(c.species_names):
        result['N_atmos_'+sp] = np.array(N_atmos[i],np.float32)
    for i,sp in enumerate(c.species_names):
        result['N_surface_'+sp] = np.array(N_surface[i],np.float32)
    result['N_ocean_CO2'] = np.array(N_ocean_CO2, np.float32)
    # Ocean information
    result['ocean_pH'] = np.array(pH, np.float32)
    result['ocean_DIC'] = np.array(DIC, np.float32)
    result['ocean_m_CO2'] = np.array(m_CO2, np.float32)
    result['ocean_m_HCO3'] = np.array(m_HCO3, np.float32)
    result['ocean_m_CO3'] = np.array(m_CO3, np.float32)
    result['ocean_Omega_cal'] = np.array(Omega_cal, np.float32)
    # RT
    result['ASR_OLR'] = np.array([ASR, OLR], np.float32)

    return result

def get_gridvals():
    T_surf = np.arange(200.0, 400.0, 10.0)
    log10N_H2O = np.arange(3.7, 4.501, 0.4)
    log10N_CO2 = np.arange(-2.0, 2.01, 0.25)
    stellar_flux = np.arange(800.0, 1500.01, 100.0)
    surface_albedo = np.arange(0.0, 0.401, 0.2)
    gridvals = (T_surf, log10N_H2O, log10N_CO2, stellar_flux, surface_albedo)
    gridnames = ['T_surf','log10N_H2O','log10N_CO2','stellar_flux','surface_albedo']
    return gridvals, gridnames

if __name__ == "__main__":
    # mpiexec -np X python filename.py

    gridvals, gridnames = get_gridvals()
    make_grid(
        model_func=model, 
        gridvals=gridvals,
        gridnames=gridnames, 
        filename='ClimateGrid.h5', 
        progress_filename='ClimateGrid.log'
    )