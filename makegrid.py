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

    T_surf, log10P_CO2, stellar_flux, surface_albedo = x
    c = CLIMATE_MODEL

    P_i = np.ones(len(c.species_names))*1e-15
    P_i[c.species_names.index('H2O')] = 270 # Assume 1 ocean of H2O
    P_i[c.species_names.index('N2')] = 1.0 # Assume 1 bar N2
    P_i[c.species_names.index('CO2')] = 10.0**log10P_CO2
    P_i *= 1e6 # bar to dynes/cm^2

    try:
        ASR, OLR = c.TOA_fluxes_custom(
            T_surf=T_surf, 
            P_i=P_i, 
            stellar_flux=stellar_flux, 
            surface_albedo=surface_albedo, 
            RH=1.0, 
            bond_albedo=0.3
        )
    except ClimaException as e:
        print('FAILED',list(x))
        raise ClimaException(e)

    res = make_result(c, x, ASR, OLR)

    return res

def make_result(c, x, ASR, OLR):

    # Atmospheric properties as a function of altitude
    P = np.append(c.P_surf,c.P)
    T = np.append(c.T_surf,c.T)
    z = np.append(0, c.z)
    f_i = np.concatenate((c.f_i_surf.reshape((1,len(c.f_i_surf))),c.f_i))

    # Reservoirs
    N_atmos = c.N_atmos
    N_surface = c.N_surface

    # Save results as 32 bit floats
    result = {}
    # Inputs
    result['x'] = x.astype(np.float32)
    # Surface Properties
    result['P_surf'] = np.array(c.P_surf,np.float32)
    result['T_surf'] = np.array(c.T_surf,np.float32)
    for i,sp in enumerate(c.species_names):
        result['f_surf_'+sp] = np.array(c.f_i_surf[i],np.float32)
    # TOA properties
    result['P_toa'] = np.array(P[-1],np.float32)
    result['T_toa'] = np.array(T[-1],np.float32)
    for i,sp in enumerate(c.species_names):
        result['f_toa_'+sp] = f_i[-1,i].astype(np.float32)
    # Reservoirs
    for i,sp in enumerate(c.species_names):
        result['N_atmos_'+sp] = np.array(N_atmos[i],np.float32)
    for i,sp in enumerate(c.species_names):
        result['N_surface_'+sp] = np.array(N_surface[i],np.float32)
    # RT
    result['ASR_OLR'] = np.array([ASR, OLR], np.float32)

    return result

def get_gridvals():
    T_surf = np.arange(100.0, 600.1, 10.0)
    log10P_CO2 = np.arange(-6.0, 2.01, 0.5)
    stellar_flux = np.arange(300.0, 1500.01, 100.0)
    surface_albedo = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    gridvals = (T_surf, log10P_CO2, stellar_flux, surface_albedo)
    gridnames = ['T_surf','log10P_CO2','stellar_flux','surface_albedo']
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