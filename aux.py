# 定义最小二乘拟合的边界以及初值
import numpy as np
from models import *

def initial_params(model, dont_fit_five_params=False):  
    initial_five_params = [
        0, # d_ra guess (deg)
        0, # d_dec guess (deg)
        0, # pmra guess (mas/yr)
        0, # pmdec guess (mas/yr)
        100] # parallax guess (mas)
    
    initial_campbell_parameters = [
        100, # period guess (days)
        0.5, # ecc guess
        50, # semi_maj guess (mas)
        np.radians(60), # incl guess (rad)
        np.radians(30), # arg_peri guess (rad)
        np.radians(40), # Omega guess (rad)
        0 # m_0 guess (rad)
    ]
    initial_ti_parameters = [
        100, # period guess (days)
        0.5, # ecc guess
        0, # M0 guess (rad)
        0, # A (mas)
        0, # B (mas)
        0, # F (mas)
        0 # G guess (mas)
    ]
    residual_to_initial = {
        single_star_model: initial_five_params,
        no_plx_model: initial_five_params[:-1], # 不拟合视差
        orbit_model_campbell: initial_five_params + initial_campbell_parameters,
        orbit_model_ti: initial_five_params + initial_ti_parameters,
    }
    
    if dont_fit_five_params is False:
        return residual_to_initial.get(model, initial_five_params + initial_campbell_parameters)
    else:
        return residual_to_initial.get(model, initial_five_params + initial_campbell_parameters)[5:]


def get_bounds(MODEL):
    lower_bounds = [
        -np.inf, -np.inf, -np.inf, -np.inf, 0, # d_ra, d_dec, pmra, pmdec, parallax
        0.01, 0, 0.01, 0, 0, 0, 0 # period, ecc, semi_maj, incl, arg_peri, omega, m_0
    ]
    upper_bounds = [
        np.inf, np.inf, np.inf, np.inf, np.inf, # d_ra, d_dec, pmra, pmdec, parallax
        np.inf,         # period
        0.99,           # ecc
        np.inf,         #semi_maj
        2*np.pi,        # incl
        2*np.pi,        # arg_peri
        2*np.pi,        # omega
        2*np.pi,        # m_0
    ]

    lower_bounds_ti = [
        -np.inf, -np.inf, -np.inf, -np.inf, 0, # d_ra, d_dec, pmra, pmdec, parallax
        0.01, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf # period, ecc, M0, A, B, F, G
    ]
    
    upper_bounds_ti = [
        np.inf, np.inf, np.inf, np.inf, np.inf, # d_ra, d_dec, pmra, pmdec, parallax
        np.inf,         # period
        0.99,           # ecc
        2*np.pi,       # M0
        np.inf,         # A
        np.inf,         # B
        np.inf,         # F
        np.inf,         # G
    ]
    
    model_to_bounds = {
        single_star_model: (lower_bounds[:5], upper_bounds[:5]),
        no_plx_model: (lower_bounds[:4], upper_bounds[:4]),
        orbit_model_campbell: (lower_bounds, upper_bounds),  
        orbit_model_ti: (lower_bounds_ti, upper_bounds_ti),
    }

    return model_to_bounds.get(MODEL)


def get_param_names(MODEL, dont_fit_five_params=False):
    five_param_names = ['ra_com', 'dec_com', 'pmra_com', 'pmdec_com', 'plx']
    campbell_params_names = ['period', 'ecc', 'semi_maj', 'incl', 'arg_peri', 'omega', 'M0']
    ti_params_names = ['period', 'ecc', 'M0', 'A', 'B', 'F', 'G']
    
    residual_to_names = {
        single_star_model: five_param_names,
        no_plx_model: five_param_names[:-1],
        orbit_model_ti: five_param_names + ti_params_names,
        orbit_model_campbell: five_param_names + campbell_params_names,
    }
    if dont_fit_five_params is False:
        return residual_to_names.get(MODEL)
    else:
        return residual_to_names.get(MODEL)[5:]