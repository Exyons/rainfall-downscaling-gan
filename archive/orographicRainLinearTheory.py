import numpy as np

g = 9.80665  # Gravitational Acceleration (m/s^2)
Rd = 287.04  # Gas Constant of dry air (J/kg/K)
Rv = 461.5   # Gas constant of moist air (J/kg/K)
Cp = 1003.5  # Specific heat (J/Kg/K)
Cp_v = 1996  # Specific heat water vapor (J/kg/K)
Lv = 2.26e6  # Latent heat of vaporization [J/kg]
es0 = 611 # Saturation vapour pressure of water at 0C (Pa)
T0 = 273.16  # 0C temperature in K

def find_wavenumber_domain(nx, ny, dx, dy):
    """Given an x-y grid, return the wavenumber domain kx-ky on which an FFT will be computed
    args:
     - nx: number of points in x-direction of grid
     - ny: number of points in y-direction of grid
     - dx: grid spacing in x direction (m)
     - dy: grid spacing in y direction (m)
    returns:
     - kx, ky: wavenumber arrays
    """
    x_n_val = np.fft.fftfreq(nx, (1. / nx))
    y_n_val = np.fft.fftfreq(ny, (1. / ny))
    
    x_len = nx * dx
    y_len = ny * dy
    
    kx_line = 2 * np.pi * x_n_val / x_len
    ky_line = 2 * np.pi * y_n_val / y_len
    kx, ky = np.meshgrid(kx_line, ky_line)
    
    return kx, ky

def find_intrinsic_frequency(u, v, kx, ky):
    """Calculate intrinsic frequency from wind speed and wavenumbers
    args:
     - u: meridional wind speed (m/s), eastward positive
     - v: zonal wind speed (m/s), northward positive
     - kx, ky: wavenumber arrays
    returns:
     - sigma: intrinsic frequency
    """
    # Note: v is -ve to account for the fact that it's northward +ve,
    # while the dem grids are sorted from north to south direction (origin at top-left)
    sigma = kx * u + ky * -v
    return sigma

def find_vertical_wavenumber(kx, ky, sigma, Nm):
    """Calculate the vertical wavenumber in linear mountain wave theory
    args:
     - kx, ky: wavenumber arrays
     - Nm: Brunt-Vaisala frequency
     - sigma: intrinsic frequency
    returns:
     - m, vertical wavenumber array
    """
    EPS = np.finfo(float).eps
    numerator = (Nm**2 - sigma**2) * (kx**2 + ky**2)
    denominator = sigma**2
    sign_sigma = np.where(sigma >= 0, 1, -1)

    # numerical stability
    numerator[numerator < 0] = 0.
    denominator[np.abs(denominator) < EPS] = EPS
    
    m = sign_sigma * np.sqrt(numerator / denominator)
    return m

def find_cw(ts, ps, gamma):
    """Calculate Cw.
    args:
     - ts: surface temperature (K)
     - ps: surface temperature (Pa)
     - gamma: environmental lapse rate (K/m)
    returns:
     - Cw
    """
    es = find_saturation_vapor_pressure(ts)
    gamma_m = find_saturated_moist_adiabatic_lapse_rate(ts, ps)
    Cw = (es * gamma_m) / (Rv * ts * gamma)
    return Cw

def find_moist_layer_penetration_depth(ts, gamma):
    """Calculate moist layer penetration depth.
    args:
     - ts: surface temperature (K)
     - gamma: environmental lapse rate (K/m)
    returns:
     - Hw, moist layer penetartion depth (m)
    """
    Hw = Rv*ts**2/(Lv*gamma)
    return Hw

def find_saturation_vapor_pressure(t):
    """Calculate saturation vapor pressure.
    args:
     - t: temperature (K)
    returns:
     - es, saturation vapor pressure
    """
    es = es0 * np.exp((Lv / Rv) * (1.0/T0 - 1.0/t))
    return es

def find_saturation_mixing_ratio(t, p):
    """Calculate saturation mixing ratio.
    args:
     - t: temperature (K)
     - p: pressure (Pa)
    returns:
     - rs, saturation mixing ratio
    """
    es = find_saturation_vapor_pressure(t)
    rs = (Rd / Rv) * es / (p - es)
    return rs

def find_saturated_moist_adiabatic_lapse_rate(t, p):
    """Calculate saturated moist adiabatic lapse rate.
    args:
     - t: temperature (K)
     - p: pressure (Pa)
    returns:
     - gamma_m, saturated moist adiabatic lapse rate
    """
    # Emanuel 1994, Atmospheric Convection, page 131
    # assuming liquid water mixing ratio = 0
    rs = find_saturation_mixing_ratio(t, p)
    gamma_m = ((g/Cp) * (1 + rs) / (1 + rs * Cp_v / Cp)) * ((1 + (Lv * rs) / (Rd * t)) / (1 + ((Lv**2 * rs * (1 + rs * Rv / Rd)) / (Rv * t**2 * (Cp + rs * Cp_v)))))
    return gamma_m

def find_brunt_vaisala_frequency(ts, ps, gamma):
    """Calculate Brunt-Vaisala frequency.
    args:
     - ts: surface temperature (K)
     - ps: surface pressure (Pa)
     - gamma: environmental lapse rate (K/m)
    returns:
     - Nm, Brunt-Vaisala frequency
    """
    # Smith 2004
    gamma_m = find_saturated_moist_adiabatic_lapse_rate(ts, ps)
    Nm = np.sqrt((g/ts)*(gamma_m-gamma))
    return Nm

def compute_orographic_rain(h, u, v, ts, ps, gamma, tauc, tauf, dx, dy, x_shift=0, y_shift=0):
    ny, nx = np.shape(h)
    kx, ky = find_wavenumber_domain(nx, ny, dx, dy)
    hhat = np.exp(1j*kx*nx*x_shift + 1j*ky*ny*y_shift) * np.fft.fft2(h)
    sigma = find_intrinsic_frequency(u, v, kx, ky)
    Nm = find_brunt_vaisala_frequency(ts, ps, gamma)
    m = find_vertical_wavenumber(kx, ky, sigma, Nm)
    cw = find_cw(ts, ps, gamma)
    hw = find_moist_layer_penetration_depth(ts, gamma)
    phat = ((cw * 1j * sigma * hhat) / ((1 - (hw * m * 1j)) * (1 + (sigma * tauc * 1j)) * (1 + (sigma * tauf * 1j))))
    p = np.real(np.fft.ifft2(phat)) * 3600
    return p

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance (in m) between two points
    on the earth (specified in decimal degrees).    
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    c = 2 * np.arcsin(np.sqrt(np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2)) # angular distance
    return 6371000 * c # angular distance multiplied by radius of the earth