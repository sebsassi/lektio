import numpy as np
from .astro_parameters import OrbitalParameters

class EarthOrbit(OrbitalParameters):
    """
    Parmeters of Earth's orbit.

    Attributes
    ----------
    epoch : str
        Label of the epoch at which the parameters are defined.
    e_semimajor_axis : float
        Semimajor axis of the orbit at the epoch in kilometers.
    e_eccentricity : float
        Eccentricity of the orbit at the epoch.
    e_mean_anomaly : float
        Mean anomaly of the orbit at the epoch.
    e_mean_speed : float
        Mean speed of the orbit at the epoch in radians per day.
    e_ecliptic_longitude : float
        Ecliptic longitude of the vernal equinox at the epoch.
    longitude_coeff : numpy.ndarray
        Coefficients of the polynomial defining the time evolution of
        the ecliptic longitude of the vernal equinox.
    e_obliquity : float
        Obliquity of the planet's axis at the epoch.
    obliquity_coeff : numpy.ndarray
        Coefficients of the polynomial defining the time evolution of
        the obliquity.
    """
    def __init__(self):
        e_semimajor_axis = 1.49598022961e+8
        e_eccentricity = 0.01671022
        e_mean_anomaly = -0.0433337328
        e_mean_speed = 2*np.pi/365.256363004
        e_ecliptic_longitude = -1.344825
        e_obliquity = 0.40909260
        
        obliquity_coeff = np.array([-2.227088e-4, 2.9e-9, 8.790e-9])
        longitude_coeff = np.array([2.43802956e-2, -5.38691e-6, -2.9e-11])

        super.__init__(
                'J2000', e_semimajor_axis, e_eccentricity, e_mean_anomaly,
                e_mean_speed, e_ecliptic_longitude, longitude_coeff,
                e_obliquity, obliquity_coeff)
