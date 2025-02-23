import numpy as np


def kepler_solve(M: np.ndarray | float, e: float, N: int):
    cond = int(e > 0.8)
    E = cond*np.pi + (1 - cond)*M
    for i in range(N):
        E -= (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
    return E


def EA_at_UT1(UT1: np.ndarray | float, M0: float, mean_speed: float, e: float):
    return kepler_solve(mean_speed*UT1 + M0, e, 7)


def TA_from_EA(EA: np.ndarray | float, e: float):
    cos_EA = np.cos(EA)
    return np.arctan2(
            np.sqrt((1 - e)*(1 + e))*np.sin(EA)/(1 - e*cos_EA),
            (cos_EA - e)/(1 - e*cos_EA))


class GalacticParameters:
    """
    Parameters defining the galactic coordinate system.
    
    Defines the galactic coordinate system relative to the equatorial
    coordinate system using the right ascension and declination of the
    north galactic pole (NGP) as well as the galactic longitude of the
    north celestial pole (NCP).
    
    Attributes
    ----------
    epoch : str
        Label of the epoch at which the parameters are defined.
    NGP_ra : float
        Right ascension of the north galactic pole at the epoch in
        radians.
    NGP_dec : float
        Declination of the north galactic pole at the epoch in radians.
    NCP_lon : float
        Galactic longitude of the north celestial pole at the epoch in
        radians.
    """
    def __init__(
        self, epoch: str, NGP_ra: float, NGP_dec: float, NCP_lon: float
    ):
        """
        Parameters
        ----------
        epoch : str
            Label of the epoch at which the parameters are defined.
        NGP_ra : float
            Right ascension of the north galactic pole at the epoch in
            radians.
        NGP_dec : float
            Declination of the north galactic pole at the epoch in radians.
        NCP_lon : float
            Galactic longitude of the north celestial pole at the epoch in
            radians.
        """
        self.epoch = epoch
        self.NGP_ra = NGP_ra
        self.NGP_dec = NGP_dec
        self.NCP_lon = NCP_lon


class OrbitalParameters:
    """
    Parameters of a planetary orbit.
    
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
    def __init__(
        self, epoch: str, e_semimajor_axis: float, e_eccentricity: float, e_mean_anomaly: float, e_mean_speed: float,
        e_ecliptic_longitude: float, longitude_coeff: np.ndarray,
        e_obliquity: float, obliquity_coeff: np.ndarray
    ):
        self.epoch = epoch
        self.e_semimajor_axis = e_semimajor_axis
        self.e_eccentricity = e_eccentricity
        self.e_mean_anomaly = e_mean_anomaly
        self.e_mean_speed = e_mean_speed
        self.e_ecliptic_longitude = e_ecliptic_longitude
        self.longitude_coeff = longitude_coeff
        self.e_obliquity = e_obliquity
        self.obliquity_coeff = obliquity_coeff
        self._longitude_coeff = np.append(
                self.e_ecliptic_longitude, self.longitude_coeff)
        self._obliquity_coeff = np.append(
                self.e_obliquity, self.obliquity_coeff)
    
    
    def orbital_velocity(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Velocity of the orbit in a given time interval.
        
        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval to compute the velocity in.
        
        Returns
        -------
        numpy.ndarray
            Velocity of the orbit.
        """
        EA = EA_at_UT1(
                time_interval, self.e_mean_anomaly, self.e_mean_speed,
                self.e_eccentricity)
        cos_EA = np.cos(EA)
        sin_EA = np.sin(EA)

        one_m_e2 = (1 - self.e_eccentricity)*(1 + self.e_eccentricity)

        ecEA = self.e_eccentricity*cos_EA
        one_minus_ecEA2 = 1 - ecEA*ecEA
        cos_gamma = -self.e_eccentricity*sin_EA/np.sqrt(one_minus_ecEA2)
        sin_gamma = np.sqrt(one_m_e2/one_minus_ecEA2)
        gamma = np.arctan2(sin_gamma, cos_gamma)

        nu = self.ecliptic_longitude(time_interval)
        TA = TA_from_EA(EA, self.e_eccentricity)

        axial_tilt = self.obliquity(time_interval)

        speed = self.e_mean_speed*self.e_semimajor_axis/86400.0
        mag_v_earth = speed*np.sqrt(one_minus_ecEA2)/(1 - ecEA)

        srev_v = np.sin(TA + nu - gamma)

        return np.stack([
                    mag_v_earth*np.cos(TA + nu - gamma),
                    mag_v_earth*np.cos(axial_tilt)*srev_v,
                    mag_v_earth*np.sin(axial_tilt)*srev_v], 
                axis=-1)
    
    
    def sun_position(self, time_interval: np.ndarray) -> np.ndarray:
    
        """
        Unit vector pointing towards the Sun in equatorial coordinates
        in a given time interval.
        
        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval to compute the unit vector in.
        
        Returns
        -------
        numpy.ndarray
            Direction vector of the Sun.
        """
        TAnu = (self.true_anomaly(time_interval)
                + self.ecliptic_longitude(time_interval))
        obliquity = self.obliquity(time_interval)
        srev = np.sin(TAnu)
        return np.column_stack((
                np.cos(TAnu), np.cos(obliquity)*srev,
                np.sin(obliquity)*srev))
    
    
    
    def eccentric_anomaly(
        self, time_interval: np.ndarray, Niter: int = 7
    ) -> np.ndarray:
        """
        Compute eccentric anomaly in a given time interval.
        
        Parameters
        ----------
        time_interval : TimeInterval object
            Time interval to compute the eccentric anomaly in.
        
        Returns
        -------
        1-D numpy.ndarray
            Array of eccentric anomaly values in the time interval.
        """
        return EA_at_UT1(
                time_interval, self.e_mean_anomaly, self.e_mean_speed,
                self.e_eccentricity, Niter)
    
    
    
    def true_anomaly(self, time_interval: np.ndarray) -> np.ndarray:
        """
        True anomaly of the orbit in a given time interval.
        
        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval to compute the true anomaly in.
        
        Returns
        -------
        numpy.ndarray
            True anomaly of the orbit.
        """
        return TA_from_EA(
                self.eccentric_anomaly(time_interval), self.e_eccentricity)
    
    
    def obliquity(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Obliquity of the planet's axis in a given time interval.
        
        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval to compute the obliquity in.
        
        Returns
        -------
        numpy.ndarray
            Obliquity of the orbit.
        """
        return np.polynomial.polynomial.polyval(
                time_interval/36525.0, self._obliquity_coeff)
    
    
    def ecliptic_longitude(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Ecliptic longitude of the vernal equinox in a given time
        interval.
        
        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval to compute the ecliptic longitude in.
        
        Returns
        -------
        numpy.ndarray
            Ecliptic longitude of the vernal equinox.
        """
        return np.polynomial.polynomial.polyval(
                time_interval/36525.0, self._longitude_coeff)

# ORBITAL PARAMETERS

# Angles are in radians

earth_J2000_eccentricity = 0.01671022

# Semi-mjaor and -minor axes
# Units: km
earth_J2000_semimajor_axis = 1.49598022961e+8
earth_J2000_semiminor_axis = 1.49577135266e+8

# Units: days
earth_J2000_orbital_period = 365.256363004

# Units: Radians per day
earth_J2000_mean_speed = 2*np.pi/earth_J2000_orbital_period

# J2000_M0 is the mean motion of the Earth at J2000
# J2000_M0 = lambda - varpi
earth_J2000_mean_anomaly = -0.0433337328

earth_J2000_obliquity = 0.40909260

# nu0 is the angle from the perihelion of the Earth's orbit to the first
# equinox of the year
earth_J2000_ecliptic_longitude = -1.344825

# Polynomial coefficients for the precession of the equinoxes.
earth_obliquity_coeff = np.array([-2.227088e-4, 2.9e-9, 8.790e-9])
earth_longitude_coeff = np.array([2.43802956e-2, -5.38691e-6, -2.9e-11])

# EARTH PARAMETERS

# Angular speed of the Earth (radians per day)
ang_speed = 6.30038736

# Radius of earth in Km
earth_radius = 6371

# Parameters of the Earth rotation angle
ERA_0 = 0.7790572732640
ERA_rate = 1.00273781191135448

# GALACTIC COORDINATE PARAMETERS

NGP_dec = 27.084*np.pi/180
NGP_ra = 192.729*np.pi/180
NCP_lon = 122.928*np.pi/180

# Peculiar velocity of the solar system.
solar_v_pec = np.array([11.1,12.24,7.25])


class MilkyWayCoordinates(GalacticParameters):
    def __init__(self):
        super().__init__('J2000', NGP_ra, NGP_dec, NCP_lon)


class EarthOrbit(OrbitalParameters):
    def __init__(self):
        super().__init__(
                'J2000', earth_J2000_semimajor_axis, earth_J2000_eccentricity,
                earth_J2000_mean_anomaly, earth_J2000_mean_speed,
                earth_J2000_ecliptic_longitude, earth_longitude_coeff,
                earth_J2000_obliquity, earth_obliquity_coeff)


