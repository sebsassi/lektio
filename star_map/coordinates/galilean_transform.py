import typing

import abc
import numpy as np

from .astro_parameters import GalacticParameters, OrbitalParameters


def latlon_str_as_float_radians(latlon: str, angle_units: str = "deg") -> float:
    """
    Interpret a latitude (longitude) string as float in radians.

    Parameters
    ----------
    latlon : str
        String representation of latitude (longitude) in the
        degrees-hemisphere notation (e.g., '55.2 N').
    angle_units : {'deg', 'rad'}, optional
        Input angle unit system.

    Returns
    -------
    float
        Latitude (longitude) in radians.
    """
    ang_str, hemisphere = latlon.split()
    sign = 1 if hemisphere in ['S','E'] else -1
    shift = 0 if hemisphere in ['W','E'] else 90
    if angle_units == 'rad':
        shift *= np.pi/180
    res = shift + sign*float(ang_str)
    if angle_units == 'deg':
        res *= np.pi/180
    return res


def latitude_as_float_radians(
    latitude: str, angle_unit: str = "deg",
    angle_convention: str = "geographical"
) -> float:
    if angle_unit not in ['deg','rad']:
        raise ValueError(f"Unsupported angle unit {angle_unit}.")
    if isinstance(latitude, str):
        return latlon_str_as_float_radians(latitude, angle_unit)
    else:
        to_rad = np.pi/180 if angle_unit == 'deg' else 1
        if angle_convention == 'geographical':
            return 0.5*np.pi - latitude*to_rad
        elif angle_convention == 'physics':
            return latitude*to_rad


def longitude_as_float_radians(
    longitude: str, angle_unit: str = "deg",
    angle_convention: str = "geographical"
) -> float:
    if angle_unit not in ['deg','rad']:
        raise ValueError(f"Unsupported angle unit {angle_unit}.")
    if isinstance(longitude, str):
        return latlon_str_as_float_radians(longitude, angle_unit)
    else:
        to_rad = np.pi/180 if angle_unit == 'deg' else 1
        return longitude*to_rad


def angle_as_radians(angle: float, angle_unit: str) -> float:
    if angle_unit not in ["deg","rad"]:
        raise ValueError(f"Unsupported angle unit {angle_unit}.")
    return angle if angle_unit == "rad" else (np.pi/180)*angle
    


def affine_transform_vector(
    matrix: np.ndarray, translation: np.ndarray, source_vector: np.ndarray, 
    translate_source: bool = False
) -> np.ndarray:
    """
    Apply an affine transformation to a vector.

    Parameters
    ----------
    matrix : numpy.ndarray
        Linear transformation matrix.
    translation : numpy.ndarray
        Translation vector.
    source_vector : numpy.ndarray
        Vector in the source coordinate system.
    translate_source : bool, optional
        If True, perform translation in the source coordinate system.
        If False, perform translation in the destination coordinate
        system.

    Returns
    -------
    numpy.ndarry
        Vector in the destination coordinate system.
    """
    if translate_source:
        return np.einsum('...ij,...j', matrix, source_vector + translation)
    else:
        return np.einsum('...ij,...j', matrix, source_vector) + translation


def linear_transform_vector(
    matrix: np.ndarray, source_vector: np.ndarray
) -> np.ndarray:
    """
    Apply an linear transformation to a vector.

    Parameters
    ----------
    matrix : numpy.ndarray
        Linear transformation matrix.
    source_vector : numpy.ndarray
        Vector in the source coordinate system.

    Returns
    -------
    numpy.ndarry
        Vector in the destination coordinate system.
    """
    return np.einsum('...ij,...j', matrix, source_vector)


def linear_transform_matrix(
    matrix: np.ndarray, source_matrix: np.ndarray
) -> np.ndarray:
    """
    Apply an linear transformation to a matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Linear transformation matrix.
    source_matrix : numpy.ndarray
        Matrix in the source coordinate system.

    Returns
    -------
    numpy.ndarry
        Vector in the destination coordinate system.
    """
    return np.einsum('...ij,...jk,...lk', matrix, source_matrix, matrix)


class GalileanTransform(abc.ABC):
    """
    Abstract base class for implementing galilean transformations.
    """
    def transform_velocity(
        self, velocity: np.ndarray, time_interval: np.ndarray
    ) -> np.ndarray:
        """
        Transform a velocity from one coordinate system to another.

        Parameters
        ----------
        velocity : numpy.ndarray
            Velocity in the source coordinate system. Its last
            dimension is the number of spatial dimensions.
        time_interval : numpy.ndarray
            Time interval over which the velocity is transformed.

        Returns
        -------
        numpy.ndarray
            Velocity in the destination cooordinate system.
        """
        return affine_transform_vector(
                    self.rotation(time_interval), self.boost(time_interval),
                    velocity)


    def transform_vector(
        self, vector: np.ndarray, time_interval: np.ndarray
    ) -> np.ndarray:
        """
        Transform a vector from one coordinate system to another.

        As opposed to transform_velocity, this is for transformation
        of generic vectors (e.g., direction vectors), which are only
        acted on by the rotation part of the Galilean transformation.

        Parameters
        ----------
        vector : numpy.ndarray
            Vector in the source coordinate system. Its last
            dimension is the number of spatial dimensions.
        time_interval : numpy.ndarray
            Time interval over which the vector is transformed.

        Returns
        -------
        numpy.ndarray
            Vector in the destination cooordinate system.
        """
        return linear_transform_vector(self.rotation(time_interval), vector)


    def transform_matrix(
        self, matrix: np.ndarray, time_interval: np.ndarray
    ) -> np.ndarray:
        """
        Transform a matrix from one coordinate system to another.

        Parameters
        ----------
        matrix : numpy.ndarray
            Matrix in the source coordinate system. Its last
            dimension is the number of spatial dimensions.
        time_interval : numpy.ndarray
            Time interval over which the matrix is transformed.

        Returns
        -------
        numpy.ndarray
            Matrix in the destination cooordinate system.
        """
        return linear_transform_matrix(self.rotation(time_interval), matrix)


    @abc.abstractmethod
    def rotation(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Rotation matrix of the Galilean transformation.

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the matrix is defined.

        Returns
        -------
        numpy.ndarray
            Rotation matrix.
        """
        return None


    @abc.abstractmethod
    def boost(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Velocity boost of the Galilean transformation.

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the boost is defined.

        Returns
        -------
        numpy.ndarray
            velocity boost.
        """
        return None
    

    def inverse(self):
        return InverseGalileanTransform(self)


class InverseGalileanTransform(GalileanTransform):
    """
    Inverse of a Galilean transform.
    """
    def __init__(self, galilean_transform: GalileanTransform):
        self._transform = galilean_transform
    

    def transform_velocity(
        self, velocity: np.ndarray, time_interval: np.ndarray
    ) -> np.ndarray:
        """
        Transform a velocity from one coordinate system to another.

        Parameters
        ----------
        velocity : numpy.ndarray
            Velocity in the source coordinate system. Its last
            dimension is the number of spatial dimensions.
        time_interval : numpy.ndarray
            Time interval over which the velocity is transformed.

        Returns
        -------
        numpy.ndarray
            Velocity in the destination cooordinate system.
        """
        return affine_transform_vector(
                    self.rotation(time_interval), self.boost(time_interval),
                    velocity, translate_source=True)
    
    

    def transform_vector(
        self, vector: np.ndarray, time_interval: np.ndarray
    ) -> np.ndarray:
        """
        Transform a vector from one coordinate system to another.

        As opposed to transform_velocity, this is for transformation
        of generic vectors (e.g., direction vectors), which are only
        acted on by the rotation part of the Galilean transformation.

        Parameters
        ----------
        vector : numpy.ndarray
            Vector in the source coordinate system. Its last
            dimension is the number of spatial dimensions.
        time_interval : numpy.ndarray
            Time interval over which the vector is transformed.

        Returns
        -------
        numpy.ndarray
            Vector in the destination cooordinate system.
        """
        return linear_transform_vector(
                self.rotation(time_interval), vector)
    

    def rotation(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Rotation matrix of the Galilean transformation.

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the matrix is defined.

        Returns
        -------
        numpy.ndarray
            Rotation matrix.
        """
        return np.einsum('...ji', self._transform.rotation(time_interval))
    

    def boost(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Velocity boost of the Galilean transformation.

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the boost is defined.

        Returns
        -------
        numpy.ndarray
            velocity boost.
        """
        return -self._transform.boost(time_interval)


class GalHelTransform(GalileanTransform):
    """
    Class for Galilean transformations from the galactic to a
    heliocentric coordinate system.

    The galactic coordinate system is a right-handed coordinate system
    centered at the Sun with its z-axis pointing to the galactic north
    and its x-axis pointing towards the galactic center. It is by
    definition always centered at the Sun at any instance, but has no
    velocity relative to the galactic center.

    The heliocentric coordinate system is a boosted version of the
    galactic coordinate system such that its velocity relative to the
    galactic coordinate system is that of the solar system.

    Attributes
    ----------
    v_circ : float
        Local circular velocity of the galxy at the Sun's position.
    v_pec : numpy.ndarray
        Peculiar velocity of the Sun.
    """
    def __init__(
        self, v_circ: float, v_pec: np.ndarray, vel_units: str = "kms"
    ):
        """
        Parameters
        ----------
        v_circ : float
            Local circular velocity of the galxy at the Sun's position.
        v_pec : numpy.ndarray
            Peculiar velocity of the Sun.
        vel_units {'kms', 'natural'}, optional
            Units of the input velocities; 'kms' is kilometers per
            second, 'natural' is as a fraction of speed of light.
        """
        if v_circ < 0:
            raise ValueError("`v_circ` must be nonnegative.")
        if vel_units == 'kms':
            self.v_circ = v_circ
            self.v_pec = v_pec
        if vel_units == 'natural':
            self.v_circ = v_circ*299792.458
            self.v_pec = v_pec*299792.458
        super().__init__()


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.v_circ!r}, {self.v_pec!r})"


    def rotation(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Rotation matrix of the Galilean transformation

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the matrix is defined.

        Returns
        -------
        numpy.ndarray
            Rotation matrix.
        """
        return np.identity(3)


    def boost(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Velocity boost of the Galilean transformation

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the boost is defined.

        Returns
        -------
        numpy.ndarray
            velocity boost.
        """
        return -self.solar_velocity()


    def solar_velocity(self) -> np.ndarray:
        """
        Velocity of solar system in galactic coordinates.

        Returns
        -------
        numpy.ndarray
            Solar velocity.
        """
        return np.array([0, self.v_circ, 0]) + self.v_pec


    def id_string(self, sep: str = "_") -> str:
        v_circ_str = '{:.0f}kms'.format(self.v_circ)
        v_pec_str = '{0:.1f}_{1:.1f}_{2:.1f}kms'.format(
                self.v_pec[0], self.v_pec[1], self.v_pec[2])
        return sep.join((v_circ_str, v_pec_str))


class HelEquTransform(GalileanTransform):
    """
    Class for Galilean transformations from a heliocentric coordinate
    system to the equatorial coordinate system.

    The heliocentric coordinate system is a right-handed coordinate
    system centered at the Sun with its z-axis pointing to the galactic
    north and its x-axis pointing towards the galactic center. Its
    velocity relative to the galactic coordinate system is that of the
    solar system.

    The equatorial coordinate system is a right-handed coordinate
    system centered at the planet with its z-axis pointing along the
    direction of the planet's angular momentum and its x-axis pointing
    towards the vernal equinox.

    Attributes
    ----------
    galactic_angles : GalacticParameters
        Parameters defining the galactic coordinate system.
    orbital_parameters : OrbitalParameters
        Parameters defining a planetary orbit.
    """
    def __init__(
        self, galactic_angles: GalacticParameters,
        orbital_parameters: OrbitalParameters
    ):
        """
        Parameters
        ----------
        galactic_angles : GalacticParameters
            Parameters defining the galactic coordinate system.
        orbital_parameters : OrbitalParameters
            Parameters defining a planetary orbit.
        """
        self.galactic_angles = galactic_angles
        self.orbital_parameters = orbital_parameters


    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.galactic_angles!r}, "
                f"{self.orbital_parameters!r})")


    def rotation(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Rotation matrix of the Galilean transformation

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the matrix is defined.

        Returns
        -------
        numpy.ndarray
            Rotation matrix.
        """
        cal = np.cos(self.galactic_angles.NGP_ra - np.pi)
        cdel = np.cos(self.galactic_angles.NGP_dec - np.pi/2)
        cel = np.cos(self.galactic_angles.NCP_lon)

        sal = np.sin(self.galactic_angles.NGP_ra - np.pi)
        sdel = np.sin(self.galactic_angles.NGP_dec - np.pi/2)
        sel = np.sin(self.galactic_angles.NCP_lon)

        return np.array([
                [cal*cdel*cel + sal*sel , cal*cdel*sel - sal*cel, cal*sdel   ],
                [sal*cdel*cel - cal*sel , sal*cdel*sel + cal*cel, sal*sdel   ],
                [-sdel*cel              , -sdel*sel             , cdel      ]])


    def boost(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Velocity boost of the Galilean transformation

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the boost is defined.

        Returns
        -------
        numpy.ndarray
            velocity boost.
        """
        obliquity_coeff_1 = -2.227088e-4
        obliquity_coeff_2 = 2.9e-9
        obliquity_coeff_3 = 8.790e-9

        longitude_coeff_1 = 0.0243802956
        longitude_coeff_2 = -5.38691e-6
        longitude_coeff_3 = -2.9e-11

        return -self.orbital_parameters.orbital_velocity(time_interval)


    def id_string(self, sep: str = "_") -> str:
        return ""


class EquHorTransform(GalileanTransform):
    """
    Class for Galilean transformations from the equatorial coordinate
    system to the horizontal coordinate system.

    The equatorial coordinate system is a right-handed coordinate
    system centered at the planet with its z-axis pointing along the
    direction of the planet's angular momentum and its x-axis pointing
    towards the vernal equinox.

    The horizontal coordinate system is a right-handed coordinate
    system centered at a point on the planet's surface with its z-axis
    pointing to the zenith and its y-axis pointing south as defined
    by the planet's rotational south pole.

    Attributes
    ----------
    epoch : str
        Label of the epoch at which the parameters are defined.
    latitude : float
        Latitude of the point on the planet's surface in radians,
        measured in the physics convention, where it is the angle from
        the north pole.
    longitude : float
        Longitude of the point on the planet's surface in radians.
    e_RA_0 : float
        Rotation angle of the planet at the epoch.
    e_RA_rate : float
        Rotation rate of the planet at the epoch.
    radius : float
        Radius of the planet in kilometers.
    """
    def __init__(
        self, latitude: float, longitude: float, epoch: str, e_RA_0: float, e_RA_rate: float, radius: float, angle_unit: str = "deg"
    ):
        """
        Parameters
        ----------
        latitude : float
            Latitude of the point on the planet's surface. The latitude
            may be given as a number or as a string using the standard
            hemisphere notation, e.g., '30 N'.
        longitude : float
            Longitude of the point on the planet's surface. The
            longitude may be given as a number or as a string using the
            standard hemisphere notation, e.g., '30 W'.
        epoch : str
            Label of the epoch at which the parameters are defined.
        e_RA_0 : float
            Rotation angle of the planet at the epoch.
        e_RA_rate : float
            Rotation rate of the planet at the epoch.
        radius : float
            Radius of the planet in kilometers.
        angle_unit : {'deg', 'rad'}, optional
            Angle units in which the latitude and longitude are
            given.
        """
        self.epoch = epoch
        self.longitude = angle_as_radians(longitude, angle_unit)
        self.latitude = angle_as_radians(latitude, angle_unit)
        self.e_RA_0 = e_RA_0
        self.e_RA_rate = e_RA_rate
        self.radius = radius


    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.epoch!r}, "
                f"{self.latitude!r}, {self.longitude!r}, {self.e_RA_0!r}, "
                f"{self.e_RA_rate!r}, {self.radius!r})")


    def rotation(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Rotation matrix of the Galilean transformation

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the matrix is defined.

        Returns
        -------
        numpy.ndarray
            Rotation matrix.
        """
        rotation = self.RA(time_interval) + self.longitude - np.pi/2

        clat = np.cos(self.latitude)
        crot = np.cos(rotation)

        slat = np.sin(self.latitude)
        srot = np.sin(rotation)

        cnt = srot.shape[0]
        one = np.ones(cnt)
        return (np.stack(
                (crot       ,srot      ,  np.zeros(cnt),
                -slat*srot  ,slat*crot , -clat*one     ,
                -clat*srot  ,clat*crot ,  slat*one      ), axis=-1)
                .reshape(-1, 3, 3))


    def boost(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Velocity boost of the Galilean transformation

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the boost is defined.

        Returns
        -------
        numpy.ndarray
            velocity boost.
        """
        vr = self.radius*(2*np.pi*self.e_RA_rate/86400)*np.sin(self.latitude)
        return -np.array([vr,0,0])


    def RA(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Rotation angle of the planet in a given time interval.

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval to compute the rotation angle in.

        Returns
        -------
        numpy.ndarray
            Rotation angle of the planet.
        """
        return 2*np.pi*(self.e_RA_0 + self.e_RA_rate*time_interval)


    def latitude_as_str(self, prec: str = 3, sep: str = " ") -> str:
        """
        Return a string representation of the latitude.

        Parameters
        ----------
        prec : int, optional
            Output precision.
        sep : str, optional
            Separator between angle and hemisphere symbol.

        Returns
        -------
        str
            String representation of the latitude.
        """
        h_str = 'N' if self.latitude > 0 else 'S'
        latitude = self.latitude*180/np.pi
        return sep.join((
                "{0:.{prec}f}".format(abs(latitude), prec=prec), h_str))


    def longitude_as_str(self, prec: str = 3, sep: str = " ") -> str:
        """
        Return a string representation of the longitude.

        Parameters
        ----------
        prec : int, optional
            Output precision.
        sep : str, optional
            Separator between angle and hemisphere symbol.

        Returns
        -------
        str
            String representation of the longitude.
        """
        h_str = 'E' if self.longitude > 0 else 'W'
        longitude = self.longitude*180/np.pi
        return sep.join((
                "{0:.{prec}f}".format(abs(longitude), prec=prec), h_str))


    def id_string(self, sep: str = "_") -> str:
        """
        Human readable string identifying the coordinate transformation.

        The string contains the relevant data that define the
        coordinate transformation. This string is NOT guaranteed to be unique.
        Numerical values are truncated for readability.

        Parameters
        ----------
        sep : str, optional
            Separator between elements of the string.

        Returns
        -------
        str
            String representation of the coordinate transformation.
        """
        lat_str = self.latitude_as_str(prec=4, sep='')
        lon_str = self.longitude_as_str(prec=4, sep='')
        return sep.join((lat_str, lon_str))


class RotationTransform(GalileanTransform):
    """
    Constant rotation between coordinate systems.

    This class wraps a general constant rotation matrix into a Galilean
    transform.

    Attributes
    ----------
    rotation : numpy.ndarray
        Rotation matrix
    """
    def __init__(self, rotation: np.ndarray, atol: float = 1.0e-9):
        """
        Parameters
        ----------
        rotation : numpy.ndarray
            Rotation matrix
        atol : float, optional
            Absolute tolerance of the checks for whether the rotation
            matrix is an SO(3) matrix. Namely, if we denote the matrix
            R, then `|det(R) - 1| < atol` and `|(RR^T - I)[i,j]| < atol`
            must hold.
        """
        if rotation.shape[-2:] != (3, 3):
            raise ValueError("Rotation matrix must be a 3x3 matrix.")
        unit_det = np.all(np.isclose(
                np.linalg.det(rotation), 1, atol=atol))
        if not unit_det:
            raise ValueError("Determinant of rotation matrix must be 1.")
        orthogonal = np.all(np.isclose(
                np.matmul(rotation, rotation.T), np.identity(3),
                atol=atol))
        if not orthogonal:
            raise ValueError("Rotation matrix must be orthogonal.")
        self._rotation = rotation


    @classmethod
    def from_plane_angles(
        cls, pol_z: float, az_z: float, rot_xy: float, atol: float = 1.0e-9
    ):
        """
        Construct a rotation from angles defining the new xy-plane.

        The rotation in this convention is defined by three angles.
        The polar and azimuthal angles of the z-axis define the z-axis
        of the target coordinate system. Then a third rotation angle
        defines the rotation of the x-axis in the target xy-plane
        relative to the intersection of the source and target
        xy-planes.

        Parameters
        ----------
        pol_z : float
            Polar angle of the target z-axis in the source coordinate
            system.
        az_z : float
            Azimuthal angle of the target z-axis in the source
            coordinate system.
        rot_xy : float
            Rotation angle of the target x-axis in the target xy-plane
            relative to the intersection of the source and target
            xy-planes.
        atol : float, optional
            Absolute tolerance of the checks for whether the rotation
            matrix is an SO(3) matrix. Namely, if we denote the matrix
            R, then `|det(R) - 1| < atol` and `|(RR^T - I)[i,j]| < atol`
            must hold.

        Returns
        -------
        RotationTransform
        """
        cpol_z = np.cos(pol_z)
        spol_z = np.sin(pol_z)
        caz_z = np.cos(az_z)
        saz_z = np.sin(az_z)
        crot_xy = np.cos(rot_xy)
        srot_xy = np.sin(rot_xy)

        rotation = np.array([
                [
                    caz_z*crot_xy - saz_z*srot_xy*cpol_z,
                    -caz_z*srot_xy - saz_z*crot_xy*cpol_z,
                    saz_z*spol_z],
                [
                    saz_z*crot_xy + caz_z*srot_xy*cpol_z,
                    -saz_z*srot_xy + caz_z*crot_xy*cpol_z,
                    caz_z*spol_z],
                [
                    srot_xy*spol_z,
                    crot_xy*spol_z,
                    cpol_z]])

        return cls(rotation, atol=atol)


    @classmethod
    def from_euler_angles(
        cls, ang_x: float, ang_y: float, ang_z: float, atol: float = 1.0e-9
    ):
        """
        Construct a rotation from Euler angles.

        Parameters
        ----------
        ang_x : float
            Angle of rotation about x-axis.
        ang_y : float
            Angle of rotation about y-axis.
        ang_z : float
            Angle of rotation about z-axis.

        Returns
        -------
        RotationTransform
        """
        c_x = np.cos(ang_x)
        s_x = np.sin(ang_x)
        c_y = np.cos(ang_y)
        s_y = np.sin(ang_y)
        c_z = np.cos(ang_z)
        s_z = np.sin(ang_z)

        rotation = np.array([
                [
                    c_y*c_z,
                    -c_x*s_z + s_x*s_y*c_z,
                    s_x*s_z + c_x*s_y*c_z],
                [
                    c_y*s_z,
                    c_x*c_z + s_x*s_y*s_z,
                    -s_x*c_z + c_x*s_y*s_z],
                [
                    -s_y,
                    s_x*c_y,
                    c_x*c_y]])

        return cls(rotation, atol=atol)


    def rotation(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Rotation matrix of the Galilean transformation

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the matrix is defined.

        Returns
        -------
        numpy.ndarray
            Rotation matrix.
        """
        return self._rotation


    def boost(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Velocity boost of the Galilean transformation

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the boost is defined.

        Returns
        -------
        numpy.ndarray
            velocity boost.
        """
        return 0


    def id_string(self, sep: str = "_"):
        return ""


class ParametricRotationTransform(GalileanTransform):
    def __init__(self, rotation_matrix_func: typing.Callable):
        self.rotation_matrix_func = rotation_matrix_func


    @classmethod
    def from_plane_angles(cls, plane_angle_func: typing.Callable):
        def rotation_matrix_func(time_interval):
            pol_z, az_z, rot_xy = plane_angle_func(time_interval)
            cpol_z = np.cos(pol_z)
            spol_z = np.sin(pol_z)
            caz_z = np.cos(az_z)
            saz_z = np.sin(az_z)
            crot_xy = np.cos(rot_xy)
            srot_xy = np.sin(rot_xy)

            rotation = np.array([
                    [
                        caz_z*crot_xy - saz_z*srot_xy*cpol_z,
                        -caz_z*srot_xy - saz_z*crot_xy*cpol_z,
                        saz_z*spol_z],
                    [
                        saz_z*crot_xy + caz_z*srot_xy*cpol_z,
                        -saz_z*srot_xy + caz_z*crot_xy*cpol_z,
                        caz_z*spol_z],
                    [
                        srot_xy*spol_z,
                        crot_xy*spol_z,
                        cpol_z]])
            return rotation

        return cls(rotation_matrix_func)


    def rotation(self, time_interval: np.ndarray) -> np.ndarray:
        return self.rotation_matrix_func(time_interval)


    def boost(self, time_interval: np.ndarray) -> np.ndarray:
        return 0


class EclEquTransform(GalileanTransform):
    """
    Class for Galilean transformations from the horizontal coordinate
    system to the equatorial coordinate system.

    The ecliptic coordinate system is a right-handed coordinate
    system centered at the planet with its z-axis normal to the
    ecliptic, such that the z-component of the planet's angular
    momentum is positive, and its x-axis pointing towards the
    vernal equinox.

    The equatorial coordinate system is a right-handed coordinate
    system centered at the planet with its z-axis pointing along the
    direction of the planet's angular momentum and its x-axis pointing
    towards the vernal equinox.
    """
    def __init__(self, epoch_obliquity: float, obliquity_coeff: np.ndarray):
        self.epoch_obliquity = epoch_obliquity
        self.obliquity_coeff = obliquity_coeff
        self._obliquity_coeff = np.append(
                self.epoch_obliquity, self.obliquity_coeff)


    def rotation(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Rotation matrix of the Galilean transformation

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the matrix is defined.

        Returns
        -------
        numpy.ndarray
            Rotation matrix.
        """
        obliqutity = self.obliquity(time_interval)

        cobl = np.cos(obliqutity)
        sobl = np.sin(obliqutity)

        cnt = cobl.shape[0]
        zero = np.zeros(cnt)
        return np.stack(
                (np.ones(cnt), zero ,  zero,
                 zero        , cobl , -sobl,
                 zero        , sobl ,  cobl ), axis=-1).reshape(-1, 3, 3)


    def boost(self, time_interval: np.ndarray) -> np.ndarray:
        """
        Velocity boost of the Galilean transformation

        Parameters
        ----------
        time_interval : numpy.ndarray
            Time interval over which the boost is defined.

        Returns
        -------
        numpy.ndarray
            velocity boost.
        """
        return 0


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
                time_interval, self._obliquity_coeff)
