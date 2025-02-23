import copy
import typing
import numpy as np

from .galilean_transform import (
        GalileanTransform, GalHelTransform, HelEquTransform, EquHorTransform,
        EclEquTransform, latitude_as_float_radians, longitude_as_float_radians)
from .coordinate_atlas import CoordinateAtlas, LocalGalileanCoordinateAtlas
from .astro_parameters import (
        MilkyWayCoordinates, EarthOrbit, earth_radius, ERA_0, ERA_rate,
        earth_radius, solar_v_pec, GalacticParameters, OrbitalParameters)


class CelestialCoordinateAtlas(LocalGalileanCoordinateAtlas):
    """
    Atlas of celestial coordinate systems connected via Galilean
    transformations.

    This is a default coordinate atlas that implements transitions
    between a group of coordinate systems:

    'gal': The galactic coordinate system, which is at any moment
    fixed at the position of the Sun, and has its z-axis pointing to
    the galactic north, and its x-axis to the galactic center, but
    it has no velocity relative to the galactic center.

    'hel': The heliocentric coordinate system, which is otherwise the
    same as the galactic coordinate system, but its velocity relative
    to that is the velocity of the Sun.

    'equ': The equatorial coordinate system, which is fixed at the
    position of the Earth, has its z-axis aligned with the Earth's axis
    of rotation, and its x-axis pointing towards the vernal equinox,
    and has the Earth's velocity added to it.

    'ecl': The ecliptic coordinate system, which has its z-axis aligned
    with the Earth's orbital plane, but is otherwise the same as the
    equatorial coordinate system.

    This default coordinate atlas can be extended with transformations
    to other coordinate systems via its methods.
    """
    def __init__(
        self, v_circ: float = 220, v_pec: np.ndarray = np.array([11.1,12.24,7.25]), vel_units: str = "kms"
    ):
        """
        Parameters
        ----------
        v_circ : float
            Local circular velocity of the galxy at the Sun's position.
        v_pec : numpy.ndarray
            Peculiar velocity of the Sun.
        vel_units : {'kms', 'natural'}, optional
            Units of the input velocities: 'kms' is kilometers per
            second, 'natural' is as a fraction of speed of light.
        """
        self.v_circ = v_circ
        self.v_pec = v_pec
        self.vel_units = vel_units
        self.galactic_angles = MilkyWayCoordinates()
        self.earth_orbital_parameters = EarthOrbit()
        transition_maps = {
                ('gal','hel'): GalHelTransform(v_circ, v_pec, vel_units),
                ('hel','equ'): HelEquTransform(
                        self.galactic_angles, self.earth_orbital_parameters),
                ('ecl','equ'): EclEquTransform(
                        self.earth_orbital_parameters.obliquity,
                        self.earth_orbital_parameters.obliquity_coeff)
                }
        super().__init__(transition_maps)


    def add_gal_hel_transform(
        self, gal_label: str, hel_label: str, v_circ: float, v_pec: np.ndarray, 
        vel_units: str = "kms"
    ):
        """
        Add a galactic to heliocentric transformation to the atlas.

        Parameters
        ----------
        gal_label : str
            Label for the galactic coordinate system.
        hel_label : str
            Label for the heliocentric coordinate system.
        v_circ : float
            Local circular velocity of the galxy at the Sun's position.
        v_pec : numpy.ndarray
            Peculiar velocity of the Sun.
        vel_units : {'kms', 'natural'}, optional
            Units of the input velocities: 'kms' is kilometers per
            second, 'natural' is as a fraction of speed of light.
        """
        transform = GalHelTransform(v_circ, v_pec, vel_units)
        self.add_edge(gal_label, hel_label, transform)


    def add_hel_equ_transform(
        self, hel_label: str, equ_label: str,
        galactic_angles: GalacticParameters,
        orbital_parameters: OrbitalParameters
    ):
        """
        Add a heliocentric to equatorial transformation to the atlas.

        Parameters
        ----------
        hel_label : str
            Label for the heliocentric coordinate system.
        equ_label : str
            Label for the equatorial coordinate system.
        galactic_angles : GalacticParameters
            Parameters defining the galactic coordinate system.
        orbital_parameters : OrbitalParameters
            Parameters defining a planetary orbit.
        """
        transform = HelEquTransform(galactic_angles, orbital_parameters)
        self.add_transform(hel_label, equ_label, transform)


    def add_equ_hor_transform(
        self, equ_label: str, hor_label: str, latitude: float,
        longitude: float, epoch: str, e_RA_0: float,
        e_RA_rate: float, radius: float, angle_unit: str = "deg"
    ):
        """
        Add an equatorial to horizontal transformation to the atlas.

        Parameters
        ----------
        equ_label : str
            Label for the equatorial coordinate system.
        hor_label : str
            Label for the horizontal coordinate system.
        latitude : float, str
            Latitude of the point on the planet's surface. The latitude
            may be given as a number or as a string using the standard
            hemisphere notation, e.g., '30 N'.
        longitude : float, str
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
        transform = EquHorTransform(
                latitude, longitude, epoch, e_RA_0, e_RA_rate, radius,
                angle_unit)
        self.add_transform(equ_label, hor_label, transform)


    def add_ecl_equ_transform(
        self, ecl_label: str, equ_label: str,
        orbital_parameters: OrbitalParameters
    ):
        """
        Add an equatorial to horizontal transformation to the atlas.

        Parameters
        ----------
        ecl_label : str
            Label for the ecliptic coordinate system.
        equ_label : str
            Label for the equatorial coordinate system.
        orbital_parameters : OrbitalParameters
            Parameters defining a planetary orbit.
        """
        transform = EclEquTransform(
                orbital_parameters.obliquity,
                orbital_parameters.obliquity_coeff)
        self.add_transform(ecl_label, equ_label, transform)
    
    def serializable(self):
        return {
            "class": self.__class__.__name__,
            "data": {
                "v_circ": self.v_circ,
                "v_pec": list(self.v_pec)
            }
        }


class CoordinateSystem:
    """
    A base class for coordinate systems.

    Attributes
    ----------
    atlas : CoordinateAtlas
        Coordinate atlas the coordinate system belongs to
    label :
        Coordinate system label.
    """
    def __init__(self, label: str, atlas: CoordinateAtlas):
        """
        A base class for creating coordinate systems

        Parameters
        ----------
        atlas : CoordinateAtlas
            Coordinate atlas the coordinate system belongs to.
        label :
            Coordinate system label. Must be a node in atlas.
        """
        self.atlas = atlas
        if label not in self.atlas.nodes:
            raise ValueError(f"{label!r} is not a coordinate system in atlas")
        self._label = label
    

    def label(self) -> str:
        return self._label


class GalileanCoordinateSystem(CoordinateSystem):
    """
    A base class for celestial coordinate systems.

    Attributes
    ----------
    label :
        Coordinate system label.
    atlas : CelestialCoordinateAtlas
        Coordinate atlas the coordinate system belongs to
    """
    def __init__(self, label: str, atlas: CelestialCoordinateAtlas):
        if not isinstance(atlas, LocalGalileanCoordinateAtlas):
            raise TypeError(
                    "atlas is not an instance of LocalGalileanCoordinateAtlas")
        super().__init__(label, atlas)


    def galilean_transform_velocity_from(
        self, source: str, velocity: np.ndarray, time_interval: np.ndarray, 
        vel_units: str = "kms"
    ) -> np.ndarray:
        """
        Transform a velocity from another celestial coordinate system.

        Parameters
        ----------
        source : str
            Source coordinate system. Must be a node in atlas.
        velocity : numpy.ndarray
            Velocity in the source coordinate system.
        time_interval : numpy.ndarray
            Time interval on which the transformation is computed.
        vel_units : {'kms', 'natural'}, optional
            Units of the input velocities; 'kms' is kilometers per
            second, 'natural' is as a fraction of speed of light.

        Returns
        -------
        numpy.ndarray
            Velocity in the coordinate system in units of km/s.
        """
        return self.atlas.galilean_transform_velocity_between(
                source, self._label, velocity, time_interval, vel_units='kms')
    

    def galilean_transform_velocity_to(
        self, dest: str, velocity: np.ndarray, time_interval: np.ndarray, vel_units: str = "kms"
    ) -> np.ndarray:
        """
        Transform a velocity to another celestial coordinate system.

        Parameters
        ----------
        source : str
            Source coordinate system. Must be a node in atlas.
        velocity : numpy.ndarray
            Velocity in the source coordinate system.
        time_interval : numpy.ndarray
            Time interval on which the transformation is computed.
        vel_units : {'kms', 'natural'}, optional
            Units of the input velocities; 'kms' is kilometers per
            second, 'natural' is as a fraction of speed of light.

        Returns
        -------
        numpy.ndarray
            Velocity in the coordinate system in units of km/s.
        """
        return self.atlas.galilean_transform_velocity_between(
                self._label, dest, velocity, time_interval, vel_units=vel_units)


    def rotate_vector_from(
        self, source: str, source_vector: np.ndarray, time_interval: np.ndarray
    ) -> np.ndarray:
        """
        Rotate a vector from another celestial coordinate system.

        Parameters
        ----------
        source : str
            Source coordinate system. Must be a node in atlas.
        source_vector : numpy.ndarray
            Vector in the source coordinate system.
        time_interval : numpy.ndarray
            Time interval on which the transformation is computed.

        Returns
        -------
        numpy.ndarray
            Vector in the coordinate system.
        """
        return self.atlas.rotate_vector_between(
                source, self._label, source_vector, time_interval)


    def rotate_vector_to(
        self, dest: str, source_vector: np.ndarray, time_interval: np.ndarray
    ) -> np.ndarray:
        """
        Rotate a vector to another celestial coordinate system.

        Parameters
        ----------
        source : str
            Source coordinate system. Must be a node in atlas.
        source_vector : numpy.ndarray
            Vector in the source coordinate system.
        time_interval : numpy.ndarray
            Time interval on which the transformation is computed.

        Returns
        -------
        numpy.ndarray
            Vector in the coordinate system.
        """
        return self.atlas.rotate_vector_between(
                self._label, dest, source_vector, time_interval)


    def rotate_matrix_from(
        self, source: str, source_matrix: np.ndarray, time_interval: np.ndarray
    ) -> np.ndarray:
        """
        Rotate matrix from another celestial coordinate system.

        Parameters
        ----------
        source : str
            Source coordinate system. Must be a node in atlas.
        source_vec : numpy.ndarray
            Matrix in the source coordinate system.
        time_interval : numpy.ndarray
            Time interval on which the transformation is computed.

        Returns
        -------
        numpy.ndarray
            Matrix in the coordinate system.
        """
        return self.atlas.rotate_matrix_between(
                source, self._label, source_matrix, time_interval)


    def rotate_matrix_to(
        self, dest: str, source_matrix: np.ndarray, time_interval: np.ndarray
    ) -> np.ndarray:
        """
        Rotate matrix dest another celestial coordinate system.

        Parameters
        ----------
        source : str
            Source coordinate system. Must be a node in atlas.
        source_vec : numpy.ndarray
            Matrix in the source coordinate system.
        time_interval : numpy.ndarray
            Time interval on which the transformation is computed.

        Returns
        -------
        numpy.ndarray
            Matrix in the coordinate system.
        """
        return self.atlas.rotate_matrix_between(
                self._label, dest, source_matrix, time_interval)
    

    def rotation_matrix_from(
        self, source: str, time_interval: np.ndarray
    ) -> np.ndarray:
        return self.atlas.rotation_matrix_between(
                source, self._label,  time_interval)
    

    def rotation_matrix_to(
        self, dest: str, time_interval: np.ndarray
    ) -> np.ndarray:
        return self.atlas.rotation_matrix_between(
                self._label, dest, time_interval)


class CelestialCoordinates(GalileanCoordinateSystem):
    """Base class for celestial coordinate systems."""
    def __init__(
        self, label: str,
        atlas: typing.Optional[LocalGalileanCoordinateAtlas] = None,
        v_circ: float = 220, v_pec: np.ndarray = solar_v_pec
    ):
        """
        Parameters
        ----------
        label : str
            Label of the coordinate system. Must be a member of
        atlas : LocalGalileanCoordinateAtlas, optional
            Coordinate atlas. If nothing is given, by default an
            instance of CelestialCoordinateAtlas is created with the
            parameters v_circ and v_pec
        v_circ : float, optional
            Local circular velocity of the galaxy at the Sun's
            position.
        v_pec : numpy.ndarray, optional
            Peculiar velocity of the Sun.
        """
        self.v_circ = v_circ
        if atlas is None:
            atlas = CelestialCoordinateAtlas(v_circ, v_pec)
        super().__init__(label, atlas)


class GalacticCoordinates(CelestialCoordinates):
    """A galactic coordinate system."""
    def __init__(
        self, atlas: typing.Optional[LocalGalileanCoordinateAtlas] = None, v_circ: float = 220, v_pec: np.ndarray = solar_v_pec
    ):
        """
        Parameters
        ----------
        label : str
            Label of the coordinate system. Must be a member of
        atlas : LocalGalileanCoordinateAtlas, optional
            Coordinate atlas. If nothing is given, by default an
            instance of CelestialCoordinateAtlas is created with the
            parameters v_circ and v_pec
        v_circ : float, optional
            Local circular velocity of the galaxy at the Sun's
            position.
        v_pec : numpy.ndarray, optional
            Peculiar velocity of the Sun.
        """
        super().__init__('gal', atlas, v_circ, v_pec)



class HeliocentricCoordinates(CelestialCoordinates):
    """A heliocentric coordinate system."""
    def __init__(
        self,
        atlas: typing.Optional[LocalGalileanCoordinateAtlas] = None,
        v_circ: float = 220, v_pec: np.ndarray = solar_v_pec
    ):
        """
        Parameters
        ----------
        label : str
            Label of the coordinate system. Must be a member of
        atlas : LocalGalileanCoordinateAtlas, optional
            Coordinate atlas. If nothing is given, by default an
            instance of CelestialCoordinateAtlas is created with the
            parameters v_circ and v_pec
        v_circ : float, optional
            Local circular velocity of the galaxy at the Sun's
            position.
        v_pec : numpy.ndarray, optional
            Peculiar velocity of the Sun.
        """
        super().__init__('hel', atlas, v_circ, v_pec)


class EquatorialCoordinates(CelestialCoordinates):
    """An equatorial coordinate system."""
    def __init__(
        self,
        atlas: typing.Optional[LocalGalileanCoordinateAtlas] = None,
        v_circ: float = 220, v_pec: np.ndarray = solar_v_pec
    ):
        """
        Parameters
        ----------
        label : str
            Label of the coordinate system. Must be a member of
        atlas : LocalGalileanCoordinateAtlas, optional
            Coordinate atlas. If nothing is given, by default an
            instance of CelestialCoordinateAtlas is created with the
            parameters v_circ and v_pec
        v_circ : float, optional
            Local circular velocity of the galaxy at the Sun's
            position.
        v_pec : numpy.ndarray, optional
            Peculiar velocity of the Sun.
        """
        super().__init__('equ', atlas, v_circ, v_pec)


class HorizontalCoordinates(GalileanCoordinateSystem):
    """
    A horizontal coordinate system.

    Attributes
    ----------
    v_circ : float
        Local circular velocity of the galaxy at the Sun's position.
    v_pec : numpy.ndarray
        Peculiar velocity of the Sun.
    latitude : float
        Latitude of the point on the planet's surface in radians,
        measured in the physics convention, where it is the angle from
        the north pole.
    longitude : float
        Longitude of the point on the planet's surface in radians.
    """
    def __init__(
        self, label: str, latitude: float, longitude: float,
        atlas: CelestialCoordinateAtlas, modify_atlas: bool = False, equ_label: str = "equ"
    ):
        self.latitude = latitude
        self.longitude = longitude
        self._equ_label = equ_label
        if equ_label not in atlas.nodes:
            raise ValueError(
                f"{equ_label!r} is not a coordinate system in atlas.")
        _atlas = atlas if modify_atlas else copy.deepcopy(atlas)
        _atlas.add_equ_hor_transform(
                equ_label, label, self.latitude, self.longitude, "J2000",
                ERA_0, ERA_rate, earth_radius, angle_unit="rad")
        self.atlas = _atlas


        super().__init__(label, _atlas)


    def latitude_as_str(self, prec: int = 3, sep: str = " ") -> str:
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


    def longitude_as_str(self, prec: int = 3, sep: str = " ") -> str:
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
        Human readable string identifying the coordinate system.

        The string contains the relevant data that define the
        coordinate system: label, and the identifier string of the
        orbit. This string is NOT guaranteed to be unique. Numerical
        values are truncated for readability.

        Parameters
        ----------
        sep : str, optional
            Separator between elements of the string.

        Returns
        -------
        str
            String representation of the coordinate system.
        """
        lat_str = self.latitude_as_str(prec=4, sep='')
        lon_str = self.longitude_as_str(prec=4, sep='')
        return sep.join((self._label, lat_str, lon_str))
    
    def serializable(self):
        return {
            "class": self.__class__.__name__,
            "data": {
                "label": self._label,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "atlas": self.atlas.serializable(),
                "equ_label": self._equ_label
            }
        }


class CrystalCoordinates(GalileanCoordinateSystem):
    def __init__(
        self, source_label: str, dest_label: str,
        crystal_orientation: GalileanTransform,
        atlas: LocalGalileanCoordinateAtlas, modify_atlas: bool = False
    ):
        _atlas = atlas if modify_atlas else copy.deepcopy(atlas)
        self.atlas = _atlas
        if source_label not in self.atlas.nodes:
            raise ValueError(
                    f"{source_label!r} is not a coordinate system in "
                    "atlas.")
        _atlas.add_transform(source_label, dest_label, crystal_orientation)

        super().__init__(dest_label, _atlas)


    @classmethod
    def from_coordinates(
        cls, coordinate_system: GalileanCoordinateSystem, label: str, crystal_orientation: LocalGalileanCoordinateAtlas,
        modify_atlas: bool = False
    ):
        return cls(
            coordinate_system._label, label, crystal_orientation,
            coordinate_system.atlas, modify_atlas)
