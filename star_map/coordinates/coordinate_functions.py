import numpy as np

def spher_vec_to_vec(
    spher_vec: np.ndarray, convention: str = "physics"
) -> np.ndarray:
    """
    Take a triple of spherical coordinates to a Cartesian vector.

    Parameters
    ----------
    spher_vec : numpy.ndarray
        array with shape (...,3) specifying a 3-D vector (or 3-D
        vectors) in spherical coordinates. The order is
        (altitude/inclination, azimuth, length).
    convention : {'physics', 'geographical'}
        Convention used by the angles. In the 'pyhsics' convention the
        first angle is the angle from the z-axis (inclination), and
        in the 'geographical' convention the first angle is the angle
        from the xy-plane (altitude).

    Returns
    -------
    numpy.ndarray
        Vectors corresponding to spherical coordinates
    """

    if spher_vec.shape[-1] != 3:
        raise ValueError("The last dimension of `spher_vec` must be 3.")

    altitude = spher_vec[...,0]
    azimuth = spher_vec[...,1]
    length = spher_vec[...,2]

    if convention == 'geographical':
        rotate = (np.cos(altitude), np.sin(altitude))
    elif convention == 'physics':
        rotate = (np.sin(altitude), np.cos(altitude))
    else:
        raise ValueError(
                f"Angle convention {convention} not recognized; `convention` " "must be one of {'geographical', 'physics'}")

    return length[..., np.newaxis]*np.stack(
            (rotate[0]*np.cos(azimuth), rotate[0]*np.sin(azimuth), rotate[1]),
            axis=-1)


def angles_to_vec(
    angles: np.ndarray, convention: str = "physics"
) -> np.ndarray:
    """
    Take pair of spherical angles to Cartesian unit vector.

    Parameters
    ----------
    angles : numpy.ndarray
        array with shape (...,2) specifying a 3-D unit vector (or unit
        vectors) in spherical coordinates. The order is
        (altitude/inclination, azimuth).
    convention : {'physics', 'geographical'}, optional
        Convention used by the angles. In the 'pyhsics' convention the
        first angle is the angle from the z-axis (inclination), and
        in the 'geographical' convention the first angle is the angle
        from the xy-plane (altitude).

    Returns
    -------
    numpy.ndarray
        Unit vector corresponding to the angles.
    """

    if angles.shape[-1] != 2:
        raise ValueError("The last dimension of `angles` must be 2.")

    s = angles.shape[:-1]
    return spher_vec_to_vec(
                np.stack((angles[...,0], angles[...,1], np.ones(s)), axis=-1),
                convention)

def cartesian_to_spherical(
    vec: np.ndarray, convention: str = "physics"
) -> np.ndarray:
    length = np.sqrt(np.sum(vec*vec, axis=-1))
    altitude = np.arccos(vec[...,2]/length)
    azimuth = np.arctan2(vec[...,1], vec[...,0])
    if convention == "geographical":
        altitude = 0.5*np.pi - altitude
        azimuth = azimuth - 2.0*np.pi*np.heaviside(azimuth - np.pi, 0.0)
    return np.stack((altitude, azimuth, length), axis=-1)
