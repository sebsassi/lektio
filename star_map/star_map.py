import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

import coordinates

def parse_ra(ra: str, sep: str = " "):
    hour, minute, second = ra.split(sep)
    return (int(hour)/24 + int(minute)/1440 + float(second)/86400)*2.0*np.pi


def parse_dec(dec: str, sep: str = " "):
    deg, arcmin, arcsec = dec.split(sep)
    return (int(deg)/360 + int(arcmin)/21600 + float(arcsec)/1296000)*2.0*np.pi


def earth_rotation_angle(UT1: float):
    return 2.0*np.pi*(0.7790572732640 + 1.00273781191135448*UT1)


def lambert_azimuthal_equal_area(longitude, colatitude):
    x = np.sin(colatitude)*np.cos(longitude)
    y = np.sin(colatitude)*np.sin(longitude)
    z = np.cos(colatitude)

    xrot = y
    yrot = z
    zrot = -x

    c = np.sqrt(2.0/(1.0 - zrot))
    xproj = c*xrot
    yproj = c*yrot

    return xproj, yproj

def flip_x(x, y):
    return -x, y

def gnomonic(longitude, colatitude):
    x = np.sin(colatitude)*np.cos(longitude)
    y = np.sin(colatitude)*np.sin(longitude)
    z = np.cos(colatitude)

    xrot = y
    yrot = z
    zrot = -x

    xproj = xrot/zrot
    yproj = yrot/zrot

    return xproj, yproj


class HorizontalTransformer:
    def __init__(self, latitude, longitude):
        self.coords = coordinates.HorizontalCoordinates("hor", latitude, longitude, coordinates.CelestialCoordinateAtlas())
    

    def from_equ(self, ra, dec, UT1):
        ra_ = parse_ra(ra) if isinstance(ra, str) else ra
        dec_ = parse_dec(dec) if isinstance(dec, str) else dec
        #return -ra_, 0.5*np.pi - dec_
        pos_equ = coordinates.angles_to_vec(np.stack((dec_, ra_), axis=-1), convention="geographical")
        pos_hor = self.coords.rotate_vector_from("equ", pos_equ, UT1)
        alt_az_len = coordinates.cartesian_to_spherical(pos_hor, convention="physics")
        if alt_az_len.ndim == 1:
            alt_az_len[1] + 0.5*np.pi, alt_az_len[0]
        else:
            return alt_az_len[...,1] + 0.5*np.pi, alt_az_len[...,0]


    def vlab(self, UT1):
        vel = -self.coords.galilean_transform_velocity_from("gal", np.zeros(3), UT1)
        alt_az_len = coordinates.cartesian_to_spherical(vel, convention="physics")
        if alt_az_len.ndim == 1:
            alt_az_len[1] + 0.5*np.pi, alt_az_len[0]
        else:
            return alt_az_len[...,1] + 0.5*np.pi, alt_az_len[...,0]


def constellation_to_proj(transformer, constellation, time):
    return {
        k: flip_x(*lambert_azimuthal_equal_area(*transformer.from_equ(*pos, time))) for k, pos in constellation.items()
    }


def serialize_np_array(arr):
    if arr.ndim == 0: return float(arr)
    elif arr.ndim == 1 and arr.size == 1: return float(arr[0])
    elif arr.ndim == 1: return list(arr)
    else: return arr

def make_constellation_graph(name, stars):
    if name == "Cygnus":
        return [
            np.squeeze(np.array([
                stars["Deneb"],
                stars["gamma Cyg"],
                stars["eta Cyg"],
                stars["Albireo"]
            ])),
            np.squeeze(np.array([
                stars["zeta Cyg"],
                stars["epsilon Cyg"],
                stars["gamma Cyg"],
                stars["delta Cyg"],
                stars["iota Cyg"],
                stars["kappa Cyg"]
            ]))
        ]
    elif name == "Ursa Major":
        return [
            np.squeeze(np.array([
                stars["Alkaid"],
                stars["Mizar"],
                stars["Alioth"],
                stars["Megrez"],
                stars["Phecda"],
                stars["Merak"],
                stars["Dubhe"],
                stars["Megrez"]
            ]))
        ]
    elif name == "Ursa Minor":
        return [
            np.squeeze(np.array([
                stars["Polaris"],
                stars["delta UMi"],
                stars["eta UMi"],
                stars["zeta UMi"],
                stars["eta UMi"],
                stars["gamma UMi"],
                stars["Kochab"],
                stars["zeta UMi"]
            ]))
        ]

if __name__ == "__main__":
    constellations = {
        "Cygnus": {
            "Deneb": ("20 41 25.91", "45 16 49.2"),
            "gamma Cyg": ("20 22 13.70", "40 15 24.1"),
            "epsilon Cyg": ("20 46 12.43", "33 58 10.0"),
            "delta Cyg": ("19 44 58.44", "45 07 50.5"),
            "Albireo": ("19 30 43.29", "27 57 34.9"),
            "zeta Cyg": ("21 12 56.18", "30 13 37.5"),
            "iota Cyg": ("19 29 42.34", "51 43 46.1"),
            "kappa Cyg": ("19 17 6.11", "53 22 5.4"),
            "eta Cyg": ("19 56 18.40", "35 5 0.6"),
            "theta Cyg": ("19 36 26.54", "50 13 3.7")
        },
        "Ursa Major": {
            "Alioth": ("12 54 1.63", "55 57 35.4"),
            "Dubhe": ("11 3 43.84", "61 45 4.0"),
            "Alkaid": ("13 47 32.55", "49 18 47.9"),
            "Mizar": ("13 23 55.54", "54 55 31.3"),
            "Merak": ("11 1 50.39", "56 22 56.4"),
            "Phecda": ("11 53 49.74", "53 41 41.0"),
            "Megrez": ("12 15 25.45", "57 1 57.4")
        },
        "Ursa Minor": {
            "Polaris": ("2 31 47.08", "89 15 50.9"),
            "Kochab": ("14 50 42.40", "74 9 19.7"),
            "gamma UMi": ("15 20 43.75", "71 50 2.3"),
            "epsilon UMi": ("16 45 58.16", "82 2 14.1"),
            "zeta UMi": ("15 44 3.46", "77 47 40.2"),
            "delta UMi": ("17 32 12.90", "86 35 10.8"),
            "eta UMi": ("16 17 30.50", "75 45 16.9")
        }
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("date", type=str)
    args = parser.parse_args()

    time = coordinates.TimeInterval(args.date)

    latitude = 60.178269*np.pi/180
    longitude = 24.938776*np.pi/180
    transformer = HorizontalTransformer(latitude, longitude)

    constellations_proj = {
        name: constellation_to_proj(transformer, constellation, time) for name, constellation in constellations.items()
    }

    vlab_proj = flip_x(*lambert_azimuthal_equal_area(*transformer.vlab(time)))

    for name, constellation in constellations_proj.items():
        xy = np.array(list(constellation.values()))
        plt.scatter(xy[:,0], xy[:,1])
    plt.scatter(vlab_proj[0], vlab_proj[1])
    #plt.xlim(-2, 2)
    #plt.ylim(-2, 2)
    plt.show()

    #with open("star_map.json", "w") as f:
    #    json.dump({"constellations": constellations_proj, "vlab": vlab_proj}, f, default=serialize_np_array)
    
    for name, constellation in constellations_proj.items():
        graph = make_constellation_graph(name, constellation)
        with open(f"constellation_{name.replace(' ', '-')}_{args.date}.dat", "w") as f:
            for line in graph:
                print(line)
                for coord in line:
                    f.write(f"{coord[0]} {coord[1]}\n")
                f.write("\n")
    
    np.savetxt(f"vlab_{args.date}.dat", np.array(vlab_proj).T)
