import typing

import numpy as np

from networkx.classes.digraph import DiGraph
from networkx.exception import NetworkXNoPath
from networkx.algorithms.shortest_paths import shortest_path

from .galilean_transform import GalileanTransform

class DisconnectedAtlasError(Exception):
    pass

class NoninvertibleMapError(Exception):
    pass

class CoordinateAtlas(DiGraph):
    """
    Base class for coordinate atlases.

    A coordinate atlas is a collection of coordinate systems connected
    by some kind of transition maps. This class implements a coordinate
    atlas using a graph. It supports arbitrary data on the edges, which
    is then accessed through the 'transition' keyword on the edge.
    """

    def __init__(self, transition_maps: dict[str, typing.Any]):
        """
        Parameters
        ----------
        transition_maps : dict
            Dictionary of transition maps with tuples of coordinate
            system labels as keys.
        """
        super().__init__()
        for ends, tmap in transition_maps.items():
            if not self.has_edge(*ends):
                self.add_edge(*ends, transition=tmap)
        

        for ends, tmap in transition_maps.items():
            if not self.has_edge(*ends[::-1]):
                try:
                    self.add_edge(*ends[::-1], transition=tmap.inverse())
                except AttributeError:
                    raise NoninvertibleMapError(
                        f"{tmap.__class__.__name__} has no 'inverse' method. Transition maps in a CoordinateAtlas must be invertible.")


    def __repr__(self) -> str:
        transition_maps = {
                e: d['transition'] for e, d in dict(self.edges).items()}
        return f"{self.__class__.__name__}({transition_maps!r})"


    def transition_maps_between(self, source: str, dest: str) -> list:
        """
        Get list of transition maps needed to get from source to dest.

        Parameters
        ----------
        source :
            Source coordinate system.
        dest :
            Destination coordinate system.

        Returns
        -------
        list
            List of transition maps needed to get from source to dest.
        """
        try:
            path = list(shortest_path(self, source, dest))
        except NetworkXNoPath:
            path = None
        if path is None:
            raise DisconnectedAtlasError(
                    f"No path exists between nodes {source} and {dest}.")
        transition_map = lambda i: self[path[i]][path[i + 1]]['transition']
        return [transition_map(i) for i in range(len(path) - 1)]


class LocalGalileanCoordinateAtlas(CoordinateAtlas):
    """
    Base class for atlases of coordinate systems connected via
    Galilean transforms.

    A Galilean transform is a combination of a rotation and a boost.
    Different objects transform differently under Galilean transforms.
    Velocities are transformed via an affine transformation combining
    the rotation and the boost, while matrices and unit vectors are
    only rotated.
    """


    def __init__(self, transition_maps: dict[GalileanTransform]):
        """
        Parameters
        ----------
        transition_maps : dict[GalileanTransform]
            Dictionary of transition maps with tuples of coordinate
            system labels as keys.
        """
        for edge, map_ in transition_maps.items():
            if not isinstance(map_, GalileanTransform):
                raise TypeError(
                        "Transition maps must be instances of subclasses of "
                        "GalileanTransform. Received map on edge {edge!r} has "
                        f"type {map_.__class__.__name__} which is not a "
                        "subclass of GalileanTransform.")
        super().__init__(transition_maps)


    def galilean_transform_velocity_between(
        self, source: str, dest: str, source_velocity: np.ndarray, 
        time_interval: np.ndarray, vel_units: str = "kms"
    ) -> np.ndarray:
        """
        Galilean transform a velocity between two coordinate systems.

        Parameters
        ----------
        source : str
            Source coordinate system. Must be a node in the atlas.
        dest : str
            Destination coordinate system. Must be a node in the atlas.
        source_velocity : numpy.ndarray
            Velocity in the source coordinate system. Its last
            dimension is the number of spatial dimensions.
        time_interval : numpy.ndarray
            Time interval on which the transformation is computed.
        vel_units {'kms', 'natural'}, optional
            Units of the input velocities; 'kms' is kilometers per
            second, 'natural' is as a fraction of speed of light.

        Returns
        -------
        numpy.ndarray
            Velocity in the destination coordinate system in units of
            km/s.
        """
        if np.all(source_velocity == 0):
            res = np.zeros(3)
        else:
            res = source_velocity if (vel_units == 'kms') else source_velocity*299792.458

        transition_chain = self.transition_maps_between(source, dest)

        for transition_map in transition_chain:
                res = transition_map.transform_velocity(res, time_interval)
        return res


    def rotate_vector_between(
        self, source: str, dest: str, source_vector: np.ndarray,
        time_interval: np.ndarray
    ) -> np.ndarray:
        """
        Rotate a vector between two coordinate systems.

        Parameters
        ----------
        source : str
            Source coordinate system. Must be a node in the atlas.
        dest : str
            Destination coordinate system. Must be a node in the atlas.
        source_vector : numpy.ndarray
            Vector in the source coordinate system.
        time_interval : numpy.ndarray
            Time interval on which the transformation is computed.

        Returns
        -------
        numpy.ndarray
            Vector in the destination coordinate system.
        """
        res = np.zeros(3) if np.all(source_vector == 0) else source_vector

        transition_chain = self.transition_maps_between(source, dest)

        for transition_map in transition_chain:
            res = transition_map.transform_vector(res, time_interval)
        return res


    def rotate_matrix_between(
        self, source: str, dest: str, source_matrix: np.ndarray,
        time_interval: np.ndarray
    ) -> np.ndarray:
        """
        Rotate a matrix between two coordinate systems.

        Parameters
        ----------
        source : str
            Source coordinate system. Must be a node in the atlas.
        dest : str
            Destination coordinate system. Must be a node in the atlas.
        source_matrix : numpy.ndarray
            Matrix in the source coordinate system.
        time_interval : numpy.ndarray
            Time interval on which the transformation is computed.

        Returns
        -------
        numpy.ndarray
            Matrix in the destination coordinate system.
        """
        if np.all(source_matrix == 0):
            return np.zeros((3,3))
        else:
            res = source_matrix

        transition_chain = self.transition_maps_between(source, dest)

        for transition_map in transition_chain:
            res = transition_map.transform_matrix(res, time_interval)
        return res
    

    def rotation_matrix_between(
        self, source: str, dest: str, time_interval: np.ndarray
    ) -> np.ndarray:
        transition_chain = self.transition_maps_between(source, dest)
        res = np.identity(3)
        for transition_map in transition_chain:
            res = np.matmul(transition_map.rotation(time_interval), res)
        return res


    def add_transform(
        self, source_label: str, dest_label: str, transform: GalileanTransform
    ):
        """
        Add a transformation between two coordinate systems.

        Parameters
        ----------
        source_label : str
            Label for the source coordinate system.
        dest_label : str
            Label for the destination coordinate system.
        """
        if not isinstance(transform, GalileanTransform):
            raise TypeError(
                    "transform is not an instance of GalileanTransform")
        if not self.has_edge(source_label, dest_label):
            self.add_edge(source_label, dest_label, transition=transform)
        if not self.has_edge(dest_label, source_label):
            self.add_edge(
                    dest_label, source_label, transition=transform.inverse())

