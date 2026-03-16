from __future__ import annotations
import typing

from abc import abstractmethod

from typing import Union, Tuple, Callable, Any, Literal
from enum import Enum

import numpy as np
from numpy import ndarray

from .coordrdinate import (
    # CooEnum
    CooEnum,
    CooEnumMeta,
    CoordinateMeta,

    # Space
    Space,

    # functions
    map_angle
)

from .coordrdinate import (
    Space2D as _Space2D,
    Coordinate2D as _Coordinate2D,
    Cartesian2D as _Cartesian2D,
    Polar as _Polar,
)

from .coordrdinate import (
    Coo2DType as _Coo2DType,
    Coo3DType,
    Function,
    DataType,
    ArrayLike,

    init_data
)

Coo2DType = Union[_Coo2DType, 'Curve2D', 'Relative2D']
CooType = Union[Coo2DType, Coo3DType]
CurveNames = Literal['circle', 'lcfs', 'inside_wall', 'outside_wall']


class Coo2D(CooEnum):
    Cartesian2D: CoordinateMeta = "Cartesian2D"
    Polar: CoordinateMeta = "Polar"
    Curve2D: CoordinateMeta = "Curve2D"
    Relative2D: CoordinateMeta = "Relative2D"


class Space2D(_Space2D, coo=Coo2D):
    coo: Union[CooEnumMeta, Coo2D]

    def __init__(self, name: str = None):
        super().__init__(name)

    def Curve2D(self, angle: ArrayLike, func_name: CurveNames):
        return self.coo.Curve2D(angle, func_name, self)

    def Relative2D(self, point2d: Coo2DType, angle: ArrayLike, magnitude: ArrayLike):
        return self.coo.Relative2D(point2d, angle, magnitude, self)


class Coordinate2D(_Coordinate2D, coo=Coo2D):
    coo: Union[CooEnumMeta, Coo2D]

    def __init__(self, space: Space2D = None):
        if space is None:
            space = Space2D(name=None)
        self._space = space
        super().__init__(space)

    @property
    def space(self) -> Space2D:
        return self._space

    @abstractmethod
    def toCartesian(self, other: Union[Coo2DType, Space2D] = None) -> Coo2D.Cartesian2D:
        pass

    @abstractmethod
    def toPolar(self, other: Union[Coo2DType, Space2D] = None) -> Coo2D.Polar:
        pass


class Cartesian2D(_Cartesian2D, Coordinate2D):
    def __init__(self, x: ArrayLike, y: ArrayLike, space: Space2D = None):
        super().__init__(x, y, space)


class Polar(_Polar, Coordinate2D):
    def __init__(self, radius: ArrayLike, angle: ArrayLike, space: Space2D = None):
        super().__init__(radius, angle, space)


class Curve2D(Coordinate2D):
    __data__ = {'angle': ndarray, 'curve_func': Function}
    curve_dict = dict()

    def __init__(self, angle: ArrayLike, func_name: CurveNames, space: Space2D = None):
        super().__init__(space)
        curve_func = self.curve_dict[func_name]
        with init_data(self):
            self.angle = np.array(angle, ndmin=1, dtype=float)
            self.curve_func = curve_func

    def toCartesian(self, other: Union[Coo2DType, Space2D] = None) -> Coo2D.Cartesian2D:
        return self.toPolar().toCartesian()

    def toPolar(self, other: Union[Coo2DType, Space2D] = None) -> Coo2D.Polar:
        radius = self.curve_func(self.angle)
        return self.space.Polar(radius, self.angle)

    def __str__(self) -> str:
        return ""

    def __getitem__(self, item):
        return self.space.Curve2D(self.angle[item], self.curve_func)

    def __iter__(self):
        return iter((self.angle, self.curve_func))

    def __array__(self):
        return self.angle

    @classmethod
    def register_curve(cls, func_name, curve_function):
        cls.curve_dict[func_name] = curve_function


class Relative2D(Coordinate2D):

    __data__ = {'point2d': Coo2DType, 'angle': ndarray, 'magnitude': ndarray}

    def __init__(self,
                 point2d: Coo2DType, angle: ArrayLike, magnitude: ArrayLike,
                 space: Space2D = None):
        """
        :param point2d: The starting position of the vector
        :param angle: The relative orientation Angle of this vector (Relative to the polar axis)
        :param magnitude: The magnitude of the vector

        A two-dimensional vector has four degrees of freedom. This class uses points, angle,
        and module length to determine a vector. Angles use relative metrics. Sent from the given point,
        it points to the polar axis at 0 degrees, and the right hand helix is positive
        """
        super().__init__(space)
        with init_data(self):
            self.point2d = point2d.toPolar()
            self.angle = np.array(angle, ndmin=1, dtype=float)
            self.magnitude = np.array(magnitude, ndmin=1, dtype=float)

    def toCartesian(self, other: Union[Coo2DType, Space2D] = None) -> Coo2D.Cartesian2D:
        return self.toPolar().toCartesian()

    def toPolar(self, other: Union[Coo2DType, Space2D] = None) -> Coo2D.Polar:
        point_angle = self.point2d.angle
        absolute_angle = point_angle + np.pi + self.angle
        return self.space.Polar(self.magnitude, absolute_angle)

    def __str__(self) -> str:
        return ""


class Coo2D(CooEnum):
    Cartesian2D = Cartesian2D
    Polar = Polar
    Curve2D = Curve2D
    Relative2D = Relative2D

# a: Cartesian2D = Coo2D.Cartesian2D(1, 2)


# myspace = Space2D()
# myspace.set_axis(Coo2D.Cartesian2D, 0.5, 0.5)
# myspace.set_axis(Coo2D.Polar, 1, 1)
# p0 = Polar(0,1)
# p = Cartesian2D([0.5, 0.3, 0.9], [1.5, 1.7, 4.3])
#
#
#
# scalar = Space2D('scalar')
# vector = Space2D('vector')
# vector.lock_all_axes()
#
# car = Space2D()
# # scalar.set_axis(Coo2D.Polar, 1, 1)
# scalar.set_axis("Polar", 1, 1)
#
# p = vector.Cartesian2D(1, 2)
# p2 = scalar.Cartesian2D(3, 2)
#
# p2.set_axis(1, 2)
#
#
# arr = np.array(p)
#
# x, y = p
#
# radius, angle = p.toPolar()
#
# pass
# p4 = Coo2D.Cartesian2D(5, 5, scalar)
# p5 = p4.toPolar()
# # p_polar = p.toPolar()
# # print(p_polar.angle)
# # print(p_polar.radius)
#
# p1 = p.toCartesian()
# p2 = p1.toPolar()
# p3 = p2.toCartesian()
#
# #
# pass
