
from __future__ import annotations
from abc import ABC, abstractmethod, ABCMeta
from typing import (
    Union, Tuple, Dict, ForwardRef,
    NewType, TypeVar, Callable, List
)
# from enum import Enum, unique, EnumMeta
# import time

import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike as _ArrayLike

# from typings import ArrayLike, CooClsType, CooType, Coo2DType, Coo3DType, CooClsInputType, DataType

__all__ = ['ArrayLike', 'CooClsType', 'CooClsInputType', 'Coo2DType',
           'Coo3DType', 'CooType', 'Function', 'DataType',
           'CoordinateMeta', 'Coordinate', 'CooEnumMeta', 'CooEnum',
           'Coo2D', 'Coo3D', 'Space', 'Space2D', 'Space3D', 'init_data',
           'Coordinate2D', 'Cartesian2D', 'Polar',
           'Cartesian3D', 'Coordinate3D', 'Cylindrical', 'Spherical']

ArrayLike = TypeVar('ArrayLike', bound=_ArrayLike)

CooClsType = Union['CoordinateMeta']
CooClsInputType = Union["CoordinateMeta", str]

Coo2DType = Union["Cartesian2D", "Polar"]
Coo3DType = Union["Cartesian3D", "Cylindrical", "Spherical"]
CooType = Union[Coo2DType, Coo3DType]

Function = NewType('Function', Callable)
DataType = Union[ndarray, CooType, Function]


def _get_type_str(dtype: Union[str, ForwardRef, CoordinateMeta, ndarray, Function]):
    if isinstance(dtype, ForwardRef):
        return dtype.__forward_arg__
    if isinstance(dtype, CoordinateMeta):
        return dtype.__name__
    if dtype == ndarray:
        return dtype.__name__
    if dtype == Function:
        return dtype.__name__
    if isinstance(dtype, str):
        return dtype
    raise TypeError("Unsupported data type: {0}".format(dtype))


class CoordinateMeta(ABCMeta):
    __data__: Dict[str, DataType] = None

    # Initialize time as list and runtime as str.
    # A list to the name of data which type is ndarray or CooType
    __data_shape_str__: Union[str, list[str]] = []

    def __new__(mcls, name, bases, namespace, coo=None):
        return super().__new__(mcls, name, bases, namespace)

    def __init__(cls, name, bases, namespace, coo=None):
        super().__init__(name, bases, namespace)
        if coo is not None:
            cls.coo = coo
        if hasattr(cls, 'coo'):
            if name in cls.coo.__members__:
                setattr(cls.coo, name, cls)

        # check __data__
        if cls.__data__ is None:
            return
        valid = False
        cls.__data_shape_str__ = []
        # coos = set([coo.__forward_arg__ for coo in get_args(CooType)])
        for data_key, dtype in cls.__data__.items():
            if isinstance(dtype, str):
                continue
            if dtype is Function:
                continue
            if dtype is ndarray:
                valid = True
                cls.__data_shape_str__.append(data_key)
                continue

            # TODO: CooType support
            # # check if it is in CooType
            # # When ndarray is not present in __data__,
            # # there should be at least one type which in CooType
            # args = get_args(dtype)
            # if len(args) == 0:
            #     args = (dtype,)
            # type_str = [_get_type_str(arg) for arg in args]
            # if set(type_str) <= coos:
            #     valid = True
            #     cls.__data_shape_str__.append(data_key)
            # else:
            #     raise TypeError("Unsupported data type: {0}".format(type_str))
        if valid is False:
            raise TypeError("At least one of the ndarray and CooType types needs to be inside __data__")

    @classmethod
    def init_axis(cls) -> tuple:
        pass


class CooEnumMeta(type):
    def __new__(mcls, name, bases, namespace):
        if len(bases) != 0:
            base = bases[0].__name__
            if base != "CooEnum":
                raise TypeError("{0}: cannot extend enumeration '{1}'".format(base, name))
        return super().__new__(mcls, name, bases, namespace)

    @property
    def __members__(cls) -> dict[str, CooClsType]:
        members = dict()
        for attr in dir(cls):
            if not attr.startswith("_"):
                members[attr] = cls.__dict__[attr]
        return members


class CooEnum(metaclass=CooEnumMeta):
    pass


class Coo2D(CooEnum):
    # str only when defined
    Cartesian2D: CoordinateMeta = "Cartesian2D"  # str only when defined
    Polar: CoordinateMeta = "Polar"


class Coo3D(CooEnum):
    # str only when defined
    Cartesian3D: CoordinateMeta = "Cartesian3D"
    Cylindrical: CoordinateMeta = "Cylindrical"
    Spherical: CoordinateMeta = "Spherical"


def _get_coo_cls_str(var: CooClsInputType):
    if isinstance(var, str):
        return var
    elif isinstance(var, CoordinateMeta):
        return var.__name__
    else:
        raise TypeError("Unsupported input type")


class Space(ABC):
    coo: CooEnumMeta = None
    _axes: Dict[str, tuple]
    _axes_lock: Dict[str, bool]

    def __init__(self, name: str = None):
        if name is None:
            name = 'space_id' + str(id(self))
        self.name = name

        self._axes = dict()
        self._axes_lock = dict()
        for name, coordinate_cls in self.coo.__members__.items():
            self._axes[name] = coordinate_cls.init_axis()
            self._axes_lock[name] = False

    def __init_subclass__(cls, **kwargs):
        if 'coo' in kwargs:
            cls.coo = kwargs['coo']


class Space2D(Space, coo=Coo2D):
    coo: Union[CooEnumMeta, Coo2D]

    @property
    def axes(self):
        return self._axes

    def get_axis(self, coordinate_cls: CooClsInputType):
        name = _get_coo_cls_str(coordinate_cls)
        return self._axes[name]

    def set_axis(self, coordinate_cls: CooClsInputType, x: float, y: float):
        name = _get_coo_cls_str(coordinate_cls)
        if name in self.coo.__members__:
            if self._axes_lock[name] is False:
                self._axes[name] = (float(x), float(y))
            else:
                raise AttributeError("Cannot set a new position on the locked axis")
        else:
            raise TypeError("Unsupported coordinate")

    def lock_axis(self, coordinate_cls: CooClsInputType):
        name = _get_coo_cls_str(coordinate_cls)
        if name in self.coo.__members__:
            self._axes_lock[name] = True
        else:
            raise TypeError("Unsupported coordinate")

    def unlock_axis(self, coordinate_cls: CooClsInputType):
        name = _get_coo_cls_str(coordinate_cls)
        if name in self.coo.__members__:
            self._axes_lock[name] = False
        else:
            raise TypeError("Unsupported coordinate")

    def lock_all_axes(self):
        for name in self.coo.__members__:
            self._axes_lock[name] = True

    def unlock_all_axes(self):
        for name in self.coo.__members__:
            self._axes_lock[name] = False

    def Cartesian2D(self, x: ArrayLike, y: ArrayLike):
        return self.coo.Cartesian2D(x, y, self)

    def Polar(self, radius: ArrayLike, angle: ArrayLike):
        return self.coo.Polar(radius, angle, self)


class Space3D(Space, coo=Coo3D):
    pass


class init_data:
    def __init__(self, coordinate_instance: CooType):
        self.coo_inst = coordinate_instance

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    def end(self):
        data_list: List[Union[ndarray, CooType]] = []
        data_size_large_than_one = {}
        for data_key in self.coo_inst.__data_shape_str__:
            data = getattr(self.coo_inst, data_key)
            data_list.append(data)
            if data.size > 1:
                data_size_large_than_one[data_key] = data
        # shape check, point self.shape to data.shape
        if len(data_size_large_than_one) == 0:  # all size equal to 1
            self.coo_inst.__data_shape_str__ = self.coo_inst.__data_shape_str__[0]
            return
        # broadcast shapes of 1 to N are allowed
        # all size larger than 1
        # shape check
        first_data_key, first_data = list(data_size_large_than_one.items())[0]
        for data in data_size_large_than_one.values():
            if data.shape != first_data.shape:
                raise RuntimeError("Data shape mismatch: {0} and {1}".format(
                    data.shape, data_list[0].shape)
                )
        # str assign
        self.coo_inst.__data_shape_str__ = first_data_key


class Coordinate(metaclass=CoordinateMeta):

    @classmethod
    @abstractmethod
    def init_axis(cls) -> tuple:
        pass

    @property
    def shape(self):
        data: Union[ndarray, CooType] = getattr(self, self.__data_shape_str__)
        return data.shape

    @property
    def size(self):
        data: Union[ndarray, CooType] = getattr(self, self.__data_shape_str__)
        return data.size


class Coordinate2D(Coordinate, coo=Coo2D):
    coo: Coo2D

    def __init__(self, space: Space2D = None):
        if space is None:
            space = Space2D(name=None)
        self._space = space

    @classmethod
    def init_axis(cls) -> Tuple[float, float]:
        return 0.0, 0.0

    @property
    def axis(self):
        return self.space.get_axis(self.__class__.__name__)

    def set_axis(self, x: float, y: float):
        self.space.set_axis(self.__class__.__name__, x, y)

    @property
    def space(self) -> "Space2D":
        return self._space

    @space.setter
    def space(self, space: "Space2D"):
        self.set_space(space)

    @property
    def space_name(self) -> str:
        return self._space.name

    @space_name.setter
    def space_name(self, space_name: str):
        self.space.name = space_name

    def set_space(self, space: "Space2D"):
        if space.__class__.__name__ == "Space2D":
            self.space = space
        else:
            raise TypeError("Wrong space type")

    @abstractmethod
    def toCartesian(self, other: Union[Coo2DType, Space2D] = None) -> Coo2D.Cartesian2D:
        pass

    @abstractmethod
    def toPolar(self, other: Union[Coo2DType, Space2D] = None) -> Coo2D.Polar:
        pass


class Cartesian2D(Coordinate2D):
    __data__ = {'x': ndarray, 'y': ndarray}

    def __init__(self, x: ArrayLike, y: ArrayLike, space: Space2D = None):
        # space = None -> init a new space: avg 6.7μs
        # space = space -> no need to init space: avg 0μs
        super().__init__(space)
        # init_data process: avg 2μs
        with init_data(self):
            # init numpy array: avg 0.32×2=0.64μs
            self.x = np.array(x, ndmin=1, dtype=float)
            self.y = np.array(y, ndmin=1, dtype=float)

    def toCartesian(self, other: Union[Coo2DType, Space2D] = None):
        return self

    def toPolar(self, other: Union[Coo2DType, Space2D] = None):
        # toPolar: avg at least 12μs (slow because of python)
        radius = np.sqrt(np.square(self.x + self.axis[0] - self.space.axes["Polar"][0]) +
                         np.square(self.y + self.axis[1] - self.space.axes["Polar"][1]))
        angle = map_angle(
            np.arctan2(self.y + self.axis[1] - self.space.axes["Polar"][1],
                       self.x + self.axis[0] - self.space.axes["Polar"][0])
        )
        return self.space.Polar(radius, angle)

    def __str__(self) -> str:
        return ""

    def __getitem__(self, item):
        return self.space.Cartesian2D(self.x[item], self.y[item])

    def __iter__(self):
        if self.size == 1:
            return iter((self.x[0], self.y[0]))
        return iter((self.x, self.y))

    def __array__(self):
        return np.array((*self,))


class Polar(Coordinate2D):
    __data__ = {'radius': ndarray, 'angle': ndarray}

    def __init__(self, radius: ArrayLike, angle: ArrayLike, space: Space2D = None):
        super().__init__(space)
        with init_data(self):
            self.radius = np.array(radius, ndmin=1, dtype=float)
            self.angle = np.array(angle, ndmin=1, dtype=float)

    def toCartesian(self, other: Union[Coo2DType, Space2D] = None):
        x = self.radius * np.cos(self.angle) + self.axis[0] - self.space.axes["Cartesian2D"][0]
        y = self.radius * np.sin(self.angle) + self.axis[1] - self.space.axes["Cartesian2D"][1]
        return self.space.Cartesian2D(x, y)

    def toPolar(self, other: Union[Coo2DType, Space2D] = None):
        return self

    def __str__(self) -> str:
        return ""

    def __getitem__(self, item):
        return self.space.Polar(self.radius[item], self.angle[item])

    def __iter__(self):
        if self.size == 1:
            return iter((self.radius[0], self.angle[0]))
        return iter((self.radius, self.angle))

    def __array__(self):
        return np.array((*self,))


class Coordinate3D(Coordinate, coo=Coo3D):
    @classmethod
    def init_axis(cls) -> Tuple[float, float, float]:
        return 0.0, 0.0, 0.0

    @abstractmethod
    def toCartesian3D(self) -> Coo3D.Cartesian3D:
        pass

    @abstractmethod
    def toCylindrical(self) -> Coo3D.Cylindrical:
        pass

    @abstractmethod
    def toSpherical(self) -> Coo3D.Spherical:
        pass


class Cartesian3D(Coordinate3D):
    def toCartesian3D(self) -> Coo3D.Cartesian3D:
        return self

    def toCylindrical(self) -> Coo3D.Cylindrical:
        pass

    def toSpherical(self) -> Coo3D.Spherical:
        pass


class Cylindrical(Coordinate3D):
    def toCartesian3D(self) -> Coo3D.Cartesian3D:
        pass

    def toCylindrical(self) -> Coo3D.Cylindrical:
        return self

    def toSpherical(self) -> Coo3D.Spherical:
        pass


class Spherical(Coordinate3D):
    def toCartesian3D(self) -> Coo3D.Cartesian3D:
        pass

    def toCylindrical(self) -> Coo3D.Cylindrical:
        pass

    def toSpherical(self) -> Coo3D.Spherical:
        return self


class Coo2D(CooEnumMeta):
    Cartesian2D = Cartesian2D
    Polar = Polar


class Coo3D(CooEnumMeta):
    Cartesian2D = Cartesian3D
    Cylindrical = Cylindrical
    Spherical = Spherical


def map_angle(angle):
    """
    map angle to [0, 2*pi)
    """
    return angle % (2 * np.pi)


# TODO: 不同坐标轴之间的投影、2D、3D、2D+3D
# sp = Space2D()
# a = Cartesian2D([1, 4, 1], [2, 2, 3], sp)
# tic = time.time()
# for i in range(int(1e5)):
#     a.toPolar()
# toc = time.time()
# print(toc - tic)
#
# tic = time.time()
# for i in range(int(1e5)):
#     a = np.array([1, 4, 1])
#     b = np.array([2, 2, 3])
# toc = time.time()
# print(toc - tic)
# b = Cartesian2D(4, 2)
# print(Cartesian2D)

pass
# TODO: 坐标系旋转

# TODO: 一个坐标系的实例代表这整个坐标系，应当包括点集、子集等


# myspace = Space2D()
# myspace.set_axis(Coo2D.Polar, 1, 1)
#
# p = Cartesian2D(1, 2, myspace)
#
# p1 = p.toCartesian()
# p2 = p1.toPolar()
# p3 = p2.toCartesian()
