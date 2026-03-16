import warnings

import scipy
from scipy import interpolate
import numpy as np
import itertools
import matplotlib
import matplotlib.path
# import matplotlib._contour as _contour
import contourpy


from .utils_base import *


def contourPaths(x, y, Z, levels, remove_boundary_points=False, smooth_factor=1):
    """
    :param x: 1D x coordinate

    :param y: 1D y coordinate

    :param Z: 2D data

    :param levels: levels to trace

    :param remove_boundary_points: remove traces at the boundary

    :param smooth_factor: smooth contours by cranking up grid resolution

    :return: list of segments
    """

    sf = int(round(smooth_factor))
    if sf > 1:
        x = scipy.ndimage.zoom(x, sf)
        y = scipy.ndimage.zoom(y, sf)
        Z = scipy.ndimage.zoom(Z, sf)

    [X, Y] = np.meshgrid(x, y)
    # contour_generator = _contour.QuadContourGenerator(X, Y, Z, None, True, 0)
    contour_generator = contourpy.contour_generator(X, Y, Z)

    mx = min(x)
    Mx = max(x)
    my = min(y)
    My = max(y)

    allsegs = []
    for level in levels:
        verts = contour_generator.create_contour(level)

        segs = verts

        if not remove_boundary_points:
            segs_ = segs
        else:
            segs_ = []
            for segarray in segs:
                segarray = np.array(segarray)
                x_ = segarray[:, 0]
                y_ = segarray[:, 1]
                valid = []
                for i in range(len(x_) - 1):
                    if np.isclose(x_[i], x_[i + 1]) and (np.isclose(x_[i], Mx) or np.isclose(x_[i], mx)):
                        continue
                    if np.isclose(y_[i], y_[i + 1]) and (np.isclose(y_[i], My) or np.isclose(y_[i], my)):
                        continue
                    valid.append((x_[i], y_[i]))
                    if i == len(x_):
                        valid.append(x_[i + 1], y_[i + 1])
                if len(valid):
                    segs_.append(np.array(valid))

        segs = list(map(matplotlib.path.Path, segs_))
        allsegs.append(segs)
    return allsegs


# -----------
# fit parabolas
# -----------
def parabola(x, y):
    """
    y = a*x^2 + b*x + c

    :return: a,b,c
    """
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise np.linalg.LinAlgError('parabola could not be fit with x=%s y=%s' % (x, y))
    # A=np.matrix([[x[0]**2,x[0],1],[x[1]**2,x[1],1],[x[2]**2,x[2],1]])
    # a,b,c=np.array(A.I*np.matrix(y[0:3]).T).flatten().tolist()
    # polyfit is equally fast but more robust when x values are similar, and matrix A becomes singular, eg.
    # x,y=[-1E-16,1,1+1E-16],[1,1,1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.exceptions.RankWarning)
        a, b, c = np.polyfit(x, y, 2)
    return a, b, c


def parabolaMax(x, y, bounded=False):
    """
    Calculate a parabola through x,y, then return the extremum point of
    the parabola

    :param x: At least three abcissae points
    :param y: The corresponding ordinate points
    :param bounded: False, 'max', or 'min'
          - False: The extremum is returned regardless of location relative to x (default)
          - 'max' ('min'): The extremum location must be within the bounds of x, and if not return the location and value of max(y) (min(y))
    :return: x_max,y_max - The location and value of the extremum
    """
    a, b, c = parabola(x, y)
    x_max = -b / (2 * a)
    y_max = a * x_max ** 2 + b * x_max + c
    if bounded and (x_max > max(x) or x_max < min(x)):
        # printe('ParabolaMax found the maximum outside the input abcissae; using arg%s(y) instead' % bounded)
        iy = eval('arg%s(y)' % bounded)
        x_max = x[iy]
        y_max = y[iy]
    return x_max, y_max


def parabolaCycle(X, Y, ix):
    ix = np.array([-1, 0, 1]) + ix

    if (X[0] - X[-1]) < 1e-15:
        if ix[0] < 0:
            ix[0] = len(X) - 1 - 1
        if ix[-1] > (len(X) - 1):
            ix[-1] = 0 + 1
    else:
        if ix[0] < 0:
            ix[0] = len(X) - 1
        if ix[-1] > (len(X) - 1):
            ix[-1] = 0

    return parabola(X[ix], Y[ix])


def parabolaMaxCycle(X, Y, ix, bounded=False):
    """
    Calculate a parabola through X[ix-1:ix+2],Y[ix-1:ix+2], with proper
    wrapping of indices, then return the extremum point of the parabola

    :param X: The abcissae points: an iterable to be treated as periodic
    :param y: The corresponding ordinate points
    :param ix: The index of X about which to find the extremum
    :param bounded: False, 'max', or 'min'
          - False: The extremum is returned regardless of location relative to x (default)
          - 'max' ('min'): The extremum location must be within the bounds of x, and if not return the location and value of max(y) (min(y))
    :return: x_max,y_max - The location and value of the extremum
    """
    ix = np.array([-1, 0, 1]) + ix

    if (X[0] - X[-1]) < 1e-15:
        if ix[0] < 0:
            ix[0] = len(X) - 1 - 1
        if ix[-1] > (len(X) - 1):
            ix[-1] = 0 + 1
    else:
        if ix[0] < 0:
            ix[0] = len(X) - 1
        if ix[-1] > (len(X) - 1):
            ix[-1] = 0

    try:
        return parabolaMax(X[ix], Y[ix], bounded=bounded)
    except Exception:
        # printe([ix, X[ix], Y[ix]])
        # warnings.warn("[ix, X[ix], Y[ix]]")
        return X[ix[1]], Y[ix[1]]


def paraboloid(x, y, z):
    """
    z = ax*x^2 + bx*x + ay*y^2 + by*y + c

    NOTE: This function uses only the first 5 points of the x, y, z arrays
    to evaluate the paraboloid coefficients

    :return: ax,bx,ay,by,c
    """
    if np.any(np.isnan(x.flatten())) or np.any(np.isnan(y.flatten())) or np.any(np.isnan(z.flatten())):
        raise np.linalg.LinAlgError('paraboloid could not be fit with x=%s y=%s z=%s' % (x, y, z))
    A = []
    for k in range(5):
        A.append([x[k] ** 2, x[k], y[k] ** 2, y[k], 1])
    A = np.array(A)
    ax, bx, ay, by, c = np.dot(np.linalg.inv(A), np.array(z[:5]))
    return ax, bx, ay, by, c


def reverse_enumerate(l):
    return zip(range(len(l) - 1, -1, -1), reversed(l))


# -----------
# interpolation
# -----------
class RectBivariateSplineNaN:
    def __init__(self, Z, R, Q, *args, **kw):
        tmp = Q.copy()
        bad = np.isnan(tmp)
        self.thereAreNaN = False
        if bad.any():
            self.thereAreNaN = True
            tmp[bad] = 0
            self.mask = interpolate.RectBivariateSpline(Z, R, bad, kx=1, ky=1)
        self.spline = interpolate.RectBivariateSpline(Z, R, tmp, *args, **kw)

    def __call__(self, *args):
        tmp = self.spline(*args)
        if self.thereAreNaN:
            mask = self.mask(*args)
            tmp[mask > 0.01] = np.nan
        return tmp

    def ev(self, *args):
        tmp = self.spline.ev(*args)
        if self.thereAreNaN:
            mask = self.mask.ev(*args)
            tmp[mask > 0.01] = np.nan
        return tmp


def deriv(x, y):
    """
    This function returns the derivative of the 2nd order lagrange interpolating polynomial of y(x)
    When re-integrating, to recover the original values `y` use `cumtrapz(dydx,x)`

    :param x: x axis array

    :param y: y axis array

    :return: dy/dx
    """
    x = np.array(x)
    y = np.array(y)

    def dlip(ra, r, f):
        '''dlip - derivative of lagrange interpolating polynomial'''
        r1, r2, r3 = r
        f1, f2, f3 = f
        return (
                ((ra - r1) + (ra - r2)) / (r3 - r1) / (r3 - r2) * f3
                + ((ra - r1) + (ra - r3)) / (r2 - r1) / (r2 - r3) * f2
                + ((ra - r2) + (ra - r3)) / (r1 - r2) / (r1 - r3) * f1
        )

    return np.array(
        [dlip(x[0], x[0:3], y[0:3])]
        + list(dlip(x[1:-1], [x[0:-2], x[1:-1], x[2:]], [y[0:-2], y[1:-1], y[2:]]))
        + [dlip(x[-1], x[-3:], y[-3:])]
    )


def line_intersect(path1, path2, return_indices=False):
    """
    intersection of two 2D paths

    :param path1: array of (x,y) coordinates defining the first path

    :param path2: array of (x,y) coordinates defining the second path

    :param return_indices: return indices of segments where intersection occurred

    :return: array of intersection points (x,y)
    """

    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value.*')

    ret = []
    index = []

    path1 = np.array(path1).astype(float)
    path2 = np.array(path2).astype(float)

    switch = False
    if len(path1[:, 0]) > len(path2[:, 0]):
        path1, path2 = path2, path1
        switch = True

    m1 = (path1[:-1, :] + path1[1:, :]) / 2.0
    m2 = (path2[:-1, :] + path2[1:, :]) / 2.0
    l1 = np.sqrt(np.diff(path1[:, 0]) ** 2 + np.diff(path1[:, 1]) ** 2)
    l2 = np.sqrt(np.diff(path2[:, 0]) ** 2 + np.diff(path2[:, 1]) ** 2)

    for k1 in range(len(path1) - 1):
        d = np.sqrt((m2[:, 0] - m1[k1, 0]) ** 2 + (m2[:, 1] - m1[k1, 1]) ** 2)
        for k2 in np.where(d <= (l2 + l1[k1]) / 2.0)[0]:
            tmp = _seg_intersect(path1[k1], path1[k1 + 1], path2[k2], path2[k2 + 1])
            if tmp is not None:
                index.append([k1, k2])
                ret.append(tmp)

    ret = np.array(ret)

    if not len(ret):
        return []

    if switch:
        index = np.array(index)
        i = np.argsort(index[:, 1])
        index = [index[i, 1], index[i, 0]]
        index = np.array(index).T.astype(int).tolist()
        ret = ret[i]

    if return_indices:
        return ret, index

    return ret


def _ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) >= (B[1] - A[1]) * (C[0] - A[0])


def _intersect(A, B, C, D):
    return _ccw(A, C, D) != _ccw(B, C, D) and _ccw(A, B, C) != _ccw(A, B, D)


def _perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def _seg_intersect(a1, a2, b1, b2):
    if not _intersect(a1, a2, b1, b2):
        return None
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = _perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom) * db + b1


class interp1e(interpolate.interp1d):
    """
    Shortcut for scipy.interpolate.interp1d with fill_value='extrapolate' and bounds_error=False as defaults
    """

    __doc__ += (
        interpolate.interp1d.__doc__.replace('\n    ----------\n', ':\n')
        .replace('\n    -------\n', ':\n')
        .replace('\n    --------\n', ':\n')
    )

    def __init__(self, x, y, *args, **kw):
        kw.setdefault('fill_value', 'extrapolate')
        kw.setdefault('bounds_error', False)
        interpolate.interp1d.__init__(self, x, y, *args, **kw)


def centroid(x, y):
    """
    Calculate centroid of polygon

    :param x: x coordinates of the polygon

    :param y: y coordinates of the polygon

    :return: tuple with x and y coordinates of centroid
    """
    x = np.array(x)
    y = np.array(y)
    dy = np.diff(y)
    dx = np.diff(x)
    x0 = (x[1:] + x[:-1]) * 0.5
    y0 = (y[1:] + y[:-1]) * 0.5
    A = np.sum(dy * x0)
    x_c = -np.sum(dx * y0 * x0) / A
    y_c = np.sum(dy * x0 * y0) / A
    return x_c, y_c


def pack_points(n, x0, p):
    """
    Packed points distribution between -1 and 1

    :param n: number of points

    :param x0: pack points around `x0`, a float between -1 and 1

    :param p: packing proportional to `p` factor >0

    :return: packed points distribution between -1 and 1
    """
    x = np.linspace(-1, 1, n)
    y = np.sinh((x - x0) * p)
    y = (y - min(y)) / (max(y) - min(y)) * (max(x) - min(x)) + min(x)
    return y
