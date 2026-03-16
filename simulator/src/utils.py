
import numpy as np
from numpy import ndarray


# Grid2DMeta
class Grid2DMeta:
    def __init__(self, R_left: float, R_right: float,
                 Z_left: float, Z_right: float,
                 nR: int, nZ: int, isogrid=False):
        self.R_left = min(R_left, R_right)  # left
        self.R_right = max(R_left, R_right)  # right
        self.Z_left = min(Z_left, Z_right)  # left
        self.Z_right = max(Z_left, Z_right)  # right
        self.nR = nR
        self.nZ = nZ
        self.dR = (self.R_right - self.R_left) / (self.nR - 1)
        self.dZ = (self.Z_right - self.Z_left) / (self.nZ - 1)
        self.invdR = 1 / self.dR
        self.invdZ = 1 / self.dZ
        self.R = np.linspace(self.R_left, self.R_right, self.nR, dtype=float)
        self.Z = np.linspace(self.Z_left, self.Z_right, self.nZ, dtype=float)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z, indexing='ij')

    def __repr__(self):
        return '{0:} x {1:}'.format(self.nR, self.nZ)

    @property
    def shape(self):
        return self.nR, self.nZ


# Grid2DMeta
class Grid2DPolarMeta:
    def __init__(self, r_left: float, r_right: float,
                 a_left: float, a_right: float,
                 nr: int, na: int):
        self.r_left = min(r_left, r_right)  # left
        self.r_right = max(r_left, r_right)  # right
        self.a_left = min(a_left, a_right)  # left
        self.a_right = max(a_left, a_right)  # right
        self.nr = nr
        self.na = na
        self.dr = (self.r_right - self.r_left) / (self.nr - 1)
        self.da = (self.a_right - self.a_left) / (self.na - 1)
        self.invdr = 1 / self.dr
        self.invda = 1 / self.da
        self.r = np.linspace(self.r_left, self.r_right, self.nr, dtype=float)
        self.a = np.linspace(self.a_left, self.a_right, self.na, dtype=float)
        self.a_half = (self.a[1:] + self.a[:-1]) / 2
        self.rr, self.aa = np.meshgrid(self.r, self.a, indexing='ij')

        self.rr_rh = (self.aa[1:, :] + self.aa[:-1, :]) / 2
        self.rr_ah = (self.rr[:, 1:] + self.rr[:, :-1]) / 2
        self.rr_rh_ah = (self.rr_ah[1:, :] + self.rr_ah[:-1, :]) / 2
        self.aa_ah = (self.aa[:, 1:] + self.aa[:, :-1]) / 2

    def __repr__(self):
        return '{0:} x {1:}'.format(self.nr, self.na)


# ord0
def cubicWeight_n1(x):
    x2 = x * x  # -1 <= x <= 0
    return -1.5 * x * x2 - 2.5 * x2 + 1.0

def cubicWeight_1(x):
    x2 = x * x  # 0 <= x <= 1
    return 1.5 * x * x2 - 2.5 * x2 + 1.0

def cubicWeight_n2(x):
    x2 = x * x  # -2 <= x <= -1
    return 0.5 * x * x2 + 2.5 * x2 + 4.0 * x + 2.0

def cubicWeight_2(x):
    x2 = x * x  # 1 <= x <= 2
    return -0.5 * x * x2 + 2.5 * x2 - 4.0 * x + 2.0

# 1阶导数可用，但拟合出的1阶导的导数连续性已经有轻微破坏，到ord2时，尽管插值可能与准确值相近，
# 但拟合出的2阶导的导数连续性被严重破坏（可使用normal2d验证），在某些情况下，插值本身也不准确。
# 最好的方法是直接对原矩阵进行差分，然后一直使用0阶拟合，这样n阶导数插值可以有更多格点参与进来，
# 并且具备更好的准确度

# ord1
def cubicWeight_n1_ord1(x):  # -1 <= x <= 0
    return -4.5 * x * x - 5.0 * x

def cubicWeight_1_ord1(x):  # 0 <= x <= 1
    return 4.5 * x * x - 5.0 * x

def cubicWeight_n2_ord1(x):  # -2 <= x <= -1
    return 1.5 * x * x + 5.0 * x + 4.0

def cubicWeight_2_ord1(x):  # 1 <= x <= 2
    return -1.5 * x * x + 5.0 * x - 4.0

# ord2
def cubicWeight_n1_ord2(x):  # -1 <= x <= 0
    return -9.0 * x - 5.0

def cubicWeight_1_ord2(x):  # 0 <= x <= 1
    return 9.0 * x - 5.0

def cubicWeight_n2_ord2(x):  # -2 <= x <= -1
    return 3.0 * x + 5.0

def cubicWeight_2_ord2(x):  # 1 <= x <= 2
    return -3.0 * x + 5.0


def sp2D(src_mat: ndarray,
         src_R_left: float, src_R_right: float, nR: int,
         src_Z_left: float, src_Z_right: float, nZ: int,
         dst_RR: ndarray, dst_ZZ: ndarray):
    dR = (src_R_right - src_R_left) / (nR - 1)
    dZ = (src_Z_right - src_Z_left) / (nZ - 1)
    dst_R_idx_double = (dst_RR - src_R_left) / dR
    dst_Z_idx_double = (dst_ZZ - src_Z_left) / dZ
    dst_R_idx = np.floor(dst_R_idx_double).astype(int)
    dst_Z_idx = np.floor(dst_Z_idx_double).astype(int)
    R_distance = dst_R_idx_double - dst_R_idx
    Z_distance = dst_Z_idx_double - dst_Z_idx
    R_weight = [
        cubicWeight_2(R_distance + 1),
        cubicWeight_1(R_distance),
        cubicWeight_n1(R_distance - 1),
        cubicWeight_n2(R_distance - 2)
    ]
    R_weight_ord1 = [
        cubicWeight_2_ord1(R_distance + 1),
        cubicWeight_1_ord1(R_distance),
        cubicWeight_n1_ord1(R_distance - 1),
        cubicWeight_n2_ord1(R_distance - 2)
    ]
    R_weight_ord2 = [
        cubicWeight_2_ord2(R_distance + 1),
        cubicWeight_1_ord2(R_distance),
        cubicWeight_n1_ord2(R_distance - 1),
        cubicWeight_n2_ord2(R_distance - 2)
    ]
    Z_weight = [
        cubicWeight_2(Z_distance + 1),
        cubicWeight_1(Z_distance),
        cubicWeight_n1(Z_distance - 1),
        cubicWeight_n2(Z_distance - 2)
    ]
    Z_weight_ord1 = [
        cubicWeight_2_ord1(Z_distance + 1),
        cubicWeight_1_ord1(Z_distance),
        cubicWeight_n1_ord1(Z_distance - 1),
        cubicWeight_n2_ord1(Z_distance - 2)
    ]
    Z_weight_ord2 = [
        cubicWeight_2_ord2(Z_distance + 1),
        cubicWeight_1_ord2(Z_distance),
        cubicWeight_n1_ord2(Z_distance - 1),
        cubicWeight_n2_ord2(Z_distance - 2)
    ]

    result = np.zeros_like(dst_RR)
    result_gR = np.zeros_like(dst_RR)
    result_gZ = np.zeros_like(dst_RR)
    result_ggR = np.zeros_like(dst_RR)
    result_ggZ = np.zeros_like(dst_RR)
    max_R_idx = nR - 1
    max_Z_idx = nZ - 1
    for m in range(-1, 3):
        R_idx = np.minimum(np.maximum(dst_R_idx + m, 0), max_R_idx)
        for n in range(-1, 3):
            Z_idx = np.minimum(np.maximum(dst_Z_idx+n, 0), max_Z_idx)
            result += src_mat[R_idx, Z_idx] * R_weight[m+1] * Z_weight[n+1]
            result_gR += (src_mat[R_idx, Z_idx] * R_weight_ord1[m+1] * Z_weight[n+1])
            result_gZ += (src_mat[R_idx, Z_idx] * R_weight[m+1] * Z_weight_ord1[n+1])
            result_ggR += (src_mat[R_idx, Z_idx] * R_weight_ord2[m+1] * Z_weight[n+1])
            result_ggZ += (src_mat[R_idx, Z_idx] * R_weight[m+1] * Z_weight_ord2[n+1])
    result_gR = result_gR / dR
    result_gZ = result_gZ / dZ
    result_ggR = result_ggR / dR**2
    result_ggZ = result_ggZ / dZ**2
    return result, result_gR, result_gZ, result_ggR, result_ggZ


class Sp2D:
    def __init__(self, src_R: ndarray, src_Z: ndarray, src_mat: ndarray):
        self.src_R_left = src_R[0]
        self.src_R_right = src_R[-1]
        self.src_Z_left = src_Z[0]
        self.src_Z_right = src_Z[-1]
        self.nR = len(src_R)
        self.nZ = len(src_Z)
        self.dR = (self.src_R_right - self.src_R_left) / (self.nR - 1)
        self.dZ = (self.src_Z_right - self.src_Z_left) / (self.nZ - 1)
        self.src_mat = src_mat
        self.src_mat_gR, self.src_mat_gZ = np.gradient(src_mat, self.dR, self.dZ, edge_order=2)
        self.src_mat_ggR = np.gradient(self.src_mat_gR, self.dR, axis=0, edge_order=2)
        self.src_mat_ggZ = np.gradient(self.src_mat_gZ, self.dZ, axis=1, edge_order=2)

    def __call__(self, dst_R, dst_Z):
        dst_R_idx_double = (dst_R - self.src_R_left) / self.dR
        dst_Z_idx_double = (dst_Z - self.src_Z_left) / self.dZ
        dst_R_idx = np.floor(dst_R_idx_double).astype(int)
        dst_Z_idx = np.floor(dst_Z_idx_double).astype(int)
        R_distance = dst_R_idx_double - dst_R_idx
        Z_distance = dst_Z_idx_double - dst_Z_idx
        R_weight = [
            cubicWeight_2(R_distance + 1),
            cubicWeight_1(R_distance),
            cubicWeight_n1(R_distance - 1),
            cubicWeight_n2(R_distance - 2)
        ]
        Z_weight = [
            cubicWeight_2(Z_distance + 1),
            cubicWeight_1(Z_distance),
            cubicWeight_n1(Z_distance - 1),
            cubicWeight_n2(Z_distance - 2)
        ]

        result = np.zeros_like(dst_R)
        max_R_idx = self.nR - 1
        max_Z_idx = self.nZ - 1
        for m in range(-1, 3):
            R_idx = np.minimum(np.maximum(dst_R_idx + m, 0), max_R_idx)
            for n in range(-1, 3):
                Z_idx = np.minimum(np.maximum(dst_Z_idx + n, 0), max_Z_idx)
                func_val = R_weight[m + 1] * Z_weight[n + 1]
                result += self.src_mat[R_idx, Z_idx] * func_val
        return result

    def level0(self, dst_R, dst_Z):
        return self.__call__(dst_R, dst_Z)

    def level1(self, dst_R, dst_Z):
        dst_R_idx_double = (dst_R - self.src_R_left) / self.dR
        dst_Z_idx_double = (dst_Z - self.src_Z_left) / self.dZ
        dst_R_idx = np.floor(dst_R_idx_double).astype(int)
        dst_Z_idx = np.floor(dst_Z_idx_double).astype(int)
        R_distance = dst_R_idx_double - dst_R_idx
        Z_distance = dst_Z_idx_double - dst_Z_idx
        R_weight = [
            cubicWeight_2(R_distance + 1),
            cubicWeight_1(R_distance),
            cubicWeight_n1(R_distance - 1),
            cubicWeight_n2(R_distance - 2)
        ]
        Z_weight = [
            cubicWeight_2(Z_distance + 1),
            cubicWeight_1(Z_distance),
            cubicWeight_n1(Z_distance - 1),
            cubicWeight_n2(Z_distance - 2)
        ]

        result = np.zeros_like(dst_R)
        result_gR = np.zeros_like(dst_R)
        result_gZ = np.zeros_like(dst_R)
        max_R_idx = self.nR - 1
        max_Z_idx = self.nZ - 1
        for m in range(-1, 3):
            R_idx = np.minimum(np.maximum(dst_R_idx + m, 0), max_R_idx)
            for n in range(-1, 3):
                Z_idx = np.minimum(np.maximum(dst_Z_idx + n, 0), max_Z_idx)
                func_val = R_weight[m + 1] * Z_weight[n + 1]
                result += self.src_mat[R_idx, Z_idx] * func_val
                result_gR += self.src_mat_gR[R_idx, Z_idx] * func_val
                result_gZ += self.src_mat_gZ[R_idx, Z_idx] * func_val
        return result, result_gR, result_gZ

    def level2(self, dst_R, dst_Z):
        dst_R_idx_double = (dst_R - self.src_R_left) / self.dR
        dst_Z_idx_double = (dst_Z - self.src_Z_left) / self.dZ
        dst_R_idx = np.floor(dst_R_idx_double).astype(int)
        dst_Z_idx = np.floor(dst_Z_idx_double).astype(int)
        R_distance = dst_R_idx_double - dst_R_idx
        Z_distance = dst_Z_idx_double - dst_Z_idx
        R_weight = [
            cubicWeight_2(R_distance + 1),
            cubicWeight_1(R_distance),
            cubicWeight_n1(R_distance - 1),
            cubicWeight_n2(R_distance - 2)
        ]
        Z_weight = [
            cubicWeight_2(Z_distance + 1),
            cubicWeight_1(Z_distance),
            cubicWeight_n1(Z_distance - 1),
            cubicWeight_n2(Z_distance - 2)
        ]

        result = np.zeros_like(dst_R)
        result_gR = np.zeros_like(dst_R)
        result_gZ = np.zeros_like(dst_R)
        result_ggR = np.zeros_like(dst_R)
        result_ggZ = np.zeros_like(dst_R)
        max_R_idx = self.nR - 1
        max_Z_idx = self.nZ - 1
        for m in range(-1, 3):
            R_idx = np.minimum(np.maximum(dst_R_idx + m, 0), max_R_idx)
            for n in range(-1, 3):
                Z_idx = np.minimum(np.maximum(dst_Z_idx + n, 0), max_Z_idx)
                func_val = R_weight[m + 1] * Z_weight[n + 1]
                result += self.src_mat[R_idx, Z_idx] * func_val
                result_gR += self.src_mat_gR[R_idx, Z_idx] * func_val
                result_gZ += self.src_mat_gZ[R_idx, Z_idx] * func_val
                result_ggR += self.src_mat_ggR[R_idx, Z_idx] * func_val
                result_ggZ += self.src_mat_ggZ[R_idx, Z_idx] * func_val
        return result, result_gR, result_gZ, result_ggR, result_ggZ


