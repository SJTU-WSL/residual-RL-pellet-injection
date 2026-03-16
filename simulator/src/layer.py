import math
import warnings

import numpy as np
from enum import Enum


_solid_density = {'D': 0.2, 'T': 0.318, 'Li': 0.534, 'Ne': 1.44, 'C': 3.3}  # 固体密度
_atomic_weight = {'D': 2.014, 'T': 3.016, 'Li': 6.9, 'Ne': 20.183, 'C': 12.011}  # 原子量
_atomic_Z = {'D': 1, 'T': 1, 'Li': 3, 'Ne': 10, 'C': 6}  # 有效核电荷


class Model(Enum):
    DT = ['D', 'T']
    Li = ['Li']
    NeD = ['Ne', 'D']
    C = ['C']


_default_ratio = {
    Model.DT: {"D": 1, "T": 0.0},
    Model.Li: {"Li": 1},
    Model.NeD: {"Ne": 0, "D": 1},
    Model.C: {"C": 1}
}


def getDensity(comp, ratio):
    aw = np.asarray([_atomic_weight[e] for e in comp if e], dtype=np.float64)
    awbyrho = np.asarray([_atomic_weight[e] / _solid_density[e] for e in comp if e], dtype=np.float64)
    nr = np.asarray([e for e in ratio if e], dtype=np.float64) / np.sum(ratio)
    return np.sum(np.dot(aw, nr)) / np.sum(np.dot(awbyrho, nr))


def getComponentNum(model: Model) -> int:
    return len(model.value)


def layerVolume(r_low, r_high):
    return 4 * np.pi / 3 * (r_high ** 3 - r_low ** 3)


class Layer:
    def __init__(self, layer_idx: int, model: Model, r_low, r_high,
                 component_ratio: dict = None,
                 min_step=200):
        """
        :param layer_idx:
        :param model:
        :param r_low: 壳层内壁半径(cm)
        :param r_high: 壳层外壁半径(cm)
        :param component_ratio:
        :param min_step:
        """
        self.idx = layer_idx
        self.model = model
        self.r_low = r_low
        self.r_high = r_high
        self.thickness = r_high - r_low  # cm
        self.volume = self.getVolumeByThickness(self.thickness)  # cm³
        self.min_step = min_step  # 最小模拟次数

        """组分比例"""
        self.ratio: dict[str, float] = dict()
        if not isinstance(component_ratio, dict):
            raise TypeError("Layer ratio: ratio must be a dictionary")
        # 如果没有输入组分比例，取默认
        if component_ratio is None:
            component_ratio = _default_ratio[self.model]
        # 读取组分比例，并判断合法性
        for component, value in component_ratio.items():
            try:
                if type(value) not in [int, float, np.ndarray]:
                    raise TypeError("Layer ratio: value type must be int, float, np.ndarray")
                value = float(value)
                if value < 0:
                    raise ValueError("Layer ratio: value cannot be negative")
                self.ratio[component] = value
            except KeyError:
                raise KeyError("Layer ratio: Missing component")
        # 对比例作归一化
        ratio_sum = sum(self.ratio.values())
        for component, value in self.ratio.items():
            self.ratio[component] = value / ratio_sum

        """物理量计算"""
        # 计算平均密度（注意固体的粒子数密度是固定的）
        ratio_np = np.asarray([values for values in self.ratio.values()])  # r
        weight_np = np.asarray([_atomic_weight[component] for component in self.model.value])  # w
        density_np = np.asarray([_solid_density[component] for component in self.model.value])  # ρ
        self.density = np.sum(ratio_np * weight_np) / np.sum(ratio_np * weight_np / density_np)  # 平均密度
        self.particle_num = self.volume / np.sum(weight_np * ratio_np / density_np) * 6.02214076e23  # 粒子数总数
        self.mean_atomic_weight = np.sum(weight_np * ratio_np)  # 平均原子量

        """弹丸消融函数"""
        # 为防止混乱的调用，外部只能通过getDrPerDt访问该层对应的消融模型
        self._function_map = {
            Model.DT: self._parks_DT,
            Model.Li: self._kuteev_Li,
            Model.NeD: self._parks_NeD,
            Model.C: self._parks_C
        }
        self.getDrPerDt = self._function_map[self.model]

        """根据模型而定的一些额外参数"""
        if self.model is Model.DT:
            ratio = self.ratio['D']
            self.weight_ratio = (1 - ratio) * _atomic_weight['T'] / _atomic_weight['D'] + ratio

        elif self.model is Model.Li:
            pass
        elif self.model is Model.NeD:
            pass
        elif self.model is Model.C:
            pass

    def __str__(self):
        return self.model.name

    def __repr__(self):
        return self.model.name

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def getVolumeByThickness(self, thickness):
        if thickness > self.thickness:
            warnings.warn("Thickness is larger than layer thickness")
        r_high = self.r_low + thickness
        return layerVolume(self.r_low, r_high)

    def _parks_DT(self, Bt, Bt_exp, Te, ne, r):
        """
        :param ratio: ratio of D
        :param r: pellet thickness? (cm)
        """
        if r < 0:
            return 0.0

        drpdt = (-(8.358 * self.weight_ratio ** (2 / 3) * (Bt / 2.0) ** Bt_exp) /
                 self.density * Te ** (5 / 3) * ne ** (1 / 3) / r ** (2 / 3))
        drpdt *= 1e-3  # cm/s to cm/ms
        return drpdt

    def _kuteev_Li(self, Bt, Bt_exp, Te, ne, r):
        if r < 0:
            return 0.0
        coeff = 1.04e15
        rhomass = 0.534e6
        amup_pl = 6.9
        z_navogadro = 6.022 * 10 ** 23
        dena = rhomass * z_navogadro / amup_pl
        # Compute the ablation rate, in atoms/s
        dndt = coeff * (Te * 10 ** 3) ** 1.64 * (ne * 10 ** 14) ** (1 / 3) * r ** (4 / 3) * amup_pl ** (-1 / 3)
        drpdt = -dndt / (dena * 4 * math.pi * (r * 0.01) ** 2)
        drpdt *= 0.1  # convert m/s to cm/ms
        return drpdt

    def _parks_NeD(self, Bt, Bt_exp, Te, ne, r):
        if r < 0:
            return 0.0
        D_weight = _atomic_weight['D']
        Ne_weight = _atomic_weight['Ne']
        # Wratio = (1 - ratio) * Ne_weight / D_weight + ratio
        ratio = self.ratio['D']
        rho_mean = getDensity(['D', 'Ne'], [ratio, 1 - ratio])
        X = ratio / (2 - ratio)
        AoX = 27.0 + np.tan(1.48 * X)

        c0 = AoX / (4 * np.pi) * (2.0 / Bt) ** Bt_exp

        drpdt = -c0 / rho_mean * Te ** (5 / 3) * ne ** (1 / 3) / r ** (2 / 3)
        drpdt *= 1e-3  # cm/s to cm/ms
        return drpdt

    def _parks_C(self, Bt, Bt_exp, Te, ne, r):
        if r < 0:
            return 0.0
        # Parks stated: multiplicative factor is ablation constant G_carbon relative to G_boron GB for pure Boron based on
        # mathmatica code "Z scaling interpolation"
        # variables appear according to their appearance in the formula of Gpr
        Te = Te * 1e3
        ne = ne * 1e14
        C0 = 8.146777e-9
        WC = _atomic_weight['C']
        gamma = 5.0 / 3.0

        ZstarPlus1C = 2.86
        Albedo = 23.920538030089528 * np.log(1 + 0.20137080524063228 * ZstarPlus1C)
        flelectro = np.exp(-1.936)
        fL = (1.0 - Albedo / 100) * flelectro

        IstC = 60
        Ttmp = Te if Te > 30 else 30
        loglamCSlow = np.log(2.0 * Ttmp / IstC * np.sqrt(np.e * 2.0))
        BLamdaq = 1 / (_atomic_Z['C'] * loglamCSlow) * (4 / (2.5 + 2.2 * np.sqrt(ZstarPlus1C)))
        Gpr = (
                C0
                * np.power(WC, 2.0 / 3.0)
                * np.power(gamma - 1.0, 1.0 / 3.0)
                * np.power(fL * ne, 1.0 / 3.0)
                * np.power(r, 4.0 / 3.0)
                * np.power(Te, 11.0 / 6.0)
                * np.power(BLamdaq, 2.0 / 3.0)
        )

        xiexp = 0.601
        lamdaa = 0.0933979540623963
        lamdab = -0.7127242270013098
        lamdac = -0.2437544205933372
        lamdad = -0.8534855445478313
        av = 10.420403555938629 * np.power(Ttmp / 2000.0, lamdaa)
        bv = 0.6879779829877795 * np.power(Ttmp / 2000.0, lamdab)
        cv = 1.5870910225610804 * np.power(Ttmp / 2000.0, lamdac)
        dv = 2.9695640286641840 * np.power(Ttmp / 2000.0, lamdad)
        fugCG = 0.777686
        CG = (
                fugCG
                * av
                * np.log(1 + bv * np.power(ne / 1e14, 2.0 / 3.0) * np.power(r, 2.0 / 3.0))
                / np.log(cv + dv * np.power(ne / 1e14, 2.0 / 3.0) * np.power(r, 2.0 / 3.0))
        )
        G = xiexp * CG * Gpr

        drpdt = -G / (4.0 * np.pi * getDensity(['C'], [1]) * r ** 2)
        drpdt *= 1e-3  # cm/s to cm/ms
        return drpdt
