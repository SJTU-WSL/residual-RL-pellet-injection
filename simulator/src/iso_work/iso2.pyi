
import numpy as np
from ..utils import Grid2DPolarMeta, Sp2D

def find_iso2(RZ_npsi_func: Sp2D, gridp: Grid2DPolarMeta,
              bdry_r: np.ndarray,
              axis_R: float, axis_Z: float,
              rtol: float=5e-4, ftol: float=5e-6,
              max_iter: int=30, verbose: bool=False) -> np.ndarray:
    """
    :return: iso2_r
    """
