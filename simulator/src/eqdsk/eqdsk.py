
# most copied from omfit_eqdsk
import os
import re
import warnings

import numpy as np
import time
import copy
import scipy
from scipy import interpolate, integrate
from scipy.integrate import cumulative_trapezoid  # 核心修复 1
from scipy.interpolate import RectBivariateSpline, interp1d  # 核心修复 2

from .utils_base import *
from .utils_math import *
from .fluxSurface import FluxSurfaces

class XPointSearchFail(ValueError):
    """x_point_search failed"""

def x_point_quick_search(rgrid, zgrid, psigrid, psi_boundary=None, psi_boundary_weight=1.0, zsign=0):
    """快速定位 X 点大致坐标"""
    rr, zz = np.meshgrid(rgrid, zgrid)
    [dpsidz, dpsidr] = np.gradient(psigrid, zgrid[1] - zgrid[0], rgrid[1] - rgrid[0])
    br = dpsidz / rr
    bz = -dpsidr / rr
    bpol2 = br**2 + bz**2
    if psi_boundary is None:
        dpsi2 = psigrid * 0
    else:
        dpsi2 = (psigrid - psi_boundary) ** 2
    gridspace2 = (zgrid[1] - zgrid[0]) * (rgrid[1] - rgrid[0])
    dpsi2norm = abs(dpsi2 / gridspace2 / rr**2)
    deviation = bpol2 + psi_boundary_weight * dpsi2norm
    if zsign == 1:
        deviation[zz <= 0] = np.nanmax(deviation) * 10
    elif zsign == -1:
        deviation[zz >= 0] = np.nanmax(deviation) * 10
    idx = np.nanargmin(deviation)
    rx = rr.flatten()[idx]
    zx = zz.flatten()[idx]
    return np.array([rx, zx])

def x_point_search(rgrid, zgrid, psigrid, r_center=None, z_center=None, dr=None, dz=None, zoom=5, hardfail=False, **kw):
    """使用 RectBivariateSpline 高精度寻找 X 点 (替代已移除的 interp2d)"""
    dr = dr or (rgrid[1] - rgrid[0]) * 5
    dz = dz or (zgrid[1] - zgrid[0]) * 5
    
    if (r_center is None) or (z_center is None):
        r_center, z_center = x_point_quick_search(rgrid, zgrid, psigrid, **kw)
        
    selr = (rgrid >= (r_center - dr)) & (rgrid <= (r_center + dr))
    selz = (zgrid >= (z_center - dz)) & (zgrid <= (z_center + dz))
    
    if sum(selr) < 2 or sum(selz) < 2:
        if hardfail: raise XPointSearchFail("Grid points insufficient in search region")
        return np.array([np.nan, np.nan])

    # 局部缩放以提升精度
    r_z = scipy.ndimage.zoom(rgrid[selr], zoom)
    z_z = scipy.ndimage.zoom(zgrid[selz], zoom)
    psi_z = scipy.ndimage.zoom(psigrid[selz, :][:, selr], zoom)
    
    rr_z, _ = np.meshgrid(r_z, z_z)
    [dpsidz, dpsidr] = np.gradient(psi_z, z_z[1] - z_z[0], r_z[1] - r_z[0])
    br_z = dpsidz / rr_z
    bz_z = -dpsidr / rr_z
    
    try:
        segments = contourPaths(r_z, z_z, br_z, [0], remove_boundary_points=True)[0]
    except Exception:
        return np.array([np.nan, np.nan])

    if len(segments):
        dist2 = [np.min((seg.vertices[:, 0] - r_center)**2 + (seg.vertices[:, 1] - z_center)**2) for seg in segments]
        verts = segments[np.argmin(dist2)].vertices
        
        # 核心替换: interp2d -> RectBivariateSpline (注意转置以匹配 R,Z)
        bz_spline = RectBivariateSpline(r_z, z_z, bz_z.T)
        bzpath = bz_spline(verts[:, 0], verts[:, 1], grid=False)
        
        try:
            rx = float(interp1d(bzpath, verts[:, 0], bounds_error=False, fill_value=np.nan)(0))
            zx = float(interp1d(bzpath, verts[:, 1], bounds_error=False, fill_value=np.nan)(0))
        except Exception:
            rx = zx = np.nan
    else:
        rx = zx = np.nan
    return np.array([rx, zx])

class Geqdsk(dict):
    transform_signals = {
        'SIMAG': 'PSI',
        'SIBRY': 'PSI',
        'BCENTR': 'BT',
        'CURRENT': 'IP',
        'FPOL': 'BT',
        'FFPRIM': 'dPSI',
        'PPRIME': 'dPSI',
        'PSIRZ': 'PSI',
        'QPSI': 'Q',
    }

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.load()

    def load(self, raw=False, add_aux=True):
        """
        Method used to read g-files
        :param raw: bool
            load gEQDSK exactly as it's on file, regardless of COCOS
        :param add_aux: bool
            Add AuxQuantities and fluxSurfaces when using `raw` mode. When not raw, these will be loaded regardless.
        """

        if self.filename is None or not os.stat(self.filename).st_size:
            return

        # todo should be rewritten using FortranRecordReader
        # based on w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
        def splitter(inv, step=16):
            value = []
            for k in range(len(inv) // step):
                value.append(inv[step * k: step * (k + 1)])
            return value

        def merge(inv):
            if not len(inv):
                return ''
            if len(inv[0]) > 80:
                # SOLPS gEQDSK files add spaces between numbers
                # and positive numbers are preceeded by a +
                return (''.join(inv)).replace(' ', '')
            else:
                return ''.join(inv)

        self.clear()

        # clean lines from the carriage returns
        with open(self.filename, 'r') as f:
            EQDSK = f.read().splitlines()

        # first line is description and sizes
        self['CASE'] = np.array(splitter(EQDSK[0][0:48], 8))
        try:
            tmp = list([_f for _f in EQDSK[0][48:].split(' ') if _f])
            [IDUM, self['NW'], self['NH']] = list(map(int, tmp[:3]))
        except ValueError:  # Can happen if no space between numbers, such as 10231023
            IDUM = int(EQDSK[0][48:52])
            self['NW'] = int(EQDSK[0][52:56])
            self['NH'] = int(EQDSK[0][56:60])
            tmp = []
            # printd('IDUM, NW, NH', IDUM, self['NW'], self['NH'], topic='OMFITgeqdsk.load')
        if len(tmp) > 3:
            self['EXTRA_HEADER'] = EQDSK[0][49 + len(
                re.findall('%d +%d +%d ' % (IDUM, self['NW'], self['NH']), EQDSK[0][49:])[0]) + 2:]
        offset = 1

        # now, the next 20 numbers (5 per row)

        # fmt: off
        [self['RDIM'], self['ZDIM'], self['RCENTR'], self['RLEFT'], self['ZMID'],
         self['RMAXIS'], self['ZMAXIS'], self['SIMAG'], self['SIBRY'], self['BCENTR'],
         self['CURRENT'], self['SIMAG'], XDUM, self['RMAXIS'], XDUM,
         self['ZMAXIS'], XDUM, self['SIBRY'], XDUM, XDUM] = list(map(eval, splitter(merge(EQDSK[offset:offset + 4]))))
        # fmt: on
        offset = offset + 4

        # now I have to read NW elements
        nlNW = int(np.ceil(self['NW'] / 5.0))
        self['FPOL'] = np.array(list(map(float, splitter(merge(EQDSK[offset: offset + nlNW])))))
        offset = offset + nlNW
        self['PRES'] = np.array(list(map(float, splitter(merge(EQDSK[offset: offset + nlNW])))))
        offset = offset + nlNW
        self['FFPRIM'] = np.array(list(map(float, splitter(merge(EQDSK[offset: offset + nlNW])))))
        offset = offset + nlNW
        self['PPRIME'] = np.array(list(map(float, splitter(merge(EQDSK[offset: offset + nlNW])))))
        offset = offset + nlNW
        try:
            # official gEQDSK file format saves PSIRZ as a single flat array of size rowsXcols
            nlNWNH = int(np.ceil(self['NW'] * self['NH'] / 5.0))
            self['PSIRZ'] = np.reshape(
                np.fromiter(splitter(merge(EQDSK[offset: offset + nlNWNH])), dtype=np.float64)[
                : self['NH'] * self['NW']],
                (self['NH'], self['NW']),
            )
            offset = offset + nlNWNH
        except ValueError:
            # sometimes gEQDSK files save row by row of the PSIRZ grid (eg. FIESTA code)
            nlNWNH = self['NH'] * nlNW
            self['PSIRZ'] = np.reshape(
                np.fromiter(splitter(merge(EQDSK[offset: offset + nlNWNH])), dtype=np.float64)[
                : self['NH'] * self['NW']],
                (self['NH'], self['NW']),
            )
            offset = offset + nlNWNH
        self['QPSI'] = np.array(list(map(float, splitter(merge(EQDSK[offset: offset + nlNW])))))
        offset = offset + nlNW

        # now vacuum vessel and limiters
        if len(EQDSK) > (offset + 1):
            self['NBBBS'], self['LIMITR'] = list(
                map(int, [_f for _f in EQDSK[offset: offset + 1][0].split(' ') if _f][:2]))
            offset += 1

            nlNBBBS = int(np.ceil(self['NBBBS'] * 2 / 5.0))
            self['RBBBS'] = np.array(list(map(float, splitter(merge(EQDSK[offset: offset + nlNBBBS]))))[0::2])[
                            : self['NBBBS']]
            self['ZBBBS'] = np.array(list(map(float, splitter(merge(EQDSK[offset: offset + nlNBBBS]))))[1::2])[
                            : self['NBBBS']]
            offset = offset + max(nlNBBBS, 1)

            try:
                # this try/except is to handle some gEQDSK files written by older versions of ONETWO
                nlLIMITR = int(np.ceil(self['LIMITR'] * 2 / 5.0))
                self['RLIM'] = np.array(list(map(float, splitter(merge(EQDSK[offset: offset + nlLIMITR]))))[0::2])[
                               : self['LIMITR']]
                self['ZLIM'] = np.array(list(map(float, splitter(merge(EQDSK[offset: offset + nlLIMITR]))))[1::2])[
                               : self['LIMITR']]
                offset = offset + nlLIMITR
            except ValueError:
                # if it fails make the limiter as a rectangle around the plasma boundary that does not exceed the computational domain
                self['LIMITR'] = 5
                dd = self['RDIM'] / 10.0
                R = np.linspace(0, self['RDIM'], 2) + self['RLEFT']
                Z = np.linspace(0, self['ZDIM'], 2) - self['ZDIM'] / 2.0 + self['ZMID']
                self['RLIM'] = np.array(
                    [
                        max([R[0], np.min(self['RBBBS']) - dd]),
                        min([R[1], np.max(self['RBBBS']) + dd]),
                        min([R[1], np.max(self['RBBBS']) + dd]),
                        max([R[0], np.min(self['RBBBS']) - dd]),
                        max([R[0], np.min(self['RBBBS']) - dd]),
                    ]
                )
                self['ZLIM'] = np.array(
                    [
                        max([Z[0], np.min(self['ZBBBS']) - dd]),
                        max([Z[0], np.min(self['ZBBBS']) - dd]),
                        min([Z[1], np.max(self['ZBBBS']) + dd]),
                        min([Z[1], np.max(self['ZBBBS']) + dd]),
                        max([Z[0], np.min(self['ZBBBS']) - dd]),
                    ]
                )
        else:
            self['NBBBS'] = 0
            self['LIMITR'] = 0
            self['RBBBS'] = []
            self['ZBBBS'] = []
            self['RLIM'] = []
            self['ZLIM'] = []

        try:
            [self['KVTOR'], self['RVTOR'], self['NMASS']] = list(
                map(float, [_f for _f in EQDSK[offset: offset + 1][0].split(' ') if _f]))
            offset = offset + 1

            if self['KVTOR'] > 0:
                self['PRESSW'] = np.array(list(map(float, splitter(merge(EQDSK[offset: offset + nlNW])))))
                offset = offset + nlNW
                self['PWPRIM'] = np.array(list(map(float, splitter(merge(EQDSK[offset: offset + nlNW])))))
                offset = offset + nlNW

            if self['NMASS'] > 0:
                self['DMION'] = np.array(list(map(float, splitter(merge(EQDSK[offset: offset + nlNW])))))
                offset = offset + nlNW

            self['RHOVN'] = np.array(list(map(float, splitter(merge(EQDSK[offset: offset + nlNW])))))
            offset = offset + nlNW

            self['KEECUR'] = int(EQDSK[offset])
            offset = offset + 1

            if self['KEECUR'] > 0:
                self['EPOTEN'] = np.array(splitter(merge(EQDSK[offset: offset + nlNW])), dtype=float)
                offset = offset + nlNW

            # This will only work when IPLCOUT==2, which is not available in older versions of EFIT
            self['PCURRT'] = np.reshape(
                np.fromiter(splitter(merge(EQDSK[offset: offset + nlNWNH])), dtype=np.float64)[
                : self['NH'] * self['NW']],
                (self['NH'], self['NW']),
            )
            offset = offset + nlNWNH
            self['CJOR'] = np.array(splitter(merge(EQDSK[offset: offset + nlNW])), dtype=float)
            offset = offset + nlNW
            self['R1SURF'] = np.array(splitter(merge(EQDSK[offset: offset + nlNW])), dtype=float)
            offset = offset + nlNW
            self['R2SURF'] = np.array(splitter(merge(EQDSK[offset: offset + nlNW])), dtype=float)
            offset = offset + nlNW
            self['VOLP'] = np.array(splitter(merge(EQDSK[offset: offset + nlNW])), dtype=float)
            offset = offset + nlNW
            self['BPOLSS'] = np.array(splitter(merge(EQDSK[offset: offset + nlNW])), dtype=float)
            offset = offset + nlNW
        except Exception:
            pass

        # add RHOVN if missing
        if 'RHOVN' not in self or not len(self['RHOVN']) or not np.sum(self['RHOVN']):
            self.add_rhovn()

        # fix some gEQDSK files that do not fill PRES info (eg. EAST)
        if not np.sum(self['PRES']):
            pres = cumulative_trapezoid(self['PPRIME'], np.linspace(self['SIMAG'], self['SIBRY'], len(self['PPRIME'])),
                                      initial=0)
            self['PRES'] = pres - pres[-1]

        # parse auxiliary namelist
        # self.addAuxNamelist()

        if raw and add_aux:
            # add AuxQuantities and fluxSurfaces
            # self.addAuxQuantities()
            # self.addFluxSurfaces(**self.OMFITproperties)
            raise RuntimeWarning("Unsupport now")

        elif not raw:
            # Convert tree representation to COCOS 1
            self._cocos = self.native_cocos()
            self.cocosify(1, calcAuxQuantities=True, calcFluxSurfaces=True)

        self.add_geqdsk_documentation()

    @property
    def cocos(self):
        """
        Return COCOS of current gEQDSK as represented in memory
        """
        if self._cocos is None:
            return self.native_cocos()
        return self._cocos

    def native_cocos(self):
        """
        Returns the native COCOS that an unmodified gEQDSK would obey, defined by sign(Bt) and sign(Ip)
        In order for psi to increase from axis to edge and for q to be positive:
        All use sigma_RpZ=+1 (phi is counterclockwise) and exp_Bp=0 (psi is flux/2.*pi)
        We want
        sign(psi_edge-psi_axis) = sign(Ip)*sigma_Bp > 0  (psi always increases in gEQDSK)
        sign(q) = sign(Ip)*sign(Bt)*sigma_rhotp > 0      (q always positive in gEQDSK)
        ::
            ============================================
            Bt    Ip    sigma_Bp    sigma_rhotp    COCOS
            ============================================
            +1    +1       +1           +1           1
            +1    -1       -1           -1           3
            -1    +1       +1           -1           5
            -1    -1       -1           +1           7
        """
        try:
            return gEQDSK_COCOS_identify(self['BCENTR'], self['CURRENT'])
        except Exception as _excp:
            # printe("Assuming COCOS=1: " + repr(_excp))
            return 1

    def add_geqdsk_documentation(self):
        gdesc = self['_desc'] = dict()
        gdesc['CASE'] = 'Identification character string'
        gdesc['NW'] = 'Number of horizontal R grid points'
        gdesc['NH'] = 'Number of vertical Z grid points'
        gdesc['RDIM'] = 'Horizontal dimension in meter of computational box'
        gdesc['ZDIM'] = 'Vertical dimension in meter of computational box'
        gdesc['RCENTR'] = 'R in meter of vacuum toroidal magnetic field BCENTR'
        gdesc['RLEFT'] = 'Minimum R in meter of rectangular computational box'
        gdesc['ZMID'] = 'Z of center of computational box in meter'
        gdesc['RMAXIS'] = 'R of magnetic axis in meter'
        gdesc['ZMAXIS'] = 'Z of magnetic axis in meter'
        gdesc['SIMAG'] = 'poloidal flux at magnetic axis in Weber /rad'
        gdesc['SIBRY'] = 'poloidal flux at the plasma boundary in Weber /rad'
        gdesc['BCENTR'] = 'Vacuum toroidal magnetic field in Tesla at RCENTR'
        gdesc['CURRENT'] = 'Plasma current in Ampere'
        gdesc['FPOL'] = 'Poloidal current function in m-T, F = RBT on flux grid'
        gdesc['PRES'] = 'Plasma pressure in nt / m2 on uniform flux grid'
        gdesc['FFPRIM'] = 'FF’(ψ) in (mT)2 / (Weber /rad) on uniform flux grid'
        gdesc['PPRIME'] = 'P’(ψ) in (nt /m2) / (Weber /rad) on uniform flux grid'
        gdesc['PSIRZ'] = 'Poloidal flux in Weber / rad on the rectangular grid points'
        gdesc['QPSI'] = 'q values on uniform flux grid from axis to boundary'
        gdesc['NBBBS'] = 'Number of boundary points'
        gdesc['LIMITR'] = 'Number of limiter points'
        gdesc['RBBBS'] = 'R of boundary points in meter'
        gdesc['ZBBBS'] = 'Z of boundary points in meter'
        gdesc['RLIM'] = 'R of surrounding limiter contour in meter'
        gdesc['ZLIM'] = 'Z of surrounding limiter contour in meter'
        gdesc['KVTOR'] = ''
        gdesc['RVTOR'] = ''
        gdesc['NMASS'] = ''
        gdesc['RHOVN'] = ''
        gdesc['AuxNamelist'] = dict()
        gdesc['AuxQuantities'] = dict()
        gdesc['fluxSurfaces'] = dict()

        ### AUX NAMELIST ###

        andesc = gdesc['AuxNamelist']

        ## EFITIN ##

        andesc['efitin'] = dict()
        andesc['efitin']['scrape'] = ''
        andesc['efitin']['nextra'] = ''
        andesc['efitin']['itek'] = ''
        andesc['efitin']['ICPROF'] = ''
        andesc['efitin']['qvfit'] = ''
        andesc['efitin']['fwtbp'] = ''
        andesc['efitin']['kffcur'] = ''
        andesc['efitin']['kppcur'] = ''
        andesc['efitin']['fwtqa'] = ''
        andesc['efitin']['zelip'] = ''
        andesc['efitin']['iavem'] = ''
        andesc['efitin']['iavev'] = ''
        andesc['efitin']['n1coil'] = ''
        andesc['efitin']['nccoil'] = ''
        andesc['efitin']['nicoil'] = ''
        andesc['efitin']['iout'] = ''
        andesc['efitin']['fwtsi'] = ''
        andesc['efitin']['fwtmp2'] = ''
        andesc['efitin']['fwtcur'] = ''
        andesc['efitin']['fitdelz'] = ''
        andesc['efitin']['fwtfc'] = ''
        andesc['efitin']['fitsiref'] = ''
        andesc['efitin']['kersil'] = ''
        andesc['efitin']['ifitdelz'] = ''
        andesc['efitin']['ERROR'] = ''
        andesc['efitin']['ERRMIN'] = ''
        andesc['efitin']['MXITER'] = ''
        andesc['efitin']['fcurbd'] = ''
        andesc['efitin']['pcurbd'] = ''
        andesc['efitin']['kcalpa'] = ''
        andesc['efitin']['kcgama'] = ''
        andesc['efitin']['xalpa'] = ''
        andesc['efitin']['xgama'] = ''
        andesc['efitin']['RELAX'] = ''
        andesc['efitin']['keqdsk'] = ''
        andesc['efitin']['CALPA'] = ''
        andesc['efitin']['CGAMA'] = ''

        ## OUT1 ##

        andesc['OUT1'] = dict()
        andesc['OUT1']['ISHOT'] = ''
        andesc['OUT1']['ITIME'] = ''
        andesc['OUT1']['BETAP0'] = ''
        andesc['OUT1']['RZERO'] = ''
        andesc['OUT1']['QENP'] = ''
        andesc['OUT1']['ENP'] = ''
        andesc['OUT1']['EMP'] = ''
        andesc['OUT1']['PLASMA'] = ''
        andesc['OUT1']['EXPMP2'] = ''
        andesc['OUT1']['COILS'] = ''
        andesc['OUT1']['BTOR'] = ''
        andesc['OUT1']['RCENTR'] = ''
        andesc['OUT1']['BRSP'] = ''
        andesc['OUT1']['ICURRT'] = ''
        andesc['OUT1']['RBDRY'] = ''
        andesc['OUT1']['ZBDRY'] = ''
        andesc['OUT1']['NBDRY'] = ''
        andesc['OUT1']['FWTSI'] = ''
        andesc['OUT1']['FWTCUR'] = ''
        andesc['OUT1']['MXITER'] = ''
        andesc['OUT1']['NXITER'] = ''
        andesc['OUT1']['LIMITR'] = ''
        andesc['OUT1']['XLIM'] = ''
        andesc['OUT1']['YLIM'] = ''
        andesc['OUT1']['ERROR'] = ''
        andesc['OUT1']['ICONVR'] = ''
        andesc['OUT1']['IBUNMN'] = ''
        andesc['OUT1']['PRESSR'] = ''
        andesc['OUT1']['RPRESS'] = ''
        andesc['OUT1']['QPSI'] = ''
        andesc['OUT1']['PRESSW'] = ''
        andesc['OUT1']['PRES'] = ''
        andesc['OUT1']['NQPSI'] = ''
        andesc['OUT1']['NPRESS'] = ''
        andesc['OUT1']['SIGPRE'] = ''

        ## BASIS ##

        andesc['BASIS'] = dict()
        andesc['BASIS']['KPPFNC'] = ''
        andesc['BASIS']['KPPKNT'] = ''
        andesc['BASIS']['PPKNT'] = ''
        andesc['BASIS']['PPTENS'] = ''
        andesc['BASIS']['KFFFNC'] = ''
        andesc['BASIS']['KFFKNT'] = ''
        andesc['BASIS']['FFKNT'] = ''
        andesc['BASIS']['FFTENS'] = ''
        andesc['BASIS']['KWWFNC'] = ''
        andesc['BASIS']['KWWKNT'] = ''
        andesc['BASIS']['WWKNT'] = ''
        andesc['BASIS']['WWTENS'] = ''
        andesc['BASIS']['PPBDRY'] = ''
        andesc['BASIS']['PP2BDRY'] = ''
        andesc['BASIS']['KPPBDRY'] = ''
        andesc['BASIS']['KPP2BDRY'] = ''
        andesc['BASIS']['FFBDRY'] = ''
        andesc['BASIS']['FF2BDRY'] = ''
        andesc['BASIS']['KFFBDRY'] = ''
        andesc['BASIS']['KFF2BDRY'] = ''
        andesc['BASIS']['WWBDRY'] = ''
        andesc['BASIS']['WW2BDRY'] = ''
        andesc['BASIS']['KWWBDRY'] = ''
        andesc['BASIS']['KWW2BDRY'] = ''
        andesc['BASIS']['KEEFNC'] = ''
        andesc['BASIS']['KEEKNT'] = ''
        andesc['BASIS']['EEKNT'] = ''
        andesc['BASIS']['EETENS'] = ''
        andesc['BASIS']['EEBDRY'] = ''
        andesc['BASIS']['EE2BDRY'] = ''
        andesc['BASIS']['KEEBDRY'] = ''
        andesc['BASIS']['KEE2BDRY'] = ''

        ## CHITOUT ##

        andesc['CHIOUT'] = dict()
        andesc['CHIOUT']['SAISIL'] = ''
        andesc['CHIOUT']['SAIMPI'] = ''
        andesc['CHIOUT']['SAIPR'] = ''
        andesc['CHIOUT']['SAIIP'] = ''

        ### AUX QUANTITIES ###

        aqdesc = gdesc['AuxQuantities']

        aqdesc['R'] = 'all R in the eqdsk grid (m)'
        aqdesc['Z'] = 'all Z in the eqdsk grid (m)'
        aqdesc['PSI'] = 'Poloidal flux in Weber / rad'
        aqdesc['PSI_NORM'] = 'Normalized polodial flux (psin = (psi-min(psi))/(max(psi)-min(psi))'
        aqdesc['PSIRZ'] = 'Poloidal flux in Weber / rad on the rectangular grid points'
        aqdesc['PSIRZ_NORM'] = 'Normalized poloidal flux in Weber / rad on the rectangular grid points'
        aqdesc['RHOp'] = 'sqrt(PSI_NORM)'
        aqdesc['RHOpRZ'] = 'sqrt(PSI_NORM) on the rectangular grid points'
        aqdesc['FPOLRZ'] = 'Poloidal current function on the rectangular grid points'
        aqdesc['PRESRZ'] = 'Pressure on the rectangular grid points'
        aqdesc['QPSIRZ'] = 'Safety factor on the rectangular grid points'
        aqdesc['FFPRIMRZ'] = "FF' on the rectangular grid points"
        aqdesc['PPRIMERZ'] = "P' on the rectangular grid points"
        aqdesc['PRES0RZ'] = 'Pressure by rotation term (eq 26 & 30 of Lao et al., FST 48.2 (2005): 968-977'
        aqdesc['Br'] = 'Radial magnetic field in Tesla on the rectangular grid points'
        aqdesc['Bz'] = 'Vertical magnetic field in Tesla on the rectangular grid points'
        aqdesc['Bp'] = 'Poloidal magnetic field in Tesla on the rectangular grid points'
        aqdesc['Bt'] = 'Toroidal magnetic field in Tesla on the rectangular grid points'
        aqdesc['Jr'] = 'Radial current density on the rectangular grid points'
        aqdesc['Jz'] = 'Vertical current density on the rectangular grid points'
        aqdesc['Jt'] = 'Toroidal current density on the rectangular grid points'
        aqdesc['Jp'] = 'Poloidal current density on the rectangular grid points'
        aqdesc['Jt_fb'] = ''
        aqdesc['Jpar'] = 'Parallel current density on the rectangular grid points'
        aqdesc['PHI'] = 'Toroidal flux in Weber / rad'
        aqdesc['PHI_NORM'] = 'Normalize toroidal flux (phin = (phi-min(phi))/(max(phi)-min(phi))'
        aqdesc['PHIRZ'] = 'Toroidal flux in Weber / rad on the rectangular grid points'
        aqdesc['RHOm'] = 'sqrt(|PHI/pi/BCENTR|)'
        aqdesc['RHO'] = 'sqrt(PHI_NORM)'
        aqdesc['RHORZ'] = 'sqrt(PHI_NORM) on the rectangular grid points'
        aqdesc['Rx1'] = ''
        aqdesc['Zx1'] = ''
        aqdesc['Rx2'] = ''
        aqdesc['Zx2'] = ''

        ### FLUX SURFACES ###

        fsdesc = gdesc['fluxSurfaces']

        ## MAIN ##

        fsdesc['R0'] = gdesc['RMAXIS'] + ' from eqdsk'
        fsdesc['Z0'] = gdesc['ZMAXIS'] + ' from eqdsk'
        fsdesc['RCENTR'] = gdesc['RCENTR']
        fsdesc['R0_interp'] = 'R0 from fit paraboloid in the vicinity of the grid-based center (m)'
        fsdesc['Z0_interp'] = 'Z0 from fit paraboloid in the vicinity of the grid-based center (m)'
        fsdesc['levels'] = "flux surfaces (normalized psi) for the 'flux' tree"
        fsdesc['BCENTR'] = gdesc['BCENTR'] + " (BCENTR = Fpol[-1] / RCENTR)"
        fsdesc['CURRENT'] = gdesc['CURRENT']

        ## FLUX ##

        fsdesc['flux'] = dict()
        fsdesc['flux']['psi'] = 'poloidal flux in Weber / rad on flux surface'
        fsdesc['flux']['R'] = 'R in meters along flux surface surface'
        fsdesc['flux']['Z'] = 'Z in meters along flux surface surface'
        fsdesc['flux']['F'] = 'poloidal current function in m-T on flux surface'
        fsdesc['flux']['P'] = 'pressure in Pa on flux surface'
        fsdesc['flux']['PPRIME'] = 'P’(ψ) in (nt /m2) / (Weber /rad) on flux surface'
        fsdesc['flux']['FFPRIM'] = 'FF’(ψ) in (mT)2 / (Weber /rad) on flux surface'
        fsdesc['flux']['Br'] = 'Br in Tesla along flux surface surface'
        fsdesc['flux']['Bz'] = 'Bz in Tesla along flux surface surface'
        fsdesc['flux']['Jt'] = 'toroidal current density along flux surface'
        fsdesc['flux']['Bmax'] = 'maximum B on flux surface'
        fsdesc['flux']['q'] = 'safety factor on flux surface'

        ## AVG ##

        fsdesc['avg'] = dict()
        fsdesc['avg']['R'] = 'flux surface average of major r (m)'
        fsdesc['avg']['a'] = 'flux surface average of minor r (m)'
        fsdesc['avg']['R**2'] = 'flux surface average of R^2 (m^2)'
        fsdesc['avg']['1/R'] = 'flux surface average of 1/R (1/m)'
        fsdesc['avg']['1/R**2'] = 'flux surface average of 1/R^2 (1/m^2)'
        fsdesc['avg']['Bp'] = 'flux surface average of poloidal B (T)'
        fsdesc['avg']['Bp**2'] = 'flux surface average of Bp^2 (T^2)'
        fsdesc['avg']['Bp*R'] = 'flux surface average of Bp*R (T m)'
        fsdesc['avg']['Bp**2*R**2'] = 'flux surface average of Bp^2*R^2 (T^2 m^2)'
        fsdesc['avg']['Btot'] = 'flux surface average of total B (T)'
        fsdesc['avg']['Btot**2'] = 'flux surface average of Btot^2 (T^2)'
        fsdesc['avg']['Bt'] = 'flux surface average of toroidal B (T)'
        fsdesc['avg']['Bt**2'] = 'flux surface average of Bt^2 (T^2)'
        fsdesc['avg']['ip'] = ''
        fsdesc['avg']['vp'] = ''
        fsdesc['avg']['q'] = 'flux surface average of saftey factor'
        fsdesc['avg']['hf'] = ''
        fsdesc['avg']['Jt'] = 'flux surface average torioidal current density'
        fsdesc['avg']['Jt/R'] = 'flux surface average torioidal current density / R'
        fsdesc['avg']['fc'] = 'flux surface average of passing particle fraction'
        fsdesc['avg']['grad_term'] = ''
        fsdesc['avg']['P'] = 'flux surface average of pressure (Pa)'
        fsdesc['avg']['F'] = 'flux surface average of Poloidal current function F (T m)'
        fsdesc['avg']['PPRIME'] = 'flux surface average of P’ in (nt /m2) / (Weber /rad)'
        fsdesc['avg']['FFPRIM'] = 'flux surface average of FF’ in (mT)2 / (Weber /rad)'
        fsdesc['avg']['dip/dpsi'] = ''
        fsdesc['avg']['Jeff'] = ''
        fsdesc['avg']['beta_t'] = 'volume averaged toroidal beta'
        fsdesc['avg']['beta_n'] = 'volume averaged normalized beta'
        fsdesc['avg']['beta_p'] = 'volume averaged poloidal beta'
        fsdesc['avg']['fcap'] = ''
        fsdesc['avg']['hcap'] = ''
        fsdesc['avg']['gcap'] = ''

        ## GEO ##

        fsdesc['geo'] = dict()
        fsdesc['geo']['psi'] = 'Poloidal flux (Wb / rad)'
        fsdesc['geo']['psin'] = 'Normalized poloidal flux'
        fsdesc['geo']['R'] = 'R0 of each flux surface (m)'
        fsdesc['geo']['R_centroid'] = ''
        fsdesc['geo']['Rmax_centroid'] = ''
        fsdesc['geo']['Rmin_centroid'] = ''
        fsdesc['geo']['Z'] = 'Z0 of each flux surface (m)'
        fsdesc['geo']['Z_centroid'] = ''
        fsdesc['geo']['a'] = 'Minor r (m)'
        fsdesc['geo']['dell'] = 'Lower triangularity'
        fsdesc['geo']['delta'] = 'Average triangularity'
        fsdesc['geo']['delu'] = 'Upper triangularity'
        fsdesc['geo']['eps'] = 'Inverse aspect ratio'
        fsdesc['geo']['kap'] = 'Average elongation'
        fsdesc['geo']['kapl'] = 'Lower elongation'
        fsdesc['geo']['kapu'] = 'Upper elongation'
        fsdesc['geo']['lonull'] = ''
        fsdesc['geo']['per'] = ''
        fsdesc['geo']['surfArea'] = 'Plasma surface area (m^2)'
        fsdesc['geo']['upnull'] = ''
        fsdesc['geo']['zeta'] = 'Average squareness'
        fsdesc['geo']['zetail'] = 'Inner lower squareness'
        fsdesc['geo']['zetaiu'] = 'Inner upper squareness'
        fsdesc['geo']['zetaol'] = 'Outer lower squareness'
        fsdesc['geo']['zetaou'] = 'Outer upper squareness'
        fsdesc['geo']['zoffset'] = ''
        fsdesc['geo']['vol'] = 'Plasma volume (m^3)'
        fsdesc['geo']['cxArea'] = 'Plasma cross-sectional area (m^2)'
        fsdesc['geo']['phi'] = 'Toroidal flux in Weber / rad'
        fsdesc['geo']['bunit'] = ''
        fsdesc['geo']['rho'] = 'sqrt(|PHI/pi/BCENTR|)'
        fsdesc['geo']['rhon'] = 'sqrt(PHI_NORM)'

        ## MIDPLANE ##

        fsdesc['midplane'] = dict()
        fsdesc['midplane']['R'] = 'R values of midplane slice in meters'
        fsdesc['midplane']['Z'] = 'Z values of midplane slice in meters'
        fsdesc['midplane']['Br'] = "Br at (R_midplane, Zmidplane) in Tesla"
        fsdesc['midplane']['Bz'] = "Br at (R_midplane, Zmidplane) in Tesla"
        fsdesc['midplane']['Bp'] = "Bp at (R_midplane, Zmidplane) in Tesla"
        fsdesc['midplane']['Bt'] = "Bt at (R_midplane, Zmidplane) in Tesla"

        ## INFO ##

        fsdesc['info'] = dict()
        fsdesc['info']['J_efit_norm'] = 'EFIT current normalization'

        info_internal_inductance = dict()
        info_internal_inductance['li_from_definition'] = 'Bp2_vol / vol / mu_0^2 / ip&2 * circum^2'
        info_internal_inductance['li_(1)_TLUCE'] = 'li_from_definition / circum^2 * 2 * vol / r_0 * correction_factor'
        info_internal_inductance['li_(2)_TLUCE'] = 'li_from_definition / circum^2 * 2 * vol / r_axis'
        info_internal_inductance['li_(3)_TLUCE'] = 'li_from_definition / circum^2 * 2 * vol / r_0'
        info_internal_inductance['li_(1)_EFIT'] = 'circum^2 * Bp2_vol / (vol * mu_0^2 * ip^2)'
        info_internal_inductance['li_(3)_IMAS'] = '2 * Bp2_vol / r_0 / ip^2 / mu_0^2'
        fsdesc['info']['internal_inductance'] = info_internal_inductance

        info_open_separatrix = dict()
        info_open_separatrix['psi'] = 'psi of last closed flux surface (Wb/rad)'
        info_open_separatrix['rhon'] = 'psi_n of last closed flux surface'
        info_open_separatrix['R'] = 'R of last closed flux surface (m)'
        info_open_separatrix['Z'] = 'Z of last closed flux surface (m)'
        info_open_separatrix['Br'] = 'Br along last closed flux surface (T)'
        info_open_separatrix['Bz'] = 'Bz along last closed flux surface (T)'
        info_open_separatrix['s'] = ''
        info_open_separatrix['mid_index'] = 'index of outer midplane location in open_separatrix arrays'
        info_open_separatrix['rho'] = 'rho of last closed flux surface (Wb/rad)'
        fsdesc['info']['open_separatrix'] = info_open_separatrix

        fsdesc['info']['rvsin'] = ''
        fsdesc['info']['rvsout'] = ''
        fsdesc['info']['zvsin'] = ''
        fsdesc['info']['zvsout'] = ''
        fsdesc['info']['xpoint'] = '(R, Z) of x-point in meters'
        fsdesc['info']['xpoint_inner_strike'] = '(R, Z) of inner strike line near the x-point in meters'
        fsdesc['info']['xpoint_outer_strike'] = '(R, Z) of outer strike line near the x-point in meters'
        fsdesc['info']['xpoint_outer_midplane'] = '(R, Z) of outer LCFS near the x-point in meters'
        fsdesc['info']['xpoint_inner_midplane'] = '(R, Z) of inner LCFS near the x-point in meters'
        fsdesc['info']['xpoint_private_region'] = '(R, Z) of private flux region near the x-point in meters'
        fsdesc['info']['xpoint_outer_region'] = '(R, Z) of outer SOL region near the x-point in meters'
        fsdesc['info']['xpoint_core_region'] = '(R, Z) of core region near the x-point in meters'
        fsdesc['info']['xpoint_inner_region'] = '(R, Z) of inner SOL region near the x-point in meters'
        fsdesc['info']['xpoint2'] = '(R, Z) of second x-point in meters'
        fsdesc['info']['rlim'] = gdesc['RLIM']
        fsdesc['info']['zlim'] = gdesc['ZLIM']

    def add_rhovn(self):
        """
        Calculate RHOVN from PSI and `q` profile
        """
        # add RHOVN if QPSI is non-zero (ie. vacuum gEQDSK)
        if np.sum(np.abs(self['QPSI'])):
            phi = cumulative_trapezoid(self['QPSI'], np.linspace(self['SIMAG'], self['SIBRY'], len(self['QPSI'])), initial=0)
            # only needed if the dimensions of phi are wanted
            # self['RHOVN'] = np.sqrt(np.abs(2 * np.pi * phi / (np.pi * self['BCENTR'])))
            self['RHOVN'] = np.sqrt(np.abs(phi))
            if np.nanmax(self['RHOVN']) > 0:
                self['RHOVN'] = self['RHOVN'] / np.nanmax(self['RHOVN'])
        else:
            # if no QPSI information, then set RHOVN to zeros
            self['RHOVN'] = self['QPSI'] * 0.0

    def cocosify(self, cocosnum, calcAuxQuantities, calcFluxSurfaces, inplace=True):
        """
        Method used to convert gEQDSK quantities to desired COCOS

        :param cocosnum: desired COCOS number (1-8, 11-18)

        :param calcAuxQuantities: add AuxQuantities based on new cocosnum

        :param calcFluxSurfaces: add fluxSurfaces based on new cocosnum

        :param inplace:  change values in True: current gEQDSK, False: new gEQDSK

        :return: gEQDSK with proper cocos
        """

        if inplace:
            gEQDSK = self
        else:
            gEQDSK = copy.deepcopy(self)

        if self.cocos != cocosnum:

            # how different gEQDSK quantities should transform
            transform = cocos_transform(self.cocos, cocosnum)

            # transform the gEQDSK quantities appropriately
            for key in self:
                if key in list(self.transform_signals.keys()):
                    gEQDSK[key] = transform[self.transform_signals[key]] * self[key]

        # set the COCOS attribute of the gEQDSK
        gEQDSK._cocos = cocosnum

        # recalculate AuxQuantities and fluxSurfaces if necessary
        if calcAuxQuantities:
            gEQDSK.addAuxQuantities()
        if calcFluxSurfaces:
            gEQDSK.addFluxSurfaces()

        return gEQDSK

    def addAuxQuantities(self):
        """
        Adds ['AuxQuantities'] to the current object

        :return: SortedDict object containing auxiliary quantities
        """

        self['AuxQuantities'] = self._auxQuantities()

        return self['AuxQuantities']

    def _auxQuantities(self):
        """
        Calculate auxiliary quantities based on the g-file equilibria
        These AuxQuantities obey the COCOS of self.cocos so some sign differences from the gEQDSK file itself

        :return: SortedDict object containing some auxiliary quantities
        """

        aux = dict()
        iterpolationType = 'linear'  # note that interpolation should not be oscillatory -> use linear or pchip

        aux['R'] = np.linspace(0, self['RDIM'], self['NW']) + self['RLEFT']
        aux['Z'] = np.linspace(0, self['ZDIM'], self['NH']) - self['ZDIM'] / 2.0 + self['ZMID']

        if self['CURRENT'] != 0.0:

            # poloidal flux and normalized poloidal flux
            aux['PSI'] = np.linspace(self['SIMAG'], self['SIBRY'], len(self['PRES']))
            aux['PSI_NORM'] = np.linspace(0.0, 1.0, len(self['PRES']))

            aux['PSIRZ'] = self['PSIRZ']
            if self['SIBRY'] != self['SIMAG']:
                aux['PSIRZ_NORM'] = abs((self['PSIRZ'] - self['SIMAG']) / (self['SIBRY'] - self['SIMAG']))
            else:
                aux['PSIRZ_NORM'] = abs(self['PSIRZ'] - self['SIMAG'])
            # rho poloidal
            aux['RHOp'] = np.sqrt(aux['PSI_NORM'])
            aux['RHOpRZ'] = np.sqrt(aux['PSIRZ_NORM'])

            # extend functions in PSI to be clamped at edge value when outside of PSI range (i.e. outside of LCFS)
            dp = aux['PSI'][1] - aux['PSI'][0]
            ext_psi_mesh = np.hstack((aux['PSI'][0] - dp * 1e6, aux['PSI'], aux['PSI'][-1] + dp * 1e6))

            def ext_arr(inv):
                return np.hstack((inv[0], inv, inv[-1]))

            # map functions in PSI to RZ coordinate
            for name in ['FPOL', 'PRES', 'QPSI', 'FFPRIM', 'PPRIME', 'PRESSW', 'PWPRIM']:
                if name in self and len(self[name]):
                    aux[name + 'RZ'] = interpolate.interp1d(ext_psi_mesh, ext_arr(self[name]), kind=iterpolationType, bounds_error=False)(
                        aux['PSIRZ']
                    )

            # Correct Pressure by rotation term (eq 26 & 30 of Lao et al., FST 48.2 (2005): 968-977.
            aux['PRES0RZ'] = copy.deepcopy(aux['PRESRZ'])
            if 'PRESSW' in self:
                aux['PRES0RZ'] = copy.deepcopy(aux['PRESRZ'])
                aux['PPRIME0RZ'] = PP0 = copy.deepcopy(aux['PPRIMERZ'])
                R = aux['R'][None, :]
                R0 = self['RCENTR']
                Pw = aux['PRESSWRZ']
                P0 = aux['PRES0RZ']
                aux['PRESRZ'] = P = P0 * np.exp(Pw / P0 * (R - R0) / R0)
                PPw = aux['PWPRIMRZ']
                aux['PPRIMERZ'] = PP0 * P / P0 * (1.0 - Pw / P0 * (R**2 - R0**2) / R0**2)
                aux['PPRIMERZ'] += PPw * P / P0 * (R**2 - R0**2) / R0**2

        else:
            # vacuum gEQDSK
            aux['PSIRZ'] = self['PSIRZ']

        # from the definition of flux
        COCOS = define_cocos(self.cocos)
        if (aux['Z'][1] != aux['Z'][0]) and (aux['R'][1] != aux['R'][0]):
            [dPSIdZ, dPSIdR] = np.gradient(aux['PSIRZ'], aux['Z'][1] - aux['Z'][0], aux['R'][1] - aux['R'][0])
        else:
            [dPSIdZ, dPSIdR] = np.gradient(aux['PSIRZ'])
        [R, Z] = np.meshgrid(aux['R'], aux['Z'])
        aux['Br'] = (dPSIdZ / R) * COCOS['sigma_RpZ'] * COCOS['sigma_Bp'] / (2.0 * np.pi) ** COCOS['exp_Bp']
        aux['Bz'] = (-dPSIdR / R) * COCOS['sigma_RpZ'] * COCOS['sigma_Bp'] / (2.0 * np.pi) ** COCOS['exp_Bp']
        if self['CURRENT'] != 0.0:
            signa = COCOS['sigma_RpZ'] * COCOS['sigma_rhotp']  # + CW, - CCW
            signBp = signa * np.sign((Z - self['ZMAXIS']) * aux['Br'] - (R - self['RMAXIS']) * aux['Bz'])  # sign(a)*sign(r x B)
            aux['Bp'] = signBp * np.sqrt(aux['Br'] ** 2 + aux['Bz'] ** 2)
            # once I have the poloidal flux as a function of RZ I can calculate the toroidal field (showing DIA/PARAmagnetism)
            aux['Bt'] = aux['FPOLRZ'] / R
        else:
            aux['Bt'] = self['BCENTR'] * self['RCENTR'] / R

        # now the current densities as curl B = mu0 J in cylindrical coords
        if (aux['Z'][2] != aux['Z'][1]) and (aux['R'][2] != aux['R'][1]):
            [dBrdZ, dBrdR] = np.gradient(aux['Br'], aux['Z'][2] - aux['Z'][1], aux['R'][2] - aux['R'][1])
            [dBzdZ, dBzdR] = np.gradient(aux['Bz'], aux['Z'][2] - aux['Z'][1], aux['R'][2] - aux['R'][1])
            [dBtdZ, dBtdR] = np.gradient(aux['Bt'], aux['Z'][2] - aux['Z'][1], aux['R'][2] - aux['R'][1])
            [dRBtdZ, dRBtdR] = np.gradient(R * aux['Bt'], aux['Z'][2] - aux['Z'][1], aux['R'][2] - aux['R'][1])
        else:
            [dBrdZ, dBrdR] = np.gradient(aux['Br'])
            [dBzdZ, dBzdR] = np.gradient(aux['Bz'])
            [dBtdZ, dBtdR] = np.gradient(aux['Bt'])
            [dRBtdZ, dRBtdR] = np.gradient(R * aux['Bt'])

        aux['Jr'] = COCOS['sigma_RpZ'] * (-dBtdZ) / (4 * np.pi * 1e-7)
        aux['Jz'] = COCOS['sigma_RpZ'] * (dRBtdR / R) / (4 * np.pi * 1e-7)
        if 'PCURRT' in self:
            aux['Jt'] = self['PCURRT']
        else:
            aux['Jt'] = COCOS['sigma_RpZ'] * (dBrdZ - dBzdR) / (4 * np.pi * 1e-7)
        if self['CURRENT'] != 0.0:
            signJp = signa * np.sign((Z - self['ZMAXIS']) * aux['Jr'] - (R - self['RMAXIS']) * aux['Jz'])  # sign(a)*sign(r x J)
            aux['Jp'] = signJp * np.sqrt(aux['Jr'] ** 2 + aux['Jz'] ** 2)
            aux['Jt_fb'] = (
                -COCOS['sigma_Bp'] * ((2.0 * np.pi) ** COCOS['exp_Bp']) * (aux['PPRIMERZ'] * R + aux['FFPRIMRZ'] / R / (4 * np.pi * 1e-7))
            )

            aux['Jpar'] = (aux['Jr'] * aux['Br'] + aux['Jz'] * aux['Bz'] + aux['Jt'] * aux['Bt']) / np.sqrt(
                aux['Br'] ** 2 + aux['Bz'] ** 2 + aux['Bt'] ** 2
            )

            # The toroidal flux PHI can be found by recognizing that the safety factor is the ratio of the differential toroidal and poloidal fluxes
            if 'QPSI' in self and len(self['QPSI']):
                aux['PHI'] = (
                    COCOS['sigma_Bp']
                    * COCOS['sigma_rhotp']
                    * cumulative_trapezoid(self['QPSI'], aux['PSI'], initial=0)
                    * (2.0 * np.pi) ** (1.0 - COCOS['exp_Bp'])
                )
                if aux['PHI'][-1] != 0 and np.isfinite(aux['PHI'][-1]):
                    aux['PHI_NORM'] = aux['PHI'] / aux['PHI'][-1]
                else:
                    aux['PHI_NORM'] = aux['PHI'] * np.nan
                    # printw('Warning: unable to properly normalize PHI')
                if abs(np.diff(aux['PSI'])).min() > 0:
                    aux['PHIRZ'] = interpolate.interp1d(
                        aux['PSI'], aux['PHI'], kind=iterpolationType, bounds_error=False, fill_value='extrapolate'
                    )(aux['PSIRZ'])
                else:
                    aux['PHIRZ'] = aux['PSIRZ'] * np.nan
                if self['BCENTR'] != 0:
                    aux['RHOm'] = float(np.sqrt(abs(aux['PHI'][-1] / np.pi / self['BCENTR'])))
                else:
                    aux['RHOm'] = np.nan
                aux['RHO'] = np.sqrt(aux['PHI_NORM'])
                with np.errstate(invalid='ignore'):
                    aux['RHORZ'] = np.nan_to_num(np.sqrt(aux['PHIRZ'] / aux['PHI'][-1]))

        aux['Rx1'], aux['Zx1'] = x_point_search(aux['R'], aux['Z'], self['PSIRZ'], psi_boundary=self['SIBRY'])
        aux['Rx2'], aux['Zx2'] = x_point_search(aux['R'], aux['Z'], self['PSIRZ'], zsign=-np.sign(aux['Zx1']))

        return aux

    def addFluxSurfaces(self, **kw):
        r"""
        Adds ['fluxSurface'] to the current object

        :param \**kw: keyword dictionary passed to fluxSurfaces class

        :return: fluxSurfaces object based on the current gEQDSK file
        """
        if self['CURRENT'] == 0.0:
            # printw('Skipped tracing of fluxSurfaces for vacuum equilibrium')
            return

        options = {}
        options.update(kw)
        options['quiet'] = kw.pop('quiet', self['NW'] <= 129)
        options['levels'] = kw.pop('levels', True)
        options['resolution'] = kw.pop('resolution', 0)
        options['calculateAvgGeo'] = kw.pop('calculateAvgGeo', True)

        # N.B., the middle option accounts for the new version of CHEASE
        #       where self['CASE'][1] = 'OM CHEAS'
        if (
            self['CASE'] is not None
            and self['CASE'][0] is not None
            and self['CASE'][1] is not None
            and ('CHEASE' in self['CASE'][0] or 'CHEAS' in self['CASE'][1] or 'TRXPL' in self['CASE'][0])
        ):
            options['forceFindSeparatrix'] = kw.pop('forceFindSeparatrix', False)
        else:
            options['forceFindSeparatrix'] = kw.pop('forceFindSeparatrix', True)

        try:
            self['fluxSurfaces'] = FluxSurfaces(gEQDSK=self, **options)
        except Exception as _excp:
            # warnings.warn('Error tracing flux surfaces: ' + repr(_excp))
            self['fluxSurfaces'] = RuntimeError('Error tracing flux surfaces: ' + repr(_excp))

        return self['fluxSurfaces']


def gEQDSK_COCOS_identify(bt, ip):
    """
    Returns the native COCOS that an unmodified gEQDSK would obey, defined by sign(Bt) and sign(Ip)
    In order for psi to increase from axis to edge and for q to be positive:
    All use sigma_RpZ=+1 (phi is counterclockwise) and exp_Bp=0 (psi is flux/2.*pi)
    We want
    sign(psi_edge-psi_axis) = sign(Ip)*sigma_Bp > 0  (psi always increases in gEQDSK)
    sign(q) = sign(Ip)*sign(Bt)*sigma_rhotp > 0      (q always positive in gEQDSK)
    ::
        ============================================
        Bt    Ip    sigma_Bp    sigma_rhotp    COCOS
        ============================================
        +1    +1       +1           +1           1
        +1    -1       -1           -1           3
        -1    +1       +1           -1           5
        -1    -1       -1           +1           7
    """
    COCOS = define_cocos(1)

    # get sign of Bt and Ip with respect to CCW phi
    sign_Bt = int(COCOS['sigma_RpZ'] * np.sign(bt))
    sign_Ip = int(COCOS['sigma_RpZ'] * np.sign(ip))
    g_cocos = {
        (+1, +1): 1,  # +Bt, +Ip
        (+1, -1): 3,  # +Bt, -Ip
        (-1, +1): 5,  # -Bt, +Ip
        (-1, -1): 7,  # -Bt, -Ip
        (+1, 0): 1,  # +Bt, No current
        (-1, 0): 3,
    }  # -Bt, No current
    return g_cocos.get((sign_Bt, sign_Ip), None)


# tic = time.time()
# gfile = Geqdsk('../../input/g093536.06060ke')
# toc = time.time()
# print(toc-tic)
# # gfile.load()
# toc = time.time()
# print(toc-tic)
# gfile.add_geqdsk_documentation()
# toc = time.time()
# print(toc-tic)
#
# pass
