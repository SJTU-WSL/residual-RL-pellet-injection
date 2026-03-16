"""Plot widgets used by the Qt dashboard."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .data_models import FIELD_COLORS, FIELD_LABELS


class ScalarTrendWidget(pg.PlotWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.showGrid(x=True, y=True, alpha=0.18)
        self.setClipToView(True)
        self.addLegend(offset=(12, 12))
        self.setLabel("bottom", "Step")
        self.setTitle("Scalar Trends")
        self._curves: dict[str, pg.PlotDataItem] = {}

    def update_series(self, step_axis: np.ndarray, series: dict[str, np.ndarray]) -> None:
        current_fields = set(series.keys())
        stale_fields = [name for name in self._curves if name not in current_fields]
        for name in stale_fields:
            self.removeItem(self._curves[name])
            del self._curves[name]

        for name, values in series.items():
            if name not in self._curves:
                self._curves[name] = self.plot(
                    pen=pg.mkPen(FIELD_COLORS.get(name, "#8ab4f8"), width=2),
                    name=FIELD_LABELS.get(name, name),
                )
            self._curves[name].setData(step_axis, np.asarray(values, dtype=np.float32))

    def clear_plot(self) -> None:
        self.update_series(np.empty((0,), dtype=np.float32), {})


class RadialProfileWidget(pg.PlotWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.showGrid(x=True, y=True, alpha=0.18)
        self.setLabel("bottom", "Normalized Radius")
        self.setTitle("First Batch Radial Profile")
        self._curve = self.plot(pen=pg.mkPen("#50fa7b", width=2.5))

    def update_profile(self, rho: np.ndarray, values: np.ndarray, field_name: str) -> None:
        color = FIELD_COLORS.get(field_name, "#50fa7b")
        self._curve.setPen(pg.mkPen(color, width=2.5))
        self._curve.setData(rho, values)
        self.setTitle(FIELD_LABELS.get(field_name, field_name))

    def clear_plot(self) -> None:
        self._curve.setData([], [])
        self.setTitle("First Batch Radial Profile")


def build_cross_section_mesh(profile: np.ndarray, geometry: dict[str, float], n_r: int = 120, n_theta: int = 200):
    profile = np.asarray(profile, dtype=np.float64).reshape(-1)
    rho_src = (np.arange(profile.size, dtype=np.float64) + 0.5) / max(profile.size, 1)
    rho_dense = np.linspace(0.0, 1.0, n_r)

    xp = np.concatenate(([0.0], rho_src, [1.0]))
    fp = np.concatenate(([profile[0]], profile, [profile[-1]]))
    profile_dense = np.interp(rho_dense, xp, fp)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=True)
    rho_grid, theta_grid = np.meshgrid(rho_dense, theta, indexing="ij")

    delta = float(np.clip(geometry.get("delta", 0.4), -0.99, 0.99))
    kappa = float(geometry.get("kappa", 1.8))
    a_minor = float(geometry.get("a_minor", 2.0))
    r_major = float(geometry.get("R_major", 6.2))

    radial = a_minor * rho_grid
    r_grid = r_major + radial * np.cos(theta_grid + np.arcsin(delta) * np.sin(theta_grid))
    z_grid = kappa * radial * np.sin(theta_grid)
    value_grid = np.tile(profile_dense[:, None], (1, n_theta))
    return r_grid, z_grid, value_grid


class PoloidalCrossSectionCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None) -> None:
        self.figure = Figure(figsize=(5, 5), facecolor="#11161d")
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes.set_facecolor("#11161d")
        self.figure.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
        self._colorbar = None
        self._draw_placeholder()

    def _clear_colorbar(self) -> None:
        if self._colorbar is None:
            return
        try:
            colorbar_axes = getattr(self._colorbar, "ax", None)
            if colorbar_axes is not None and colorbar_axes in self.figure.axes:
                self._colorbar.remove()
        except (KeyError, ValueError):
            # Matplotlib can already detach the colorbar axes during redraws.
            pass
        finally:
            self._colorbar = None

    def _draw_placeholder(self) -> None:
        self._clear_colorbar()
        self.axes.clear()
        self.axes.set_facecolor("#11161d")
        self.axes.text(
            0.5,
            0.5,
            "2D cross section preview",
            color="#7f8ba3",
            ha="center",
            va="center",
            transform=self.axes.transAxes,
        )
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.draw_idle()

    def clear_plot(self) -> None:
        self._draw_placeholder()

    def update_profile(self, field_name: str, profile: np.ndarray, geometry: dict[str, float]) -> None:
        if profile.size == 0:
            self._draw_placeholder()
            return

        self._clear_colorbar()

        r_grid, z_grid, values = build_cross_section_mesh(profile, geometry)
        self.axes.clear()
        self.axes.set_facecolor("#11161d")
        contour = self.axes.contourf(r_grid, z_grid, values, levels=60, cmap="magma")
        self.axes.plot(r_grid[-1, :], z_grid[-1, :], color="#cfd8dc", linewidth=1.2, linestyle="--")
        self.axes.set_aspect("equal")
        self.axes.set_title(f"{FIELD_LABELS.get(field_name, field_name)} Cross Section", color="#e6edf7")
        self.axes.set_xlabel("R [m]", color="#aeb7c2")
        self.axes.set_ylabel("Z [m]", color="#aeb7c2")
        self.axes.tick_params(colors="#aeb7c2")
        self._colorbar = self.figure.colorbar(contour, ax=self.axes, pad=0.03, shrink=0.86)
        self._colorbar.ax.tick_params(colors="#aeb7c2")
        self._colorbar.outline.set_edgecolor("#2d3340")
        self.draw_idle()
