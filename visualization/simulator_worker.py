"""Background simulation thread for the visualization dashboard."""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal

from .data_models import FIELD_LABELS, SCALAR_FIELDS, SimulationSettings, VECTOR_FIELDS, load_visual_geometry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.FPAD_simulator import PelletSimulator
from simulator.torax_simulator import TransportSimulator


LAST_OUTPUT_SCALARS = {"Q_fusion", "H98", "q95", "q_min", "W_thermal_total"}
DIAGNOSTIC_SCALARS = {
    "fgw_n_e_volume_avg",
    "P_fusion",
    "tau_E",
    "P_external_total",
    "n_e_volume_avg",
    "T_e_volume_avg",
    "T_i_volume_avg",
    "n_e_core",
    "T_e_core",
    "T_i_core",
    "S_pellet",
}
STATE_VECTOR_FIELDS = {"q_face", "s_face"}
PLASMA_VECTOR_FIELDS = {"T_e", "T_i", "n_e", "n_i"}


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "value"):
        value = value.value
    array = np.asarray(jax.device_get(value))
    return array


def _reshape_sharded(array: np.ndarray, env: TransportSimulator) -> np.ndarray:
    if array.ndim >= 2 and array.shape[0] == env.num_devices and array.shape[1] == env.batch_per_device:
        return array.reshape((env.total_batch_size,) + array.shape[2:])
    return array


class SimulationWorker(QThread):
    sig_log = pyqtSignal(str)
    sig_preview = pyqtSignal(object)
    sig_state = pyqtSignal(object)
    sig_finished = pyqtSignal(object)
    sig_error = pyqtSignal(str)

    def __init__(self, settings: SimulationSettings, parent=None) -> None:
        super().__init__(parent=parent)
        self.settings = settings
        self._stop_requested = False
        self.run_result: dict[str, Any] | None = None

    def stop(self) -> None:
        self._stop_requested = True
        self.sig_log.emit("Stop requested. Waiting for current simulation step to finish.")

    def _extract_last_output_scalar(self, env: TransportSimulator, name: str) -> np.ndarray:
        value = getattr(env.last_outputs, name)
        return _reshape_sharded(_to_numpy(value), env).reshape(env.total_batch_size)

    def _extract_diagnostic_scalar(
        self,
        env: TransportSimulator,
        diagnostics: dict[str, Any],
        name: str,
    ) -> np.ndarray:
        value = diagnostics.get(name)
        if value is None:
            return np.full((env.total_batch_size,), np.nan, dtype=np.float32)
        return _reshape_sharded(_to_numpy(value), env).reshape(env.total_batch_size).astype(np.float32)

    def _extract_state_vector(self, env: TransportSimulator, name: str) -> np.ndarray:
        value = getattr(env.current_states.core_profiles, name)
        return _reshape_sharded(_to_numpy(value), env)

    def run(self) -> None:
        try:
            self.run_result = self._run_impl()
            self.sig_finished.emit(
                {
                    "run_id": self.run_result["metadata"]["run_id"],
                    "executed_steps": self.run_result["metadata"]["executed_steps"],
                    "stopped_early": self.run_result["metadata"]["stopped_early"],
                }
            )
        except Exception:
            error_message = traceback.format_exc()
            self.sig_error.emit(error_message)

    def _run_impl(self) -> dict[str, Any]:
        settings = self.settings
        geometry = load_visual_geometry(settings.config_path)

        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        if settings.config_path:
            self.sig_log.emit(f"Config: {settings.config_path}")
        self.sig_log.emit(f"Batch size: {settings.batch_size} | Steps: {settings.total_steps}")

        device = torch.device(device_name)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.set_device(device)

        env = TransportSimulator(settings.config_path, total_batch_size=settings.batch_size)
        pellet_sim = PelletSimulator(device=device)
        self.sig_log.emit("TransportSimulator and PelletSimulator initialized.")

        scalar_fields = [field for field in settings.selected_fields if field in SCALAR_FIELDS]
        vector_fields = [field for field in settings.selected_fields if field in VECTOR_FIELDS]

        results_scalars = {field: [] for field in scalar_fields}
        results_vectors = {field: [] for field in vector_fields}
        preview_scalar_history = {field: [] for field in scalar_fields}

        current_triggers = torch.zeros((settings.batch_size,), dtype=torch.bool, device=device)
        current_locs = torch.full((settings.batch_size,), 0.5, dtype=torch.float32, device=device)
        current_widths = torch.full((settings.batch_size,), 0.05, dtype=torch.float32, device=device)
        current_rates = torch.zeros((settings.batch_size,), dtype=torch.float32, device=device)

        velocity = torch.zeros((settings.batch_size, 2), dtype=torch.float32, device=device)
        thickness = torch.zeros((settings.batch_size,), dtype=torch.float32, device=device)
        thickness.fill_(settings.thickness_m)

        rho_axis: np.ndarray | None = None
        start_time = time.time()
        executed_steps = 0

        for step in range(settings.total_steps):
            if self._stop_requested:
                break

            env.step(current_triggers, current_locs, current_widths, current_rates)
            T_e, n_e, _P_e, T_i, n_i, _P_i, sp_D, sp_T = env.get_plasma_tensor()
            pellet_sim.update_plasma_state(T_e, n_e, T_i, n_i, sp_D, sp_T)
            diagnostics = env.get_diagnostics()

            if rho_axis is None:
                rho_axis = np.linspace(0.0, 1.0, T_e.shape[1], dtype=np.float32)

            next_should_inject = ((step + 1) % max(1, settings.injection_interval_ms)) == 0
            if next_should_inject:
                next_triggers = torch.ones((settings.batch_size,), dtype=torch.bool, device=device)
            else:
                next_triggers = torch.zeros((settings.batch_size,), dtype=torch.bool, device=device)

            velocity[:, 0].fill_(settings.velocity_mps)
            velocity[:, 1].fill_(0.0)
            sim_loc, sim_width, sim_rate = pellet_sim.simulate_pellet_injection(
                batch_size=settings.batch_size,
                velocity=velocity,
                thickness=thickness,
            )

            injection_debug = {
                "injection_applied": current_triggers.float().detach().cpu().numpy(),
                "injection_commanded": next_triggers.float().detach().cpu().numpy(),
                "pellet_location": current_locs.detach().cpu().numpy(),
                "pellet_width": current_widths.detach().cpu().numpy(),
                "pellet_rate": current_rates.detach().cpu().numpy(),
            }

            current_triggers = next_triggers
            current_locs = sim_loc
            current_widths = sim_width
            current_rates = sim_rate * next_triggers.float()

            scalar_cache: dict[str, np.ndarray] = {}
            for field in scalar_fields:
                if field in injection_debug:
                    value = injection_debug[field].astype(np.float32)
                elif field in diagnostics:
                    value = self._extract_diagnostic_scalar(env, diagnostics, field)
                elif field in LAST_OUTPUT_SCALARS:
                    value = self._extract_last_output_scalar(env, field).astype(np.float32)
                else:
                    continue
                scalar_cache[field] = value
                results_scalars[field].append(value)
                preview_scalar_history[field].append(float(value[0]))

            vector_cache: dict[str, np.ndarray] = {}
            plasma_vectors = {
                "T_e": _to_numpy(T_e).astype(np.float32),
                "T_i": _to_numpy(T_i).astype(np.float32),
                "n_e": _to_numpy(n_e).astype(np.float32),
                "n_i": _to_numpy(n_i).astype(np.float32),
            }
            for field in vector_fields:
                if field in PLASMA_VECTOR_FIELDS:
                    value = plasma_vectors[field]
                elif field in STATE_VECTOR_FIELDS:
                    value = self._extract_state_vector(env, field).astype(np.float32)
                else:
                    continue
                vector_cache[field] = value
                results_vectors[field].append(value)

            executed_steps = step + 1
            status_scalars = {
                "Q_fusion": self._extract_last_output_scalar(env, "Q_fusion").astype(np.float32),
                "fGW": self._extract_diagnostic_scalar(env, diagnostics, "fgw_n_e_volume_avg"),
                "n_e_core": self._extract_diagnostic_scalar(env, diagnostics, "n_e_core"),
                "T_e_core": self._extract_diagnostic_scalar(env, diagnostics, "T_e_core"),
                "T_i_core": self._extract_diagnostic_scalar(env, diagnostics, "T_i_core"),
            }

            if step % max(1, settings.preview_every_steps) == 0 or executed_steps == settings.total_steps:
                elapsed = max(time.time() - start_time, 1e-6)
                first_status = {name: float(values[0]) for name, values in status_scalars.items()}
                print(
                    "[Visualization preview] "
                    f"step={executed_steps}/{settings.total_steps} "
                    f"Q_fusion={first_status['Q_fusion']:.6g} "
                    f"fGW={first_status['fGW']:.6g} "
                    f"n_e_core={first_status['n_e_core']:.6g} "
                    f"T_e_core={first_status['T_e_core']:.6g} "
                    f"T_i_core={first_status['T_i_core']:.6g}",
                    flush=True,
                )
                latest_preview = {
                    "step": executed_steps,
                    "total_steps": settings.total_steps,
                    "scalar_history": {
                        field: np.asarray(values, dtype=np.float32)
                        for field, values in preview_scalar_history.items()
                    },
                    "vector_snapshot": {
                        field: values[0].astype(np.float32)
                        for field, values in vector_cache.items()
                    },
                    "rho_axis": rho_axis,
                    "geometry": geometry,
                    "sps": executed_steps / elapsed,
                    "batch_size": settings.batch_size,
                    "status": {
                        "Q_fusion": first_status["Q_fusion"],
                        "fGW": first_status["fGW"],
                        "n_e_core": first_status["n_e_core"],
                        "T_e_core": first_status["T_e_core"],
                        "T_i_core": first_status["T_i_core"],
                        "injection": float(injection_debug["injection_applied"][0]),
                    },
                }
                self.sig_preview.emit(latest_preview)
                self.sig_state.emit(
                    {
                        "step": executed_steps,
                        "total_steps": settings.total_steps,
                        "sps": executed_steps / elapsed,
                        "status": latest_preview["status"],
                    }
                )

            if step % 50 == 0:
                q_value = status_scalars["Q_fusion"][0]
                fgw_value = status_scalars["fGW"][0]
                self.sig_log.emit(
                    f"Step {executed_steps:5d}/{settings.total_steps} | "
                    f"Q={q_value:.3f} | fGW={fgw_value:.3f} | "
                    f"next_injection={'yes' if next_should_inject else 'no'}"
                )

        elapsed = max(time.time() - start_time, 1e-6)
        metadata = settings.to_metadata()
        metadata.update(
            {
                "run_id": time.strftime("%Y%m%d_%H%M%S"),
                "executed_steps": executed_steps,
                "stopped_early": bool(self._stop_requested),
                "elapsed_sec": elapsed,
                "sps": executed_steps / elapsed,
                "geometry": geometry,
                "preview_fields": {
                    "scalar": scalar_fields,
                    "vector": vector_fields,
                },
            }
        )

        run_result = {
            "metadata": metadata,
            "time_axis": np.arange(executed_steps, dtype=np.int32),
            "scalars": {
                field: np.stack(values, axis=0).astype(np.float32)
                if values else np.empty((0, settings.batch_size), dtype=np.float32)
                for field, values in results_scalars.items()
            },
            "vectors": {
                field: np.stack(values, axis=0).astype(np.float32)
                if values else np.empty((0, settings.batch_size, 0), dtype=np.float32)
                for field, values in results_vectors.items()
            },
            "rho_axis": rho_axis if rho_axis is not None else np.empty((0,), dtype=np.float32),
            "field_labels": {
                field: FIELD_LABELS.get(field, field)
                for field in settings.selected_fields
            },
        }
        self.sig_log.emit(
            f"Simulation finished. Executed {executed_steps} steps at {metadata['sps']:.1f} steps/s."
        )
        return run_result
