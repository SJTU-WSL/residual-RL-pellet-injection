"""Qt dashboard for parallel pellet injection simulation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUALIZATION_ROOT = PROJECT_ROOT / "visualization"
RUNTIME_CACHE_ROOT = VISUALIZATION_ROOT / ".runtime_cache"
RUNTIME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(RUNTIME_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_ROOT / "matplotlib"))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.data_models import (
    DEFAULT_SELECTED_FIELDS,
    FIELD_LABELS,
    SCALAR_FIELDS,
    VECTOR_FIELDS,
    SimulationSettings,
    build_run_directory,
    default_iter_path,
    merge_selected_fields,
    save_run_bundle,
    split_supported_fields,
)
from visualization.plotting import PoloidalCrossSectionCanvas, RadialProfileWidget, ScalarTrendWidget
from visualization.simulator_worker import SimulationWorker
from visualization.theme import APP_STYLESHEET, configure_pyqtgraph


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.worker: SimulationWorker | None = None
        self.latest_result: dict | None = None
        self.latest_preview: dict | None = None
        self.last_saved_paths: dict[str, str] | None = None

        self.setWindowTitle("NuclearRL Parallel Pellet Injection Console")
        self.resize(1680, 980)
        self.setStyleSheet(APP_STYLESHEET)

        self._build_ui()
        self._populate_output_list()
        self._set_defaults()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        title = QLabel("Parallel Pellet Injection Simulation")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #f4f7fb;")
        subtitle = QLabel("Dark-theme Qt dashboard for ITER-based batch simulation")
        subtitle.setProperty("class", "caption")

        header = QVBoxLayout()
        header.addWidget(title)
        header.addWidget(subtitle)
        root_layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([340, 900, 360])
        root_layout.addWidget(splitter, 1)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        setup_box = QGroupBox("Simulation Setup")
        setup_layout = QVBoxLayout(setup_box)

        config_row = QHBoxLayout()
        self.config_path_edit = QLineEdit()
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_config)
        config_row.addWidget(self.config_path_edit, 1)
        config_row.addWidget(browse_button)
        setup_layout.addWidget(QLabel("ITER config path"))
        setup_layout.addLayout(config_row)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 2048)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 1_000_000)
        self.steps_spin.setSingleStep(100)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 10_000)
        self.interval_spin.setSuffix(" ms")

        self.velocity_spin = QDoubleSpinBox()
        self.velocity_spin.setRange(1.0, 5000.0)
        self.velocity_spin.setDecimals(1)
        self.velocity_spin.setSuffix(" m/s")

        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(0.1, 20.0)
        self.thickness_spin.setDecimals(2)
        self.thickness_spin.setSuffix(" mm")

        form.addRow("Parallel batch", self.batch_spin)
        form.addRow("Simulation steps", self.steps_spin)
        form.addRow("Injection interval", self.interval_spin)
        form.addRow("Pellet velocity", self.velocity_spin)
        form.addRow("Pellet thickness", self.thickness_spin)
        setup_layout.addLayout(form)

        output_box = QGroupBox("Output Selection")
        output_layout = QVBoxLayout(output_box)
        output_layout.addWidget(QLabel("Preset outputs"))
        self.output_list = QListWidget()
        self.output_list.setMinimumHeight(260)
        output_layout.addWidget(self.output_list, 1)
        output_layout.addWidget(QLabel("Custom outputs (comma separated)"))
        self.custom_output_edit = QLineEdit()
        self.custom_output_edit.setPlaceholderText("Example: q95, q_face")
        output_layout.addWidget(self.custom_output_edit)

        control_box = QGroupBox("Controls")
        control_layout = QHBoxLayout(control_box)
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.save_button)

        layout.addWidget(setup_box)
        layout.addWidget(output_box, 1)
        layout.addWidget(control_box)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        scalar_box = QGroupBox("Scalar Monitoring")
        scalar_layout = QVBoxLayout(scalar_box)
        self.scalar_plot = ScalarTrendWidget()
        scalar_layout.addWidget(self.scalar_plot)

        lower_row = QHBoxLayout()

        profile_box = QGroupBox("1D Profile")
        profile_layout = QVBoxLayout(profile_box)
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Vector field"))
        self.vector_field_combo = QComboBox()
        self.vector_field_combo.currentTextChanged.connect(self._refresh_vector_views)
        combo_row.addWidget(self.vector_field_combo, 1)
        profile_layout.addLayout(combo_row)
        self.profile_plot = RadialProfileWidget()
        profile_layout.addWidget(self.profile_plot)

        cross_section_box = QGroupBox("2D Cross Section")
        cross_layout = QVBoxLayout(cross_section_box)
        self.cross_section_canvas = PoloidalCrossSectionCanvas()
        cross_layout.addWidget(self.cross_section_canvas)

        lower_row.addWidget(profile_box, 1)
        lower_row.addWidget(cross_section_box, 1)

        layout.addWidget(scalar_box, 3)
        layout.addLayout(lower_row, 2)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        status_box = QGroupBox("Run Status")
        status_layout = QGridLayout(status_box)
        status_layout.setHorizontalSpacing(16)
        status_layout.setVerticalSpacing(10)

        self.status_labels: dict[str, QLabel] = {}
        status_items = [
            ("state", "State"),
            ("step", "Step"),
            ("sps", "Speed"),
            ("Q_fusion", "Q Fusion"),
            ("fGW", "fGW"),
            ("n_e_core", "n_e Core"),
            ("T_e_core", "T_e Core"),
            ("T_i_core", "T_i Core"),
            ("injection", "Injection"),
        ]

        for row, (key, title) in enumerate(status_items):
            label = QLabel(title)
            label.setProperty("class", "caption")
            value = QLabel("--")
            value.setProperty("class", "metricValue")
            self.status_labels[key] = value
            status_layout.addWidget(label, row, 0)
            status_layout.addWidget(value, row, 1)

        path_box = QGroupBox("Paths")
        path_layout = QVBoxLayout(path_box)
        self.save_path_label = QLabel("Pending save path")
        self.save_path_label.setWordWrap(True)
        self.save_path_label.setProperty("class", "caption")
        path_layout.addWidget(self.save_path_label)

        log_box = QGroupBox("Runtime Log")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "font-family: Menlo, Consolas, monospace; font-size: 11px; color: #dfe8ff;"
        )
        log_layout.addWidget(self.log_view)

        layout.addWidget(status_box)
        layout.addWidget(path_box)
        layout.addWidget(log_box, 1)
        return panel

    def _populate_output_list(self) -> None:
        for field_name in SCALAR_FIELDS + VECTOR_FIELDS:
            item = QListWidgetItem(FIELD_LABELS.get(field_name, field_name))
            item.setData(Qt.UserRole, field_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if field_name in DEFAULT_SELECTED_FIELDS else Qt.Unchecked)
            self.output_list.addItem(item)

    def _set_defaults(self) -> None:
        self.config_path_edit.setText(default_iter_path())
        self.batch_spin.setValue(4)
        self.steps_spin.setValue(10_000)
        self.interval_spin.setValue(100)
        self.velocity_spin.setValue(300.0)
        self.thickness_spin.setValue(4.0)
        self.vector_field_combo.addItem("n_e")

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ITER config file",
            self.config_path_edit.text() or str(PROJECT_ROOT / "config"),
            "Python files (*.py)",
        )
        if path:
            self.config_path_edit.setText(path)

    def _checked_output_fields(self) -> list[str]:
        fields: list[str] = []
        for index in range(self.output_list.count()):
            item = self.output_list.item(index)
            if item.checkState() == Qt.Checked:
                fields.append(item.data(Qt.UserRole))
        return fields

    def _build_settings(self) -> tuple[SimulationSettings, list[str]]:
        selected_fields = merge_selected_fields(
            self._checked_output_fields(),
            self.custom_output_edit.text(),
        )
        if not selected_fields:
            selected_fields = DEFAULT_SELECTED_FIELDS.copy()

        scalar_fields, vector_fields, unsupported = split_supported_fields(selected_fields)
        cleaned_fields = scalar_fields + vector_fields
        if not vector_fields:
            cleaned_fields.append("n_e")
            vector_fields.append("n_e")

        settings = SimulationSettings(
            config_path=self.config_path_edit.text().strip() or default_iter_path(),
            batch_size=int(self.batch_spin.value()),
            total_steps=int(self.steps_spin.value()),
            injection_interval_ms=int(self.interval_spin.value()),
            velocity_mps=float(self.velocity_spin.value()),
            thickness_mm=float(self.thickness_spin.value()),
            selected_fields=cleaned_fields,
            preview_every_steps=10,
        )
        return settings, unsupported

    def start_simulation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Simulation running", "Stop the current run before starting a new one.")
            return

        settings, unsupported = self._build_settings()
        if unsupported:
            self._append_log(f"Ignoring unsupported custom outputs: {', '.join(unsupported)}")

        self.latest_result = None
        self.latest_preview = None
        self.last_saved_paths = None
        self.scalar_plot.clear_plot()
        self.profile_plot.clear_plot()
        self.cross_section_canvas.clear_plot()
        self.log_view.clear()
        self._set_status_defaults()

        self.worker = SimulationWorker(settings)
        self.worker.sig_log.connect(self._append_log)
        self.worker.sig_preview.connect(self._on_preview)
        self.worker.sig_state.connect(self._on_state)
        self.worker.sig_error.connect(self._on_error)
        self.worker.sig_finished.connect(self._on_finished)
        self.worker.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.save_path_label.setText(f"Pending save to {build_run_directory('latest')}")
        self.status_labels["state"].setText("Running")

    def stop_simulation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.stop_button.setEnabled(False)

    def save_results(self) -> None:
        if self.latest_result is None:
            self._append_log("No completed run is available to save.")
            return
        run_id = self.latest_result["metadata"]["run_id"]
        save_dir = build_run_directory(run_id)
        self.last_saved_paths = save_run_bundle(self.latest_result, save_dir)
        self.save_path_label.setText(str(save_dir))
        self._append_log(f"Saved run bundle to {save_dir}")

    def _on_preview(self, payload: dict) -> None:
        self.latest_preview = payload
        scalar_history = payload.get("scalar_history", {})
        if scalar_history:
            first_series = next(iter(scalar_history.values()))
            step_axis = payload.get("step")
            x_axis = np.arange(len(first_series), dtype=np.float32)
            self.scalar_plot.update_series(x_axis, scalar_history)

        vector_snapshot = payload.get("vector_snapshot", {})
        if vector_snapshot:
            current_text = self.vector_field_combo.currentText()
            existing_items = {self.vector_field_combo.itemText(i) for i in range(self.vector_field_combo.count())}
            for field in vector_snapshot:
                if field not in existing_items:
                    self.vector_field_combo.addItem(field)
            if current_text not in vector_snapshot:
                self.vector_field_combo.setCurrentText(next(iter(vector_snapshot)))
            self._refresh_vector_views()

        status = payload.get("status", {})
        self.status_labels["step"].setText(f"{payload.get('step', 0)} / {payload.get('total_steps', 0)}")
        self.status_labels["sps"].setText(f"{payload.get('sps', 0.0):.1f} steps/s")
        self.status_labels["Q_fusion"].setText(f"{status.get('Q_fusion', float('nan')):.3f}")
        self.status_labels["fGW"].setText(f"{status.get('fGW', float('nan')):.3f}")
        self.status_labels["n_e_core"].setText(f"{status.get('n_e_core', float('nan')):.3e}")
        self.status_labels["T_e_core"].setText(f"{status.get('T_e_core', float('nan')):.3f}")
        self.status_labels["T_i_core"].setText(f"{status.get('T_i_core', float('nan')):.3f}")
        self.status_labels["injection"].setText("ON" if status.get("injection", 0.0) > 0.5 else "OFF")

    def _refresh_vector_views(self) -> None:
        if not self.latest_preview:
            return
        vector_snapshot = self.latest_preview.get("vector_snapshot", {})
        field = self.vector_field_combo.currentText()
        if not field or field not in vector_snapshot:
            return
        rho_axis = self.latest_preview.get("rho_axis")
        profile = vector_snapshot[field]
        self.profile_plot.update_profile(rho_axis, profile, field)
        self.cross_section_canvas.update_profile(field, profile, self.latest_preview["geometry"])

    def _on_state(self, payload: dict) -> None:
        self.status_labels["state"].setText("Running")
        self.status_labels["step"].setText(f"{payload.get('step', 0)} / {payload.get('total_steps', 0)}")
        self.status_labels["sps"].setText(f"{payload.get('sps', 0.0):.1f} steps/s")

    def _on_error(self, error_message: str) -> None:
        self._append_log(error_message)
        self.status_labels["state"].setText("Error")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(False)
        QMessageBox.critical(self, "Simulation error", "The simulation thread exited with an error. See the log panel.")

    def _on_finished(self, payload: dict) -> None:
        if self.worker is not None:
            self.latest_result = self.worker.run_result
        self.status_labels["state"].setText("Finished")
        self.status_labels["step"].setText(
            f"{payload.get('executed_steps', 0)} / {self.steps_spin.value()}"
        )
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(self.latest_result is not None)
        if self.latest_result is not None:
            run_id = self.latest_result["metadata"]["run_id"]
            self.save_path_label.setText(str(build_run_directory(run_id)))

    def _append_log(self, message: str) -> None:
        self.log_view.append(message)
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_status_defaults(self) -> None:
        for label in self.status_labels.values():
            label.setText("--")
        self.status_labels["state"].setText("Idle")


def main() -> int:
    configure_pyqtgraph()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
