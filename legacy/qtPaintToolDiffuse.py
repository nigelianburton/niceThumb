"""
Modular Diffuse Tool for NiceThumb (Template-based)

This class implements a self-contained diffusion tool for PaintView.
It manages its own UI, options, and network calls to a stable diffusion server.
It replaces the image and clears overlays only when a new image is received.

Features:
- Provides metadata (name, display flags) for toolbar and UI logic.
- Returns a complex options widget (prompt, model, config, LoRA).
- Manages all network calls and async job handling.
- Emits signals for image save, error, and busy state.
- No palette, brush, or brush slider required.

Usage:
- Instantiate and register with the tool system.
- Call `on_selected` when the tool is activated, passing canvas and callbacks.
- Use the returned options widget for all user interaction.
"""

from __future__ import annotations
from PyQt6 import QtCore, QtGui, QtWidgets
from pathlib import Path
from typing import Optional, Callable, List, Set, Any
import os
import json
import base64
import time
import urllib.request
import urllib.error
import threading
from qtPaintHelpers import next_edit_filename
from qt_paint_tools.qtPaintToolUtilities import make_brush_cursor

# Fixed Qwen model and local LoRA directory
QWEN_MODEL_NAME = r"C:\_MODELS-SD\Qwen\Qwen-Image-Edit"
QWEN_LIGHTNING_DIR = r"C:\_CONDA\niceThumb\Qwen-Image-Lightning"

def _list_safetensors_dir(dir_path: str) -> List[str]:
    try:
        if not os.path.isdir(dir_path):
            return []
        return sorted([f for f in os.listdir(dir_path) if f.lower().endswith(".safetensors")])
    except Exception:
        return []

runColumnHStretch = 0.5
configColumnHStretch = 0.25
loraColumnHStretch = 0.25
marginDefault=(0,0,0,0)
spacingDefault=2

class PaintToolDiffuse(QtCore.QObject):
    name = "diffuse"
    display_brush_controls = False
    display_palette = False
    display_tool_options = True
    display_cursor_preview = False

    saved = QtCore.pyqtSignal(Path)
    error = QtCore.pyqtSignal(str)
    busyChanged = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._api_base = os.environ.get("NT6DIFF_API", "http://127.0.0.1:5015").rstrip("/")

        self._current_path: Optional[Path] = None
        self._compose_image_provider: Optional[Callable[[], Optional[QtGui.QImage]]] = None
        self._mask_image_provider: Optional[Callable[[], Optional[QtGui.QImage]]] = None

        self._diffuse_ready = False
        self._sdxl_models: List[str] = []
        self._all_loras: List[str] = []
        self._active_loras: Set[str] = set()
        self._last_used_seed: Optional[int] = None

        self._models_timer: Optional[QtCore.QTimer] = None
        self._loras_timer: Optional[QtCore.QTimer] = None
        self._progress_timer: Optional[QtCore.QTimer] = None
        self._job_id: Optional[str] = None

        self._canvas = None
        self._options_widget: Optional[QtWidgets.QWidget] = None

        self._build_ui()
        QtCore.QTimer.singleShot(0, self._start_models_poll)
        QtCore.QTimer.singleShot(0, self._start_loras_poll)

    def button_name(self) -> str:
        return "Diffuse"

    def create_options_widget(self) -> QtWidgets.QWidget:
        # Ensure the options widget exists (may have been deleted by parent)
        self._ensure_ui_alive()
        return self._options_widget

    def on_selected(
        self,
        canvas,
        root_path: Path,
        current_path: Optional[Path],
        compose_image_provider: Callable[[], Optional[QtGui.QImage]],
        mask_image_provider: Callable[[], Optional[QtGui.QImage]],
        tool_callback: Optional[Callable[[str, str | bool], None]] = None
    ) -> dict:
        if canvas is None:
            print("[PaintToolDiffuse] ERROR: canvas is None in on_selected")
            return {}
        if root_path is None:
            print("[PaintToolDiffuse] ERROR: root_path is None in on_selected")
            return {}
        if compose_image_provider is None:
            print("[PaintToolDiffuse] ERROR: compose_image_provider is None in on_selected")
            return {}
        self._canvas = canvas
        self._current_path = current_path
        self._compose_image_provider = compose_image_provider
        self._mask_image_provider = mask_image_provider
        self._tool_callback = tool_callback

        # Make sure our UI exists before we touch any child widgets
        self._ensure_ui_alive()

        # --- Now the UI is built and combo exists, so refresh it! ---
        self._refresh_model_combo()
        self._refresh_model_controls()

        return {
            "display_brush_controls": self.display_brush_controls,
            "display_palette": self.display_palette,
            "display_tool_options": self.display_tool_options,
            "display_cursor_preview": self.display_cursor_preview,
        }

    # ---------- UI Construction ----------
    def _build_ui(self):
        self._options_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(self._options_widget)
        main_layout.setContentsMargins(*marginDefault)
        main_layout.setSpacing(spacingDefault)

        # --- Left Column (65%) ---
        left_col = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_col)
        left_layout.setContentsMargins(*marginDefault)
        left_layout.setSpacing(spacingDefault)

        # Q toggle + model combo
        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(*marginDefault)
        top_row.setSpacing(spacingDefault)
        self.cb_qwen = QtWidgets.QCheckBox("Q")
        self.cb_qwen.setToolTip("Qwen Edit")
        self.cb_qwen.toggled.connect(self._refresh_model_controls)
        # Also refresh LoRA source when toggled
        self.cb_qwen.toggled.connect(self._on_qwen_toggled)
        self.cbo_model = QtWidgets.QComboBox()
        self.cbo_model.setEnabled(False)
        top_row.addWidget(self.cb_qwen, 0)
        top_row.addWidget(self.cbo_model, 1)
        left_layout.addLayout(top_row)

        # Prompt box
        self.te_prompt = QtWidgets.QTextEdit()
        self.te_prompt.setPlaceholderText("prompt...")
        self.te_prompt.setAcceptRichText(False)
        fm = self._options_widget.fontMetrics()
        self.te_prompt.setFixedHeight(int(fm.lineSpacing() * 3.2))
        left_layout.addWidget(self.te_prompt, 0)

        # --- Sub-columns for progress/buttons and config ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.destroyed.connect(lambda: setattr(self, "_splitter", None))

        # Sub-column 1: Progress block
        progress_col = QtWidgets.QVBoxLayout()
        progress_col.setContentsMargins(*marginDefault)
        progress_col.setSpacing(spacingDefault)
        self.pb_progress = QtWidgets.QProgressBar()
        self.pb_progress.setRange(0, 100)
        self.pb_progress.setValue(0)
        self.pb_progress.setTextVisible(False)
        self.pb_progress.setMinimumWidth(0)  # allow shrinking
        progress_col.addWidget(self.pb_progress)

        self.btn_t2i = QtWidgets.QPushButton("T2I")
        self.btn_i2i = QtWidgets.QPushButton("I2I")
        self.btn_t2i.setEnabled(False)
        self.btn_i2i.setEnabled(False)
        self.btn_t2i.clicked.connect(self._on_t2i_clicked)
        self.btn_i2i.clicked.connect(self._on_i2i_clicked)

        btns_row = QtWidgets.QHBoxLayout()
        btns_row.setContentsMargins(*marginDefault)
        btns_row.setSpacing(spacingDefault)
        btns_row.addWidget(self.btn_t2i)
        btns_row.addWidget(self.btn_i2i)
        progress_col.addLayout(btns_row)

        seed_row = QtWidgets.QHBoxLayout()
        seed_row.setContentsMargins(*marginDefault)
        seed_row.setSpacing(spacingDefault)
        self.cb_lock_seed = QtWidgets.QCheckBox("")
        self.cb_lock_seed.setToolTip("Lock seed")
        self.sp_seed = QtWidgets.QSpinBox()
        self.sp_seed.setRange(0, 2**31 - 1)
        self.sp_seed.setEnabled(False)
        self.cb_lock_seed.toggled.connect(self.sp_seed.setEnabled)
        self.cb_lock_seed.toggled.connect(self._on_seed_lock_toggled)
        seed_row.addWidget(self.cb_lock_seed)
        seed_row.addWidget(QtWidgets.QLabel("Seed"))
        seed_row.addWidget(self.sp_seed, 1)
        progress_col.addLayout(seed_row)

        progress_col.addStretch(1)
        progress_w = QtWidgets.QWidget()
        progress_w.setLayout(progress_col)
        progress_w.setMinimumWidth(0)

        # Sub-column 2: Config (sliders)
        config_col = QtWidgets.QVBoxLayout()
        config_col.setContentsMargins(*marginDefault)
        config_col.setSpacing(spacingDefault)
        self._config_grid = QtWidgets.QGridLayout()
        self._config_grid.setContentsMargins(*marginDefault)
        self._config_grid.setHorizontalSpacing(spacingDefault)
        self._config_grid.setVerticalSpacing(spacingDefault)

        # SDXL sliders
        self.lbl_gs = QtWidgets.QLabel("Guidance")
        self.sl_guidance = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_guidance.setRange(10, 200)
        self.sl_guidance.setValue(105)
        self.val_gs = QtWidgets.QLabel("10.5")
        self.val_gs.setFixedWidth(44)
        self.sl_guidance.valueChanged.connect(lambda v: self.val_gs.setText(f"{v/10.0:.1f}"))

        self.lbl_steps = QtWidgets.QLabel("Steps")
        self.sl_steps = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_steps.setRange(1, 200)
        self.sl_steps.setValue(40)
        self.val_steps = QtWidgets.QLabel("40")
        self.val_steps.setFixedWidth(44)
        self.sl_steps.valueChanged.connect(lambda v: self.val_steps.setText(str(v)))

        self.lbl_strength = QtWidgets.QLabel("Strength (I2I)")
        self.sl_strength = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_strength.setRange(0, 100)
        self.sl_strength.setValue(80)
        self.val_strength = QtWidgets.QLabel("0.80")
        self.val_strength.setFixedWidth(44)
        self.sl_strength.valueChanged.connect(lambda v: self.val_strength.setText(f"{v/100.0:.2f}"))

        # Qwen sliders
        self.lbl_q_truecfg = QtWidgets.QLabel("True CFG")
        self.sl_q_truecfg = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_q_truecfg.setRange(10, 200)
        self.sl_q_truecfg.setValue(40)
        self.val_q_truecfg = QtWidgets.QLabel("4.0")
        self.val_q_truecfg.setFixedWidth(44)
        self.sl_q_truecfg.valueChanged.connect(lambda v: self.val_q_truecfg.setText(f"{v/10.0:.1f}"))

        self.lbl_q_steps = QtWidgets.QLabel("Steps")
        self.sl_q_steps = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_q_steps.setRange(1, 200)
        self.sl_q_steps.setValue(50)
        self.val_q_steps = QtWidgets.QLabel("50")
        self.val_q_steps.setFixedWidth(44)
        self.sl_q_steps.valueChanged.connect(lambda v: self.val_q_steps.setText(f"{v}"))

        # Grid layout (make the slider column stretchy)
        row = 0
        self._config_grid.addWidget(self.lbl_gs, row, 0)
        self._config_grid.addWidget(self.sl_guidance, row, 1)
        self._config_grid.addWidget(self.val_gs, row, 2); row += 1
        self._config_grid.addWidget(self.lbl_steps, row, 0)
        self._config_grid.addWidget(self.sl_steps, row, 1)
        self._config_grid.addWidget(self.val_steps, row, 2); row += 1
        self._config_grid.addWidget(self.lbl_strength, row, 0)
        self._config_grid.addWidget(self.sl_strength, row, 1)
        self._config_grid.addWidget(self.val_strength, row, 2); row += 1
        self._config_grid.addWidget(self.lbl_q_truecfg, row, 0)
        self._config_grid.addWidget(self.sl_q_truecfg, row, 1)
        self._config_grid.addWidget(self.val_q_truecfg, row, 2); row += 1
        self._config_grid.addWidget(self.lbl_q_steps, row, 0)
        self._config_grid.addWidget(self.sl_q_steps, row, 1)
        self._config_grid.addWidget(self.val_q_steps, row, 2)

        # slider column stretches; labels/value columns do not
        self._config_grid.setColumnStretch(0, 0)
        self._config_grid.setColumnStretch(1, 1)
        self._config_grid.setColumnStretch(2, 0)

        config_col.addLayout(self._config_grid)
        config_col.addStretch(1)
        config_w = QtWidgets.QWidget()
        config_w.setLayout(config_col)
        config_w.setMinimumWidth(0)

        splitter.addWidget(progress_w)
        splitter.addWidget(config_w)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        # Apply 40:60 once after first real resize, then stop
        self._splitter = splitter
        self._split_ratio_applied = False
        self._options_widget.installEventFilter(self)

        left_layout.addWidget(splitter, 1)

        # --- Right Column (35%) ---  [RESTORED]
        right_col = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_col)
        right_layout.setContentsMargins(*marginDefault)
        right_layout.setSpacing(spacingDefault)

        self.list_loras = QtWidgets.QListWidget()
        self.list_loras.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list_loras.itemSelectionChanged.connect(self._update_lora_buttons)
        right_layout.addWidget(self.list_loras, 1)

        btnrow = QtWidgets.QHBoxLayout()
        btnrow.setContentsMargins(*marginDefault)
        btnrow.setSpacing(spacingDefault)
        self.btn_lora_add = QtWidgets.QPushButton("+")
        self.btn_lora_remove = QtWidgets.QPushButton("-")
        self.btn_lora_add.clicked.connect(self._on_lora_add)
        self.btn_lora_remove.clicked.connect(self._on_lora_remove)
        btnrow.addStretch(1)
        btnrow.addWidget(self.btn_lora_add)
        btnrow.addWidget(self.btn_lora_remove)
        right_layout.addLayout(btnrow)

        # --- Assemble columns + stretch ---
        main_layout.addWidget(left_col, 65)
        main_layout.addWidget(right_col, 35)

        # Initial populate
        self._refresh_model_combo()
        self._refresh_model_controls()
        self._fetch_loras()

    def _ensure_ui_alive(self):
        """
        If the options widget (or any critical child) was deleted by Qt, rebuild the UI.
        """
        def _is_deleted(w):
            if w is None:
                return True
            try:
                # Access a trivial property; will raise RuntimeError if deleted
                _ = w.objectName()
                return False
            except RuntimeError:
                return True

        if _is_deleted(getattr(self, "_options_widget", None)) \
           or _is_deleted(getattr(self, "cb_qwen", None)) \
           or _is_deleted(getattr(self, "cbo_model", None)) \
           or _is_deleted(getattr(self, "te_prompt", None)):
            self._build_ui()

    # ---------- Model/Lora Polling ----------
    def _start_models_poll(self):
        if not self._fetch_models():
            self._models_timer = QtCore.QTimer(self._options_widget)
            self._models_timer.setInterval(1000)
            self._models_timer.timeout.connect(self._fetch_models)
            self._models_timer.start()

    def _fetch_models(self) -> bool:
        try:
            data = self._get_json(f"{self._api_base}/models")
            models = data.get("models") or []
            self._sdxl_models = [m for m in models if isinstance(m, str) and m.strip()]
            self._diffuse_ready = len(self._sdxl_models) > 0
            self._refresh_model_combo()
            self._refresh_model_controls()
            if self._models_timer:
                self._models_timer.stop()
                self._models_timer.deleteLater()
                self._models_timer = None
            return True
        except Exception:
            return False

    def _start_loras_poll(self):
        # Start a poll; will fill from server (SDXL) or local (Qwen) depending on toggle
        if not self._fetch_loras():
            self._loras_timer = QtCore.QTimer(self._options_widget)
            self._loras_timer.setInterval(1200)
            self._loras_timer.timeout.connect(self._fetch_loras)
            self._loras_timer.start()

    def _fetch_loras(self) -> bool:
        try:
            if self.cb_qwen.isChecked():
                # Qwen mode: only local Qwen Lightning LoRAs
                self._all_loras = _list_safetensors_dir(QWEN_LIGHTNING_DIR)
            else:
                # SDXL mode: from server
                data = self._get_json(f"{self._api_base}/loras")
                loras = data.get("loras") or []
                # Keep only sane names
                self._all_loras = [n for n in loras if isinstance(n, str) and n.strip()]
            self._refresh_lora_list()
            if self._loras_timer:
                self._loras_timer.stop()
                self._loras_timer.deleteLater()
                self._loras_timer = None
            return True
        except Exception:
            return False

    # ---------- Actions ----------
    def _guidance_value(self) -> float:
        return self.sl_guidance.value() / 10.0

    def _strength_value(self) -> float:
        return self.sl_strength.value() / 100.0

    def _true_cfg_value(self) -> float:
        return self.sl_q_truecfg.value() / 10.0

    def _seed_value(self) -> Optional[int]:
        return int(self.sp_seed.value()) if self.cb_lock_seed.isChecked() else None

    def _on_seed_lock_toggled(self, checked: bool):
        """
        When seed lock is enabled, display the most recently used seed.
        If none or 0, default to 17.
        """
        if checked:
            val = self._last_used_seed if isinstance(self._last_used_seed, int) and self._last_used_seed != 0 else 17
            try:
                self.sp_seed.blockSignals(True)
                self.sp_seed.setValue(int(val))
            finally:
                self.sp_seed.blockSignals(False)

    def _selected_loras(self) -> List[str]:
        """Return a stable-ordered list of active LoRAs."""
        try:
            return sorted(self._active_loras)
        except Exception:
            return []

    def _on_t2i_clicked(self):
        if not self._diffuse_ready:
            return
        is_qwen = self.cb_qwen.isChecked()
        prompt = self.te_prompt.toPlainText().strip()
        seed = self._seed_value()
        if isinstance(seed, int):
            self._last_used_seed = int(seed)
        if is_qwen:
            payload = {
                "model_file": QWEN_MODEL_NAME,  # send fixed model name
                "prompt": prompt,
                "true_cfg_scale": self._true_cfg_value(),
                "num_inference_steps": int(self.sl_q_steps.value()),
                "generator": seed,
            }
            l = self._selected_qwen_lora()
            if l:
                payload["lora"] = l
            self.log_prompt_request("qwen_t2i", prompt, params=payload)
            endpoint = "/qwen_edit_async"
        else:
            # SDXL expects: model_file, prompt, width/height, guidance_scale, num_inference_steps, generator
            payload = {
                "model_file": self.cbo_model.currentText().strip(),
                "prompt": prompt,
                "width": 1024, "height": 1024,
                "guidance_scale": self._guidance_value(),
                "num_inference_steps": int(self.sl_steps.value()),
                "generator": seed,
            }
            # Submit all selected LoRAs (server supports a list)
            loras = self._selected_loras()
            if loras:
                payload["loras"] = loras
            self.log_prompt_request("sdxl_t2i", prompt, params=payload)
            endpoint = "/t2i_async"
        self._start_async_job(endpoint, payload)

    def _on_i2i_clicked(self):
        if not self._diffuse_ready:
            return
        img = self._compose_image_provider() if self._compose_image_provider else None
        if img is None or img.isNull():
            self._emit_error("I2I: no source image")
            return
        is_qwen = self.cb_qwen.isChecked()
        prompt = self.te_prompt.toPlainText().strip()
        seed = self._seed_value()
        if isinstance(seed, int):
            self._last_used_seed = int(seed)
        if is_qwen:
            payload = {
                "model_file": QWEN_MODEL_NAME,  # send fixed model name
                "prompt": prompt,
                "true_cfg_scale": self._true_cfg_value(),
                "num_inference_steps": int(self.sl_q_steps.value()),
                "generator": seed,
                "init_image": self._qimage_to_data_url(img),
            }
            if self._mask_image_provider:
                m = self._mask_image_provider()
                if m and not m.isNull():
                    if m.size() != img.size():
                        m = m.scaled(img.size(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                    payload["mask_image"] = self._qimage_to_data_url(m)
            l = self._selected_qwen_lora()
            if l:
                payload["lora"] = l
            self.log_prompt_request("qwen_i2i", prompt, params={k: v for k, v in payload.items() if k not in ("init_image", "mask_image")})
            endpoint = "/qwen_edit_async"
        else:
            payload = {
                "model_file": self.cbo_model.currentText().strip,
                "prompt": prompt,
                "width": img.width(), "height": img.height(),
                "guidance_scale": self._guidance_value(),
                "num_inference_steps": int(self.sl_steps.value()),
                "strength": self._strength_value(),
                "generator": seed,
                "init_image": self._qimage_to_data_url(img),
            }
            if self._mask_image_provider:
                m = self._mask_image_provider()
                if m and not m.isNull():
                    if m.size() != img.size():
                        m = m.scaled(img.size(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                    payload["mask_image"] = self._qimage_to_data_url(m)
            # Submit all selected LoRAs for SDXL I2I
            loras = self._selected_loras()
            if loras:
                payload["loras"] = loras
            # Log without huge base64 fields
            log_params = {k: v for k, v in payload.items() if k not in ("init_image", "mask_image")}
            self.log_prompt_request("sdxl_i2i", prompt, params=log_params)
            endpoint = "/i2i_async"
        self._start_async_job(endpoint, payload)

    def _start_async_job(self, path: str, payload: dict):
        self._set_busy(True)
        try:
            # Persist the seed being used for this job (if provided)
            gen = payload.get("generator", None)
            if isinstance(gen, int):
                self._last_used_seed = int(gen)
            res = self._post_json(f"{self._api_base}{path}", payload)
            jid = (res or {}).get("job_id")
            if not jid:
                 raise RuntimeError("missing job_id")
            self._job_id = jid
            if self._progress_timer is None:
                self._progress_timer = QtCore.QTimer(self._options_widget)
                self._progress_timer.setInterval(250)
                self._progress_timer.timeout.connect(self._poll_progress)
            if not self._progress_timer.isActive():
                self._progress_timer.start()
            self.pb_progress.setValue(0)
        except Exception as ex:
            self._set_busy(False); self._emit_error(str(ex))

    def _poll_progress(self):
        if not self._job_id:
            return
        try:
            res = self._get_json(f"{self._api_base}/progress/{self._job_id}")
        except Exception:
            if self._tool_callback:
                self._tool_callback("error", "Server not found")
            return
        self.pb_progress.setValue(int(res.get("percent", 0)))
        st = res.get("status")
        if st == "done":
            if self._progress_timer and self._progress_timer.isActive():
                self._progress_timer.stop()
            self._job_id = None
            data_url = res.get("image_data_url", "")
            out = self._save_data_url_image(data_url, self._current_path)
            self._set_busy(False)
            if out:
                if self._tool_callback:
                    self._tool_callback("save", str(out))
                # Clear overlays and mask, replace image
                if self._canvas:
                    if hasattr(self._canvas, "set_pixmap"):
                        pm = QtGui.QPixmap(str(out))
                        self._canvas.set_pixmap(pm)
                    if hasattr(self._canvas, "clear_overlay"):
                        self._canvas.clear_overlay()
                    if hasattr(self._canvas, "_mask_overlay"):
                        self._canvas._mask_overlay = None
            else:
                self._emit_error("Invalid image data")
        elif st == "error":
            if self._progress_timer and self._progress_timer.isActive():
                self._progress_timer.stop()
            self._job_id = None
            self._set_busy(False)
            self._emit_error(res.get("error", "Unknown error"))

    # ---------- Model/Lora UI ----------
    def _refresh_model_combo(self):
        cbo = getattr(self, "cbo_model", None)
        if cbo is None:
            return
        is_qwen = self.cb_qwen.isChecked()
        cbo.blockSignals(True)
        cbo.clear()
        if is_qwen:
            # Show fixed Qwen model and disable combo
            cbo.addItems([QWEN_MODEL_NAME])
            cbo.setCurrentIndex(0)
            cbo.setEnabled(False)
        else:
            cbo.addItems(self._sdxl_models)
            cbo.setEnabled(cbo.count() > 0 and self._diffuse_ready)
            if cbo.count() > 0:
                cbo.setCurrentIndex(0)
        cbo.blockSignals(False)

    def _refresh_model_controls(self):
        is_qwen = self.cb_qwen.isChecked()
        self._refresh_model_combo()
        if hasattr(self, "btn_t2i"):
            self.btn_t2i.setVisible(not is_qwen)
            self.btn_t2i.setEnabled(self._diffuse_ready and not is_qwen)
        if hasattr(self, "btn_i2i"):
            self.btn_i2i.setEnabled(self._diffuse_ready)
        # Toggle sliders visibility
        for w in (self.lbl_gs, self.sl_guidance, self.val_gs,
                  self.lbl_steps, self.sl_steps, self.val_steps,
                  self.lbl_strength, self.sl_strength, self.val_strength):
            w.setVisible(not is_qwen)
        for w in (self.lbl_q_truecfg, self.sl_q_truecfg, self.val_q_truecfg,
                  self.lbl_q_steps, self.sl_q_steps, self.val_q_steps):
            w.setVisible(is_qwen)

    def _on_qwen_toggled(self, checked: bool):
        # Refresh LoRA list source on toggle
        self._fetch_loras()
        self._update_lora_buttons()
        if checked:
            # Qwen defaults: CFG=1.0, Steps=4
            try:
                self.sl_q_truecfg.setValue(10)  # 1.0
                self.sl_q_steps.setValue(4)
            except Exception:
                pass
            # Select the first LoRA by default if none selected
            try:
                if self.list_loras.count() > 0 and self.list_loras.currentItem() is None:
                    self.list_loras.setCurrentRow(0)
            except Exception:
                pass
        else:
            # SDXL defaults: Guidance=10.5, Steps=40, Strength=0.80
            try:
                self.sl_guidance.setValue(105)  # 10.5
                self.sl_steps.setValue(40)
                self.sl_strength.setValue(80)   # 0.80
            except Exception:
                pass

    def _refresh_lora_list(self):
        # Only show Qwen LoRAs when Q is selected; otherwise only SDXL LoRAs
        def is_qwen_lightning(name: str) -> bool:
            s = (name or "").lower()
            return ("qwen" in s) and ("lightning" in s)

        show_qwen = self.cb_qwen.isChecked()
        cur = self.list_loras.currentItem().text() if self.list_loras.currentItem() else ""
        self.list_loras.blockSignals(True)
        self.list_loras.clear()
        for name in self._all_loras:
            if show_qwen and not is_qwen_lightning(name) and QWEN_LIGHTNING_DIR:
                # Local Qwen lightning file names may not contain 'qwen'/'lightning'; show as-is
                pass
            it = QtWidgets.QListWidgetItem(name)
            if name in self._active_loras:
                it.setIcon(self._tick_icon())
            self.list_loras.addItem(it)
        # Restore previous selection if possible
        if cur:
            matches = self.list_loras.findItems(cur, QtCore.Qt.MatchFlag.MatchExactly)
            if matches:
                self.list_loras.setCurrentItem(matches[0])
        # If Qwen and nothing selected yet, select first item by default
        if show_qwen and self.list_loras.currentItem() is None and self.list_loras.count() > 0:
            self.list_loras.setCurrentRow(0)
        self.list_loras.blockSignals(False)
        self._update_lora_buttons()

    def _selected_qwen_lora(self) -> Optional[str]:
        it = self.list_loras.currentItem()
        if not it:
            return None
        name = it.text().strip()
        if not name:
            return None
        # Return absolute path for server to load
        return name if os.path.isabs(name) else os.path.join(QWEN_LIGHTNING_DIR, name)

    def _update_lora_buttons(self):
        it = self.list_loras.currentItem()
        if not it:
            self.btn_lora_add.setEnabled(False)
            self.btn_lora_remove.setEnabled(False)
            self.btn_lora_add.setVisible(True)
            self.btn_lora_remove.setVisible(True)
            return
        name = it.text()
        active = name in self._active_loras
        self.btn_lora_add.setVisible(not active)
        self.btn_lora_add.setEnabled(not active)
        self.btn_lora_remove.setVisible(active)
        self.btn_lora_remove.setEnabled(active)

    def _on_lora_add(self):
        it = self.list_loras.currentItem()
        if not it: return
        self._active_loras.add(it.text()); self._refresh_lora_list()

    def _on_lora_remove(self):
        it = self.list_loras.currentItem()
        if not it: return
        self._active_loras.discard(it.text()); self._refresh_lora_list()

    # ---------- Helpers ----------
    def _get_json(self, url: str) -> dict:
        req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=1.2) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _qimage_to_data_url(self, img: QtGui.QImage) -> str:
        buf = QtCore.QBuffer(); buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        img.save(buf, "PNG")
        b64 = base64.b64encode(bytes(buf.data())).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _save_data_url_image(self, data_url: str, suggest_from: Optional[Path]) -> Optional[Path]:
        try:
            if not isinstance(data_url, str) or "," not in data_url:
                return None
            _, b64 = data_url.split(",", 1); raw = base64.b64decode(b64)
            img = QtGui.QImage()
            if not img.loadFromData(raw): return None
            if suggest_from and isinstance(suggest_from, Path) and suggest_from.exists():
                out = next_edit_filename(suggest_from)
            else:
                base_dir = self._root_path if self._root_path.exists() else Path.cwd()
                out = base_dir / f"t2i_{int(time.time())}.png"
            return out if img.save(str(out)) else None
        except Exception:
            return None

    def _tick_icon(self) -> QtGui.QIcon:
        sz = 14; pm = QtGui.QPixmap(sz, sz); pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm); p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        pen = QtGui.QPen(QtGui.QColor("#1DB954")); pen.setWidth(2); p.setPen(pen)
        p.drawLine(int(sz * 0.2), int(sz * 0.55), int(sz * 0.8), int(sz * 0.3))
        p.end(); return QtGui.QIcon(pm)

    # Replace signal usage with callback
    def _set_busy(self, busy: bool):
        if self._tool_callback:
            self._tool_callback("busy", busy)
        for w in (
            self.btn_t2i, self.btn_i2i, self.te_prompt, self.cbo_model,
            self.cb_qwen, self.sl_guidance, self.sl_steps, self.sl_strength,
            self.sl_q_truecfg, self.sl_q_steps, self.cb_lock_seed, self.sp_seed
        ):
            w.setEnabled(not busy)
        if busy:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.BusyCursor)
        else:
            try:
                QtWidgets.QApplication.restoreOverrideCursor()
            except Exception:
                pass

    def _emit_error(self, msg: str):
        if self._tool_callback:
            self._tool_callback("error", str(msg))

    def log_prompt_request(self, kind, prompt, params: Optional[dict] = None):
        redacted = {}
        if isinstance(params, dict):
            redacted = dict(params)
            for k in ("init_image", "mask_image", "image_data_url"):
                if k in redacted and isinstance(redacted[k], str):
                    redacted[k] = f"<{k} omitted: {len(redacted[k])} chars>"
        print(f"[qtDiffuseClient][{kind}] prompt={repr(prompt)} params={redacted if params else {}}")

    # ---------- Template-required stubs ----------
    def on_mouse_event(self, event_type: str, pos: QtCore.QPoint, left_down: bool, right_down: bool):
        """Diffuse tool does not handle mouse events."""
        pass

    def paintOverlay(self, canvas, painter: QtGui.QPainter):
        """Optional: Draw additional overlays on the canvas."""
        pass

    def cursorFor(self, canvas) -> Optional[QtGui.QCursor]:
        """Optional: Return a custom QCursor for this tool."""
        return None

    def eventFilter(self, obj, event):
        if obj is self._options_widget and event.type() == QtCore.QEvent.Type.Resize:
            if not getattr(self, "_split_ratio_applied", False):
                splitter = getattr(self, "_splitter", None)
                if splitter is not None:
                    try:
                        w = splitter.width()
                    except RuntimeError:
                        # Underlying C++ object was deleted; drop reference
                        self._splitter = None
                    else:
                        if w > 20:
                            splitter.setSizes([int(w * 0.4), int(w * 0.6)])
                            self._split_ratio_applied = True
        return super().eventFilter(obj, event)