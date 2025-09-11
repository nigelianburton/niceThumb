from __future__ import annotations
import sys
import os
import time
import threading
import subprocess
import urllib.request
import urllib.error
import json
import socket
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional
from PyQt6 import QtCore, QtWidgets
# from PyPyQt6 import QtCore, QtWidgets  # NOTE: If this is a typo, revert to: from PyQt6 import QtCore, QtWidgets

from qtBrowse import mount_browse, DEFAULT_THUMB_SIZE
from qtBrowse import VIDEO_SUFFIXES
from qtPreview1 import PreviewView
from qtPaintMain import PaintView

import traceback
import shutil  # <-- added for terminal width sizing

# ---------- Diffusion (qtd) service endpoint/script ----------
_DEFAULT_QTD_HOST = "127.0.0.1"
_DEFAULT_QTD_PORT = 5015
_raw_qtd_server = os.environ.get("QTD_SERVER", "").strip()

def _normalize_server_url(raw: str) -> str:
    if not raw:
        return f"http://{_DEFAULT_QTD_HOST}:{_DEFAULT_QTD_PORT}"
    if "://" not in raw:
        raw = "http://" + raw
    return raw.rstrip("/")

DIFFUSION_BASE_URL = _normalize_server_url(_raw_qtd_server if _raw_qtd_server else "")
DIFFUSION_PING_URL = f"{DIFFUSION_BASE_URL}/ping"

if _raw_qtd_server:
    print("[diffusion][config] QTD_SERVER override detected.")
    print(f"[diffusion][config] Using server base: {DIFFUSION_BASE_URL}")
    print(f"[diffusion][config] Ping URL:         {DIFFUSION_PING_URL}")
    _default_base = f"http://{_DEFAULT_QTD_HOST}:{_DEFAULT_QTD_PORT}"
    print(f"[diffusion][config] Default (no QTD_SERVER) would be: {_default_base}")
    print(f"[diffusion][config] Windows (cmd) to set default explicitly:")
    print(f"    SET QTD_SERVER={_default_base}")
else:
    pass

DIFFUSION_SCRIPT = (Path(__file__).with_name("qtd") / "qtdServer.py").resolve()

def require_port_open(url: str, timeout: float = 0.3) -> None:
    p = urlparse(url)
    host = p.hostname or "127.0.0.1"
    port = p.port or (443 if p.scheme == "https" else 80)
    with socket.create_connection((host, port), timeout=timeout):
        return

def _start_minimized_console(py_script: Path) -> Optional[subprocess.Popen]:
    try:
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_CONSOLE
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 1
            return subprocess.Popen(
                [sys.executable, str(py_script)],
                creationflags=creationflags,
                startupinfo=startupinfo,
                close_fds=False,
            )
        else:
            return subprocess.Popen([sys.executable, str(py_script)], close_fds=False)
    except Exception as e:
        print("[qtMain][launch] Failed spawning diffusion script:", py_script)
        traceback.print_exc()
        raise RuntimeError(f"Could not launch diffusion server script: {py_script}") from e

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, start_path: Path, thumbs_dir: Path):
        super().__init__()
        self.setWindowTitle("NiceThumb (PyQt6)")
        self.setMinimumSize(800, 600)

        self._selected_path: Optional[Path] = None
        self._mcp_info_printed = False  # <-- prevent duplicate prints

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, central)
        layout.addWidget(self.splitter, 1)

        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.right_stack = QtWidgets.QStackedWidget()
        self.preview = PreviewView()
        self.paint = PaintView(root_path=start_path)
        self.right_stack.addWidget(self.preview)
        self.right_stack.addWidget(self.paint)
        self.right_stack.setCurrentIndex(0)

        app_state = {
            "current_path": str(start_path),
            "thumb_size": DEFAULT_THUMB_SIZE,
        }

        def on_item_selected(p: Optional[Path]):
            self._selected_path = p
            self.preview.set_path(p)
            is_video = bool(p and p.is_file() and p.suffix.lower() in set(VIDEO_SUFFIXES))
            if is_video:
                print(f"Selected video file: {p}")
                self._re_enable_browse()
                self.right_stack.setCurrentWidget(self.preview)
            else:
                if self.right_stack.currentWidget() is self.paint:
                    self.paint.set_path(p)

        def on_edit_request():
            if hasattr(self, "browser") and self.browser:
                self.browser.set_interactive(False)
            self.paint.set_path(self._selected_path)
            self.right_stack.setCurrentWidget(self.paint)

        self.browser = mount_browse(left_container, app_state, on_item_selected, start_path, thumbs_dir, on_edit_request)

        if hasattr(self.browser, "btn_llm"):
            self.browser.btn_llm.clicked.connect(self._activate_llm_mode)
        self.browser.llm_mode_requested.connect(self._activate_llm_mode)

        self.paint.canceled.connect(lambda: (self._re_enable_browse(), self.right_stack.setCurrentWidget(self.preview)))

        def _on_saved(out_path: Path):
            self._selected_path = out_path
            if hasattr(self, "browser") and self.browser:
                self.browser.refresh_and_select(out_path)
                self.browser.set_interactive(True)
            self.preview.set_path(out_path)
            self.right_stack.setCurrentWidget(self.preview)
        self.paint.saved.connect(_on_saved)

        def _on_diffused(out_path: Path):
            self._selected_path = out_path
            if hasattr(self, "browser") and self.browser:
                self.browser.refresh_and_select(out_path)
                self.browser.set_interactive(False)
            self.paint.set_path(self._selected_path)
            self.right_stack.setCurrentWidget(self.paint)
        self.paint.diffused.connect(_on_diffused)

        self.splitter.addWidget(left_container)
        self.splitter.addWidget(self.right_stack)
        self.splitter.setChildrenCollapsible(False)

        self._svc_procs: dict[str, subprocess.Popen] = {}
        QtCore.QTimer.singleShot(0, self._set_splitter_half)
        QtCore.QTimer.singleShot(0, self._ensure_diffusion_async)
        QtCore.QTimer.singleShot(0, self._print_mcp_config_info)  # <-- replaced dialog with console print

    @QtCore.pyqtSlot()
    def _refocus_mainwindow(self):
        try:
            self.activateWindow()
            self.raise_()
            self.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason)
        except Exception:
            print("[qtMain][refocus] Failed to refocus main window.")
            traceback.print_exc()

    def _re_enable_browse(self):
        try:
            if hasattr(self, "browser") and self.browser:
                self.browser.set_interactive(True)
        except Exception:
            print("[qtMain][browse] Re-enable failed.")
            traceback.print_exc()

    def _set_splitter_half(self):
        total = max(1, self.splitter.width())
        half = total // 2
        self.splitter.setSizes([half, total - half])

    def _ensure_diffusion_async(self):
        t = threading.Thread(target=self._ensure_diffusion, name="diffusion_boot", daemon=True)
        t.start()

    def _ensure_diffusion(self):
        service_running = True
        try:
            require_port_open(DIFFUSION_PING_URL, timeout=0.3)
        except OSError:
            service_running = False
        if not service_running:
            proc = _start_minimized_console(DIFFUSION_SCRIPT)
            if proc:
                self._svc_procs["diffusion"] = proc
                QtCore.QMetaObject.invokeMethod(self, "_refocus_mainwindow", QtCore.Qt.ConnectionType.QueuedConnection)
                for _ in range(20):
                    try:
                        require_port_open(DIFFUSION_PING_URL, timeout=0.3)
                        break
                    except OSError:
                        time.sleep(0.25)

    def _build_mcp_config_json(self) -> str:
        py_exe = Path(sys.executable)
        mcp_path = Path(__file__).with_name("qtMcp.py").resolve()
        if not py_exe.exists():
            py_exe = Path("C:/Users/nigel/miniconda3/envs/niceThumb/python.exe")
        if not mcp_path.exists():
            mcp_path = Path("C:/_CONDA/niceThumb/qtMcp.py")
        cfg = {
            "mcpServers": {
                "qtGenerate": {
                    "command": py_exe.as_posix(),
                    "args": [mcp_path.as_posix()]
                }
            }
        }
        return json.dumps(cfg, indent=2)

    # ---------------- MCP CONFIG (Console Print Replacement) ------------------
    def _print_mcp_config_info(self):
        if self._mcp_info_printed:
            return
        self._mcp_info_printed = True
        try:
            env_name = os.environ.get("CONDA_DEFAULT_ENV", "niceThumb")
            txt = self._build_mcp_config_json().splitlines()
            term_width = shutil.get_terminal_size((100, 40)).columns
            term_width = max(60, min(140, term_width))
            title = f" MCP CONFIG (env: {env_name}) "
            box_char_h = "─"
            inner_width = term_width - 2
            def clamp_line(s: str) -> str:
                if len(s) <= inner_width - 2:
                    return s
                return s[:inner_width - 5] + "..."
            top = "┌" + box_char_h * (term_width - 2) + "┐"
            bottom = "└" + box_char_h * (term_width - 2) + "┘"
            title_line = "│" + title.center(term_width - 2) + "│"
            info_lines = [
                "Paste the JSON below into your LM Studio settings (mcpServers).",
                "If an entry already exists, merge keys instead of overwriting.",
                ""
            ]
            formatted = [top, title_line, "├" + box_char_h * (term_width - 2) + "┤"]
            for l in info_lines:
                formatted.append("│" + clamp_line(l).ljust(term_width - 2) + "│")
            formatted.append("├" + box_char_h * (term_width - 2) + "┤")
            for line in txt:
                formatted.append("│" + clamp_line(line).ljust(term_width - 2) + "│")
            formatted.append(bottom)
            print("\n".join(formatted))
            print("[qtMain][mcp-config] Printed MCP config to console (dialog suppressed).")
        except Exception:
            print("[qtMain][mcp-config] Failed to print MCP config info.")
            traceback.print_exc()

    # -------------------------------------------------------------------------

    def _activate_llm_mode(self):
        self.browser.set_interactive(False)
        if not hasattr(self, "llm_preview"):
            from qtPreviewLLM import PreviewLLMView
            self.llm_preview = PreviewLLMView()
            self.right_stack.addWidget(self.llm_preview)
        self.right_stack.setCurrentWidget(self.llm_preview)

    def exit_llm_mode(self):
        self.browser.set_interactive(True)
        self.right_stack.setCurrentWidget(self.preview)

def main():
    import faulthandler; faulthandler.enable()
    app = QtWidgets.QApplication(sys.argv)
    start_path = Path("T:/")
    thumbs = start_path / "_thumbnails"
    win = MainWindow(start_path, thumbs)
    win.resize(1600, 900)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()