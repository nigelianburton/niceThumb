from PyQt6 import QtCore, QtWidgets, QtGui
import urllib.request
import json
import os
import sys
import argparse

_DEFAULT_API_BASE = "http://127.0.0.1:5015"

def _resolve_api_base(explicit: str | None) -> str:
    """
    Resolve API base priority:
      1. --api-base argument (explicit)
      2. QTD_LLM_API environment variable
      3. QTD_SERVER environment variable (used by main app)
      4. Default localhost:5015
    """
    if explicit:
        return explicit.rstrip("/")
    env1 = os.environ.get("QTD_LLM_API", "").strip().rstrip("/")
    if env1:
        return env1
    env2 = os.environ.get("QTD_SERVER", "").strip().rstrip("/")
    if env2:
        if "://" not in env2:
            env2 = "http://" + env2
        return env2
    return _DEFAULT_API_BASE

class PreviewLLMView(QtWidgets.QWidget):
    def __init__(self, parent=None, api_base: str | None = None):
        super().__init__(parent)
        self._api_base = _resolve_api_base(api_base)
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self._poll_llm_requests)
        self._poll_timer.start()

        self.layout = QtWidgets.QVBoxLayout(self)
        self.toolbar = self._build_toolbar()
        self.layout.addWidget(self.toolbar)

        # Status label (shows API base + last update)
        self.status_label = QtWidgets.QLabel(f"API: {self._api_base}")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.layout.addWidget(self.status_label)

        self.list_widget = QtWidgets.QListWidget()
        self.layout.addWidget(self.list_widget, 1)

    def _build_toolbar(self):
        # Placeholder toolbar (extend with controls if needed)
        bar = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)

        refresh_btn = QtWidgets.QPushButton("Refresh Now")
        refresh_btn.clicked.connect(self._poll_llm_requests)
        lay.addWidget(refresh_btn)

        base_edit = QtWidgets.QLineEdit(self._api_base)
        base_edit.setPlaceholderText("API Base URL")
        base_edit.setMinimumWidth(240)
        def _apply_new_base():
            txt = base_edit.text().strip().rstrip("/")
            if txt and txt != self._api_base:
                if "://" not in txt:
                    txt = "http://" + txt
                self._api_base = txt
                self.status_label.setText(f"API: {self._api_base} (updated)")
        base_edit.editingFinished.connect(_apply_new_base)
        lay.addWidget(base_edit)

        lay.addStretch(1)
        return bar

    def _poll_llm_requests(self):
        try:
            url = f"{self._api_base}/llm_requests"
            with urllib.request.urlopen(url, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            self._update_list(data.get("requests", []))
            self.status_label.setText(f"API: {self._api_base} | {len(data.get('requests', []))} items")
        except Exception:
            # Silent failure acceptable for periodic polling (UI remains responsive)
            self.status_label.setText(f"API: {self._api_base} (unreachable)")

    def _update_list(self, requests):
        self.list_widget.clear()
        for req in requests:
            ts = req.get("timestamp", "?")
            desc = req.get("description", "(no description)")
            item = QtWidgets.QListWidgetItem(f"{ts} - {desc}")
            self.list_widget.addItem(item)


# ---------------- Standalone Execution Support ----------------

def main():
    parser = argparse.ArgumentParser(description="Standalone PreviewLLM panel")
    parser.add_argument("--api-base", help="Override API base (e.g. http://127.0.0.1:5015)", default=None)
    parser.add_argument("--interval", type=int, default=2000, help="Polling interval ms (default 2000)")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Preview LLM (Standalone)")
    view = PreviewLLMView(api_base=args.api_base)
    if args.interval > 0:
        view._poll_timer.setInterval(args.interval)
    win.setCentralWidget(view)
    win.resize(600, 500)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()