from __future__ import annotations
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv'}


class PreviewView(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._path: Optional[Path] = None
        self._orig_pixmap: Optional[QtGui.QPixmap] = None

        # Create a stacked widget to switch between image and video preview.
        self._stack = QtWidgets.QStackedWidget(self)

        # Image preview: QLabel
        self._label = QtWidgets.QLabel("No preview")
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label.setBackgroundRole(QtGui.QPalette.ColorRole.Base)
        self._label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self._label.setScaledContents(False)
        self._stack.addWidget(self._label)

        # Video preview: QVideoWidget with QMediaPlayer
        self._video_widget = QVideoWidget()
        self._stack.addWidget(self._video_widget)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._stack, 1)

        # Initialize media player and audio output for video playback.
        self._media_player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._media_player.setAudioOutput(self._audio_output)
        self._media_player.setVideoOutput(self._video_widget)

    def set_path(self, path: Optional[Path]):
        # Stop any current video playback.
        if self._media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._media_player.stop()

        self._path = path
        self._orig_pixmap = None

        if not path or not path.exists() or not path.is_file():
            self._label.setText("No preview")
            self._label.setPixmap(QtGui.QPixmap())
            self._stack.setCurrentWidget(self._label)
            return

        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            pix = QtGui.QPixmap(str(path))
            if pix.isNull():
                self._label.setText("Failed to load")
                self._label.setPixmap(QtGui.QPixmap())
            else:
                self._orig_pixmap = pix
                self._update_scaled()
            self._stack.setCurrentWidget(self._label)
        elif suffix in VIDEO_SUFFIXES:
            self._stack.setCurrentWidget(self._video_widget)
            video_url = QUrl.fromLocalFile(str(path))
            self._media_player.setSource(video_url)
            self._media_player.play()
        else:
            self._label.setText("Unsupported preview")
            self._label.setPixmap(QtGui.QPixmap())
            self._stack.setCurrentWidget(self._label)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self):
        if not self._orig_pixmap or self._orig_pixmap.isNull():
            return
        target = self._label.size()
        scaled = self._orig_pixmap.scaled(
            target,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self._label.setPixmap(scaled)
        self._label.setText("")