#!/usr/bin/env python3
"""
GridShredder with Grid/Box modes

Qt-based GUI for defining a straight grid over a plate image (e.g. 96-well agar
plate), loading metadata, and exporting per-well image crops with consistent
well IDs.

Features preserved:
- Grid mode: adjustable straight grid lines.
- Box mode: per-well boxes (implicit R×C plate), each box movable/resizable.
- Add/remove boxes in box mode, indices map to rows/cols in plate order.
- Well IDs (A01, A02, ...) drawn on top of each well in both modes.
- Left/right orientation toggle applies to both grid and box modes.

Cross-platform hardening:
- No forced Linux-only Qt backend on Windows/macOS.
- Optional RAW support imported lazily only when needed.
- pathlib-based path handling.
- PNG export uses Python byte writing for safer cross-platform filesystem behavior.
- Startup diagnostics are available in fatal errors.
"""

from __future__ import annotations

import csv
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".arw", ".dng"}
IMAGE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif "
    "*.nef *.NEF *.cr2 *.CR2 *.cr3 *.CR3 *.arw *.ARW *.dng *.DNG);;"
    "All files (*)"
)


def configure_qt_environment() -> None:
    platform = sys.platform
    if platform.startswith("win"):
        os.environ.pop("QT_QPA_PLATFORM", None)
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    elif platform == "darwin":
        os.environ.pop("QT_QPA_PLATFORM", None)
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    else:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


def runtime_diagnostics() -> str:
    return "\n".join([
        f"Python: {sys.version.split()[0]}",
        f"Platform: {sys.platform}",
        f"Qt: {QtCore.QT_VERSION_STR}",
        f"PyQt: {QtCore.PYQT_VERSION_STR}",
        f"NumPy: {np.__version__}",
        f"OpenCV: {cv2.__version__}",
    ])


class WellPlateConfig:
    def __init__(self, rows: int = 8, cols: int = 12, margin: float = 0.02):
        self.rows = rows
        self.cols = cols
        self.margin = margin
        y_start = margin
        y_end = 1.0 - margin
        x_start = margin
        x_end = 1.0 - margin
        self.row_lines = np.linspace(y_start, y_end, rows + 1).tolist()
        self.col_lines = np.linspace(x_start, x_end, cols + 1).tolist()

    @property
    def n_row_segments(self) -> int:
        return len(self.row_lines) - 1

    @property
    def n_col_segments(self) -> int:
        return len(self.col_lines) - 1


class BoxLayout:
    def __init__(self, rows: int = 8, cols: int = 12, margin: float = 0.02):
        self.rows = rows
        self.cols = cols
        self.margin = margin
        self.boxes = []
        self.init_regular_grid()

    def init_regular_grid(self) -> None:
        self.boxes.clear()
        y_start = self.margin
        y_end = 1.0 - self.margin
        x_start = self.margin
        x_end = 1.0 - self.margin
        row_lines = np.linspace(y_start, y_end, self.rows + 1)
        col_lines = np.linspace(x_start, x_end, self.cols + 1)
        for r in range(self.rows):
            for c in range(self.cols):
                tile_x1 = float(col_lines[c])
                tile_x2 = float(col_lines[c + 1])
                tile_y1 = float(row_lines[r])
                tile_y2 = float(row_lines[r + 1])
                tile_cx = 0.5 * (tile_x1 + tile_x2)
                tile_cy = 0.5 * (tile_y1 + tile_y2)
                tile_w = tile_x2 - tile_x1
                tile_h = tile_y2 - tile_y1
                box_w = 0.5 * tile_w
                box_h = 0.5 * tile_h
                x1 = tile_cx - 0.5 * box_w
                x2 = tile_cx + 0.5 * box_w
                y1 = tile_cy - 0.5 * box_h
                y2 = tile_cy + 0.5 * box_h
                self.boxes.append([x1, y1, x2, y2])

    def ensure_size(self) -> None:
        target = self.rows * self.cols
        if len(self.boxes) < target:
            template = self.boxes[-1] if self.boxes else [0.1, 0.1, 0.2, 0.2]
            while len(self.boxes) < target:
                self.boxes.append(list(template))
        elif len(self.boxes) > target:
            self.boxes = self.boxes[:target]


class ImageCanvas(QtWidgets.QLabel):
    grid_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.image = None
        self.qpix = None
        self.plate_config = WellPlateConfig()
        self.layout_mode = "grid"
        self.box_layout = None
        self.flip_label = False
        self.dragging_line = None
        self.drag_threshold = 10
        self.hover_line = None
        self.selected_box_idx = None
        self.dragging_box = None
        self._scaled_rect = None
        self.zoom = 1.0
        self.setStyleSheet("background-color: #202020;")

    def _load_raw_image(self, path: Path):
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError(
                "RAW image support requires the optional 'rawpy' package to be installed."
            ) from exc
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(output_bps=16, no_auto_bright=True)
        if rgb.dtype != np.uint16:
            rgb = rgb.astype(np.uint16)
        rgb8 = (rgb / 256).astype(np.uint8)
        return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)

    def load_image(self, path: str):
        image_path = Path(path)
        ext = image_path.suffix.lower()
        if ext in RAW_EXTENSIONS:
            img = self._load_raw_image(image_path)
        else:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not load image: {image_path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.image = img
        self._update_qpix_from_image()
        self._update_scaled_pixmap()
        self.grid_changed.emit()

    def _update_qpix_from_image(self):
        if self.image is None:
            self.qpix = None
            self.setPixmap(QtGui.QPixmap())
            self._scaled_rect = None
            return
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        self.qpix = QtGui.QPixmap.fromImage(qimg)

    def _update_scaled_pixmap(self):
        if self.qpix is None:
            return
        img_w = self.qpix.width()
        img_h = self.qpix.height()
        w = max(1, int(img_w * self.zoom))
        h = max(1, int(img_h * self.zoom))
        scaled = self.qpix.scaled(w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.resize(scaled.size())
        self._scaled_rect = QtCore.QRect(0, 0, scaled.width(), scaled.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap():
            self._scaled_rect = QtCore.QRect(0, 0, self.pixmap().width(), self.pixmap().height())

    def set_zoom(self, factor: float):
        self.zoom = max(0.1, min(8.0, factor))
        self._update_scaled_pixmap()
        self.grid_changed.emit()

    def zoom_in(self):
        self.set_zoom(self.zoom * 1.25)

    def zoom_out(self):
        self.set_zoom(self.zoom / 1.25)

    def _image_coords_to_screen(self, x_norm: float, y_norm: float):
        if self._scaled_rect is None:
            return None
        x = int(x_norm * self._scaled_rect.width())
        y = int(y_norm * self._scaled_rect.height())
        return QtCore.QPoint(x, y)

    def _screen_to_image_norm(self, x: int, y: int) -> Tuple[Optional[float], Optional[float]]:
        if self._scaled_rect is None:
            return None, None
        rel_x = x / self._scaled_rect.width()
        rel_y = y / self._scaled_rect.height()
        rel_x = min(max(rel_x, 0), 1)
        rel_y = min(max(rel_y, 0), 1)
        return rel_x, rel_y

    def set_plate_config(self, config: WellPlateConfig):
        self.plate_config = config
        self.update()
        self.grid_changed.emit()

    def set_layout_mode(self, mode: str, box_layout: Optional[BoxLayout] = None):
        self.layout_mode = mode
        self.box_layout = box_layout
        self.dragging_line = None
        self.dragging_box = None
        self.selected_box_idx = None
        self.update()
        self.grid_changed.emit()

    def set_flip_label(self, flip: bool):
        self.flip_label = flip
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image is None or self._scaled_rect is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 60))
        if self.pixmap():
            painter.drawPixmap(0, 0, self.pixmap())
        if self.layout_mode == "grid":
            self._paint_grid(painter)
            self._paint_well_labels_grid(painter)
        else:
            self._paint_boxes(painter)
            self._paint_well_labels_boxes(painter)
        painter.end()

    def _paint_grid(self, painter: QtGui.QPainter):
        base_pen = QtGui.QPen(QtCore.Qt.green, 1, QtCore.Qt.SolidLine)
        painter.setPen(base_pen)
        for idx, x_norm in enumerate(self.plate_config.col_lines):
            p1 = self._image_coords_to_screen(x_norm, 0)
            p2 = self._image_coords_to_screen(x_norm, 1)
            if p1 and p2:
                if self.hover_line == ("col", idx):
                    painter.setPen(QtGui.QPen(QtCore.Qt.yellow, 2))
                    painter.drawLine(p1, p2)
                    painter.setPen(base_pen)
                else:
                    painter.drawLine(p1, p2)
        for idx, y_norm in enumerate(self.plate_config.row_lines):
            p1 = self._image_coords_to_screen(0, y_norm)
            p2 = self._image_coords_to_screen(1, y_norm)
            if p1 and p2:
                if self.hover_line == ("row", idx):
                    painter.setPen(QtGui.QPen(QtCore.Qt.yellow, 2))
                    painter.drawLine(p1, p2)
                    painter.setPen(base_pen)
                else:
                    painter.drawLine(p1, p2)

    def _paint_boxes(self, painter: QtGui.QPainter):
        if not self.box_layout:
            return
        base_pen = QtGui.QPen(QtCore.Qt.green, 1, QtCore.Qt.SolidLine)
        sel_pen = QtGui.QPen(QtCore.Qt.yellow, 2, QtCore.Qt.SolidLine)
        for idx, (x1, y1, x2, y2) in enumerate(self.box_layout.boxes):
            p1 = self._image_coords_to_screen(x1, y1)
            p2 = self._image_coords_to_screen(x2, y2)
            if not p1 or not p2:
                continue
            rect = QtCore.QRect(
                min(p1.x(), p2.x()),
                min(p1.y(), p2.y()),
                abs(p2.x() - p1.x()),
                abs(p2.y() - p1.y()),
            )
            painter.setPen(sel_pen if idx == self.selected_box_idx else base_pen)
            painter.drawRect(rect)

    def _paint_well_labels_grid(self, painter: QtGui.QPainter):
        rows = self.plate_config.n_row_segments
        cols = self.plate_config.n_col_segments
        if rows <= 0 or cols <= 0:
            return
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        for r in range(rows):
            for c in range(cols):
                y1 = self.plate_config.row_lines[r]
                y2 = self.plate_config.row_lines[r + 1]
                x1 = self.plate_config.col_lines[c]
                x2 = self.plate_config.col_lines[c + 1]
                cx_norm = 0.5 * (x1 + x2)
                cy_norm = 0.5 * (y1 + y2)
                center = self._image_coords_to_screen(cx_norm, cy_norm)
                if not center:
                    continue
                row_letter = chr(ord("A") + r)
                num = c + 1 if not self.flip_label else cols - c
                well_id = f"{row_letter}{num:02d}"
                painter.drawText(center, well_id)

    def _paint_well_labels_boxes(self, painter: QtGui.QPainter):
        if not self.box_layout:
            return
        rows = self.box_layout.rows
        cols = self.box_layout.cols
        self.box_layout.ensure_size()
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx >= len(self.box_layout.boxes):
                    continue
                x1, y1, x2, y2 = self.box_layout.boxes[idx]
                cx_norm = 0.5 * (x1 + x2)
                cy_norm = 0.5 * (y1 + y2)
                center = self._image_coords_to_screen(cx_norm, cy_norm)
                if not center:
                    continue
                row_letter = chr(ord("A") + r)
                num = c + 1 if not self.flip_label else cols - c
                well_id = f"{row_letter}{num:02d}"
                painter.drawText(center, well_id)

    def mousePressEvent(self, event):
        if self.image is None or self._scaled_rect is None:
            return
        if self.layout_mode == "grid":
            self._mouse_press_grid(event)
        else:
            self._mouse_press_boxes(event)
        super().mousePressEvent(event)

    def _mouse_press_grid(self, event):
        pos = event.pos()
        x = pos.x()
        y = pos.y()
        nearest = None
        nearest_dist = self.drag_threshold + 1
        for idx, x_norm in enumerate(self.plate_config.col_lines):
            p = self._image_coords_to_screen(x_norm, 0.5)
            if p:
                dist = abs(p.x() - x)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = ("col", idx)
        for idx, y_norm in enumerate(self.plate_config.row_lines):
            p = self._image_coords_to_screen(0.5, y_norm)
            if p:
                dist = abs(p.y() - y)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = ("row", idx)
        self.dragging_line = nearest if nearest_dist <= self.drag_threshold else None

    def _mouse_press_boxes(self, event):
        if not self.box_layout:
            return
        pos = event.pos()
        x_norm, y_norm = self._screen_to_image_norm(pos.x(), pos.y())
        if x_norm is None:
            return
        hit_idx = None
        for idx, (x1, y1, x2, y2) in enumerate(self.box_layout.boxes):
            if x1 <= x_norm <= x2 and y1 <= y_norm <= y2:
                hit_idx = idx
                break
        self.selected_box_idx = hit_idx
        self.dragging_box = None
        if hit_idx is not None:
            x1, y1, x2, y2 = self.box_layout.boxes[hit_idx]
            margin = 0.01
            near_left = abs(x_norm - x1) < margin
            near_right = abs(x_norm - x2) < margin
            near_top = abs(y_norm - y1) < margin
            near_bottom = abs(y_norm - y2) < margin
            edge = "move"
            if near_left and near_top:
                edge = "topleft"
            elif near_right and near_top:
                edge = "topright"
            elif near_left and near_bottom:
                edge = "bottomleft"
            elif near_right and near_bottom:
                edge = "bottomright"
            elif near_left:
                edge = "left"
            elif near_right:
                edge = "right"
            elif near_top:
                edge = "top"
            elif near_bottom:
                edge = "bottom"
            self.dragging_box = (hit_idx, edge)
        self.update()

    def mouseMoveEvent(self, event):
        if self._scaled_rect is None:
            super().mouseMoveEvent(event)
            return
        if self.layout_mode == "grid":
            self._mouse_move_grid(event)
        else:
            self._mouse_move_boxes(event)
        super().mouseMoveEvent(event)

    def _mouse_move_grid(self, event):
        x = event.pos().x()
        y = event.pos().y()
        if self.dragging_line is not None:
            x_norm, y_norm = self._screen_to_image_norm(x, y)
            kind, idx = self.dragging_line
            if kind == "col":
                if 0 <= idx < len(self.plate_config.col_lines):
                    neighbors = self.plate_config.col_lines
                    low = neighbors[idx - 1] + 0.01 if idx > 0 else 0.0
                    high = neighbors[idx + 1] - 0.01 if idx < len(neighbors) - 1 else 1.0
                    neighbors[idx] = min(max(x_norm, low), high)
            else:
                if 0 <= idx < len(self.plate_config.row_lines):
                    neighbors = self.plate_config.row_lines
                    low = neighbors[idx - 1] + 0.01 if idx > 0 else 0.0
                    high = neighbors[idx + 1] - 0.01 if idx < len(neighbors) - 1 else 1.0
                    neighbors[idx] = min(max(y_norm, low), high)
            self.update()
            self.grid_changed.emit()
            return
        nearest = None
        nearest_dist = self.drag_threshold + 1
        for idx, x_norm in enumerate(self.plate_config.col_lines):
            p = self._image_coords_to_screen(x_norm, 0.5)
            if p:
                dist = abs(p.x() - x)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = ("col", idx)
        for idx, y_norm in enumerate(self.plate_config.row_lines):
            p = self._image_coords_to_screen(0.5, y_norm)
            if p:
                dist = abs(p.y() - y)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = ("row", idx)
        self.hover_line = nearest if nearest_dist <= self.drag_threshold else None
        self.update()

    def _mouse_move_boxes(self, event):
        if not self.box_layout or self.dragging_box is None:
            return
        x_norm, y_norm = self._screen_to_image_norm(event.pos().x(), event.pos().y())
        if x_norm is None:
            return
        idx, edge = self.dragging_box
        if not (0 <= idx < len(self.box_layout.boxes)):
            return
        x1, y1, x2, y2 = self.box_layout.boxes[idx]
        if edge == "move":
            w = x2 - x1
            h = y2 - y1
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            dx = x_norm - cx
            dy = y_norm - cy
            nx1 = min(max(x1 + dx, 0.0), 1.0 - w)
            ny1 = min(max(y1 + dy, 0.0), 1.0 - h)
            nx2 = nx1 + w
            ny2 = ny1 + h
            self.box_layout.boxes[idx] = [nx1, ny1, nx2, ny2]
        else:
            if "left" in edge:
                x1 = min(max(x_norm, 0.0), x2 - 0.01)
            if "right" in edge:
                x2 = max(min(x_norm, 1.0), x1 + 0.01)
            if "top" in edge:
                y1 = min(max(y_norm, 0.0), y2 - 0.01)
            if "bottom" in edge:
                y2 = max(min(y_norm, 1.0), y1 + 0.01)
            self.box_layout.boxes[idx] = [x1, y1, x2, y2]
        self.update()
        self.grid_changed.emit()

    def mouseReleaseEvent(self, event):
        self.dragging_line = None
        self.dragging_box = None
        super().mouseReleaseEvent(event)

    def get_well_roi(self, row: int, col: int, require_enclosed: bool = True):
        if self.image is None:
            return None
        h, w, _ = self.image.shape
        x1 = int(round(self.plate_config.col_lines[col] * w))
        x2 = int(round(self.plate_config.col_lines[col + 1] * w))
        y1 = int(round(self.plate_config.row_lines[row] * h))
        y2 = int(round(self.plate_config.row_lines[row + 1] * h))
        x1_clamped = max(0, min(w, x1))
        x2_clamped = max(0, min(w, x2))
        y1_clamped = max(0, min(h, y1))
        y2_clamped = max(0, min(h, y2))
        if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
            return None
        if require_enclosed and (x1_clamped == 0 or y1_clamped == 0 or x2_clamped == w or y2_clamped == h):
            return None
        return self.image[y1_clamped:y2_clamped, x1_clamped:x2_clamped].copy()

    def get_box_roi(self, box_idx: int, require_enclosed: bool = True):
        if self.image is None or not self.box_layout:
            return None
        if not (0 <= box_idx < len(self.box_layout.boxes)):
            return None
        h, w, _ = self.image.shape
        x1n, y1n, x2n, y2n = self.box_layout.boxes[box_idx]
        x1 = int(round(x1n * w))
        x2 = int(round(x2n * w))
        y1 = int(round(y1n * h))
        y2 = int(round(y2n * h))
        x1_clamped = max(0, min(w, x1))
        x2_clamped = max(0, min(w, x2))
        y1_clamped = max(0, min(h, y1))
        y2_clamped = max(0, min(h, y2))
        if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
            return None
        if require_enclosed and (x1_clamped == 0 or y1_clamped == 0 or x2_clamped == w or y2_clamped == h):
            return None
        return self.image[y1_clamped:y2_clamped, x1_clamped:x2_clamped].copy()


class MetadataDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load metadata CSV")
        self.setModal(True)
        self.resize(500, 400)
        self.csv_path_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse_csv)
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Well ID", "Species"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        load_btn = QtWidgets.QPushButton("Load")
        load_btn.clicked.connect(self.load_csv)
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        path_layout = QtWidgets.QHBoxLayout()
        path_layout.addWidget(self.csv_path_edit)
        path_layout.addWidget(browse_btn)
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(path_layout)
        layout.addWidget(self.table)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.metadata = {}

    def browse_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select metadata CSV", "", "CSV files (*.csv);;All files (*)")
        if path:
            self.csv_path_edit.setText(path)

    def load_csv(self):
        path = Path(self.csv_path_edit.text().strip())
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "Error", "CSV path invalid.")
            return
        meta = {}
        try:
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    well_id = row.get("well_id") or row.get("WellID") or row.get("well") or row.get("Well")
                    species = row.get("species") or row.get("Species") or row.get("strain") or row.get("Strain")
                    if not well_id or not species:
                        continue
                    meta[well_id.strip()] = species.strip()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to read CSV:\n{e}")
            return
        if not meta:
            QtWidgets.QMessageBox.warning(
                self,
                "No metadata",
                "No valid well_id/species rows were found.\nCheck column names (e.g. well_id, species).",
            )
            return
        self.metadata = meta
        self.populate_table()

    def populate_table(self):
        self.table.setRowCount(len(self.metadata))
        for row_idx, (well_id, species) in enumerate(self.metadata.items()):
            self.table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(well_id))
            self.table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(species))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GridShredder")
        self.resize(1400, 900)
        self.canvas = ImageCanvas()
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(False)
        self.setCentralWidget(self.scroll_area)
        self.plate_config = self.canvas.plate_config
        self.metadata = {}
        self.media_name = "media"
        self.flip_label = False
        self.layout_mode = "grid"
        self.box_layout = None
        self._added_vertical_indices = []
        self._added_horizontal_indices = []
        self._create_toolbar()
        self._create_statusbar()
        self.canvas.grid_changed.connect(self._on_grid_changed)
        self._apply_dark_palette()

    def _apply_dark_palette(self):
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#303030"))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#202020"))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#252525"))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#F0F0F0"))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#F0F0F0"))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#404040"))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#F0F0F0"))
        self.setPalette(palette)
        self.setStyleSheet(
            """
            QToolBar { spacing: 6px; padding: 4px; background-color: #303030; }
            QToolButton { padding: 4px 8px; border: 1px solid #101010; border-radius: 3px; color: #F0F0F0; background-color: #0097A7; }
            QToolButton:hover { background-color: #00BCD4; }
            QToolButton:pressed { background-color: #006978; padding-top: 5px; padding-left: 9px; }
            QStatusBar { background-color: #303030; color: #F0F0F0; }
            QTableWidget { gridline-color: #555555; }
            """
        )

    def _create_toolbar(self):
        tb = QtWidgets.QToolBar("Main")
        tb.setIconSize(QtCore.QSize(20, 20))
        self.addToolBar(tb)
        open_action = QtWidgets.QAction("Open image", self)
        open_action.setToolTip("Open plate image")
        open_action.triggered.connect(self.open_image)
        tb.addAction(open_action)
        meta_action = QtWidgets.QAction("Load metadata", self)
        meta_action.setToolTip("Load CSV with well_id → species mapping")
        meta_action.triggered.connect(self.load_metadata)
        tb.addAction(meta_action)
        media_action = QtWidgets.QAction("Set media", self)
        media_action.setToolTip("Set media label used in filenames")
        media_action.triggered.connect(self.set_media)
        tb.addAction(media_action)
        export_action = QtWidgets.QAction("Export wells", self)
        export_action.setToolTip("Export wells/boxes as individual PNGs using current layout")
        export_action.triggered.connect(self.export_wells)
        tb.addAction(export_action)
        tb.addSeparator()
        grid_action = QtWidgets.QAction("Reset grid", self)
        grid_action.setToolTip("Reset grid to default 8×12")
        grid_action.triggered.connect(self.reset_grid)
        tb.addAction(grid_action)
        set_size_action = QtWidgets.QAction("Set plate size", self)
        set_size_action.setToolTip("Set plate rows x columns (e.g. 8x12, 16x24)")
        set_size_action.triggered.connect(self.set_plate_size)
        tb.addAction(set_size_action)
        add_col_action = QtWidgets.QAction("Add vertical", self)
        add_col_action.setToolTip("Insert extra vertical gridline (grid mode)")
        add_col_action.triggered.connect(self.add_vertical_line)
        tb.addAction(add_col_action)
        add_row_action = QtWidgets.QAction("Add horizontal", self)
        add_row_action.setToolTip("Insert extra horizontal gridline (grid mode)")
        add_row_action.triggered.connect(self.add_horizontal_line)
        tb.addAction(add_row_action)
        remove_lines_action = QtWidgets.QAction("Remove last lines", self)
        remove_lines_action.setToolTip("Remove the most recently added vertical and horizontal gridlines")
        remove_lines_action.triggered.connect(self.remove_last_added_lines)
        tb.addAction(remove_lines_action)
        tb.addSeparator()
        zoom_in_action = QtWidgets.QAction("+", self)
        zoom_in_action.setToolTip("Zoom in")
        zoom_in_action.triggered.connect(self.canvas.zoom_in)
        tb.addAction(zoom_in_action)
        zoom_out_action = QtWidgets.QAction("−", self)
        zoom_out_action.setToolTip("Zoom out")
        zoom_out_action.triggered.connect(self.canvas.zoom_out)
        tb.addAction(zoom_out_action)
        tb.addSeparator()
        self.flip_label_action = QtWidgets.QAction("Left→Right labels", self)
        self.flip_label_action.setCheckable(True)
        self.flip_label_action.setChecked(False)
        self.flip_label_action.setToolTip(
            "Toggle column numbering orientation for well IDs.\n"
            "Unchecked: A01 at left, A12 at right.\n"
            "Checked: A01 at right, A12 at left."
        )
        self.flip_label_action.toggled.connect(self.set_flip_labels)
        tb.addAction(self.flip_label_action)
        tb.addSeparator()
        self.mode_action = QtWidgets.QAction("Mode: Grid", self)
        self.mode_action.setCheckable(True)
        self.mode_action.setChecked(False)
        self.mode_action.setToolTip("Toggle between Grid mode (unchecked) and Box mode (checked).")
        self.mode_action.toggled.connect(self.toggle_mode)
        tb.addAction(self.mode_action)
        self.add_box_action = QtWidgets.QAction("Add box", self)
        self.add_box_action.setToolTip("Add a new box (box mode)")
        self.add_box_action.triggered.connect(self.add_box)
        tb.addAction(self.add_box_action)
        self.remove_box_action = QtWidgets.QAction("Remove box", self)
        self.remove_box_action.setToolTip("Remove selected box (box mode)")
        self.remove_box_action.triggered.connect(self.remove_box)
        tb.addAction(self.remove_box_action)
        self._update_mode_actions_enabled()

    def _create_statusbar(self):
        self.statusBar().showMessage("Ready")

    def _on_grid_changed(self):
        if self.layout_mode == "grid":
            rows = self.canvas.plate_config.n_row_segments
            cols = self.canvas.plate_config.n_col_segments
        else:
            if self.box_layout:
                rows = self.box_layout.rows
                cols = self.box_layout.cols
            else:
                rows = cols = 0
        self.statusBar().showMessage(f"Layout updated: {rows} rows × {cols} columns")

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open plate image", "", IMAGE_FILTER)
        if not path:
            return
        try:
            self.canvas.load_image(path)
            self.statusBar().showMessage(f"Loaded image: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    def load_metadata(self):
        dlg = MetadataDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.metadata = dlg.metadata
            self.statusBar().showMessage(f"Loaded metadata for {len(self.metadata)} wells")

    def set_media(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Set media name", "Media label (e.g. YPD, PDA):", text=self.media_name)
        if ok and text.strip():
            self.media_name = text.strip()
            self.statusBar().showMessage(f"Media set to: {self.media_name}")

    def reset_grid(self):
        self.plate_config = WellPlateConfig(rows=8, cols=12, margin=0.02)
        self.canvas.set_plate_config(self.plate_config)
        self._added_vertical_indices.clear()
        self._added_horizontal_indices.clear()
        self.statusBar().showMessage("Grid reset to default 8×12.")

    def set_plate_size(self):
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Set plate size",
            "Rows x Columns (e.g. 8x12, 16x24):",
            text=f"{self.plate_config.rows}x{self.plate_config.cols}",
        )
        if not ok or not text.strip():
            return
        try:
            parts = text.lower().replace(" ", "").split("x")
            if len(parts) != 2:
                raise ValueError
            rows = int(parts[0])
            cols = int(parts[1])
            if rows < 1 or cols < 1:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid size", "Please enter like 8x12 or 16x24 with positive integers.")
            return
        self.plate_config = WellPlateConfig(rows=rows, cols=cols, margin=0.02)
        self.canvas.set_plate_config(self.plate_config)
        if self.box_layout is None:
            self.box_layout = BoxLayout(rows=rows, cols=cols, margin=0.02)
        else:
            self.box_layout.rows = rows
            self.box_layout.cols = cols
            self.box_layout.ensure_size()
        self._added_vertical_indices.clear()
        self._added_horizontal_indices.clear()
        self.statusBar().showMessage(f"Plate size set to {rows}×{cols}.")

    def add_vertical_line(self):
        if self.layout_mode != "grid":
            return
        lines = self.plate_config.col_lines
        if len(lines) < 2:
            return
        widths = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
        idx = int(np.argmax(widths))
        new_x = (lines[idx] + lines[idx + 1]) / 2.0
        insert_pos = idx + 1
        lines.insert(insert_pos, new_x)
        self._added_vertical_indices.append(insert_pos)
        self.canvas.update()
        self.canvas.grid_changed.emit()
        self.statusBar().showMessage("Inserted extra vertical gridline.")

    def add_horizontal_line(self):
        if self.layout_mode != "grid":
            return
        lines = self.plate_config.row_lines
        if len(lines) < 2:
            return
        heights = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
        idx = int(np.argmax(heights))
        new_y = (lines[idx] + lines[idx + 1]) / 2.0
        insert_pos = idx + 1
        lines.insert(insert_pos, new_y)
        self._added_horizontal_indices.append(insert_pos)
        self.canvas.update()
        self.canvas.grid_changed.emit()
        self.statusBar().showMessage("Inserted extra horizontal gridline.")

    def remove_last_added_lines(self):
        if self.layout_mode != "grid":
            return
        removed_any = False
        if self._added_vertical_indices:
            idx = self._added_vertical_indices.pop()
            if 0 < idx < len(self.plate_config.col_lines) - 1:
                self.plate_config.col_lines.pop(idx)
                removed_any = True
        if self._added_horizontal_indices:
            idx = self._added_horizontal_indices.pop()
            if 0 < idx < len(self.plate_config.row_lines) - 1:
                self.plate_config.row_lines.pop(idx)
                removed_any = True
        if removed_any:
            self.canvas.update()
            self.canvas.grid_changed.emit()
            self.statusBar().showMessage("Removed last added gridlines (if any).")
        else:
            self.statusBar().showMessage("No added gridlines left to remove.")

    def toggle_mode(self, checked: bool):
        if checked:
            self.layout_mode = "box"
            self.mode_action.setText("Mode: Box")
            rows = self.plate_config.rows
            cols = self.plate_config.cols
            if self.box_layout is None:
                self.box_layout = BoxLayout(rows=rows, cols=cols, margin=0.02)
            else:
                self.box_layout.rows = rows
                self.box_layout.cols = cols
                self.box_layout.ensure_size()
            self.canvas.set_layout_mode("box", self.box_layout)
            self.statusBar().showMessage("Switched to Box mode.")
        else:
            self.layout_mode = "grid"
            self.mode_action.setText("Mode: Grid")
            self.canvas.set_layout_mode("grid", None)
            self.statusBar().showMessage("Switched to Grid mode.")
        self._update_mode_actions_enabled()
        self.canvas.set_flip_label(self.flip_label)

    def _update_mode_actions_enabled(self):
        grid_only = self.layout_mode == "grid"
        box_only = self.layout_mode == "box"
        for act in self.findChildren(QtWidgets.QAction):
            if act.text() in ("Add vertical", "Add horizontal", "Remove last lines"):
                act.setEnabled(grid_only)
        self.add_box_action.setEnabled(box_only)
        self.remove_box_action.setEnabled(box_only)

    def add_box(self):
        if self.layout_mode != "box" or self.box_layout is None:
            return
        if self.box_layout.boxes:
            x1, y1, x2, y2 = self.box_layout.boxes[-1]
            box = [x1, y1, x2, y2]
        else:
            box = [0.4, 0.4, 0.6, 0.6]
        self.box_layout.boxes.append(box)
        self.box_layout.ensure_size()
        self.canvas.set_layout_mode("box", self.box_layout)
        self.statusBar().showMessage("Added a new box.")

    def remove_box(self):
        if self.layout_mode != "box" or self.box_layout is None:
            return
        idx = self.canvas.selected_box_idx
        if idx is None or not (0 <= idx < len(self.box_layout.boxes)):
            self.statusBar().showMessage("No box selected to remove.")
            return
        self.box_layout.boxes.pop(idx)
        self.box_layout.ensure_size()
        self.canvas.selected_box_idx = None
        self.canvas.set_layout_mode("box", self.box_layout)
        self.statusBar().showMessage("Removed selected box.")

    def set_flip_labels(self, checked: bool):
        self.flip_label = checked
        if self.flip_label:
            self.flip_label_action.setText("Right→Left labels (ON)")
            self.statusBar().showMessage("Right→Left label orientation active (A01 at right).")
        else:
            self.flip_label_action.setText("Left→Right labels (ON)")
            self.statusBar().showMessage("Left→Right label orientation active (A01 at left).")
        self.canvas.set_flip_label(self.flip_label)

    def well_id_for(self, row: int, col: int, n_cols: int) -> str:
        row_letter = chr(ord("A") + row)
        num = col + 1 if not self.flip_label else n_cols - col
        return f"{row_letter}{num:02d}"

    def _write_png(self, file_path: Path, roi) -> bool:
        ok, buf = cv2.imencode(".png", roi)
        if not ok:
            return False
        file_path.write_bytes(buf.tobytes())
        return True

    def export_wells(self):
        if self.canvas.image is None:
            QtWidgets.QMessageBox.warning(self, "No image", "Load an image first.")
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm export",
            "Export wells using current layout?\nOnly wells/boxes fully enclosed (not touching image edges)\nwill be exported.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
        if not out_dir:
            return
        out_path = Path(out_dir)
        export_count = 0
        skipped_open = 0
        if self.layout_mode == "grid":
            rows = self.canvas.plate_config.n_row_segments
            cols = self.canvas.plate_config.n_col_segments
            for r in range(rows):
                for c in range(cols):
                    roi = self.canvas.get_well_roi(r, c, require_enclosed=True)
                    if roi is None:
                        skipped_open += 1
                        continue
                    well_id = self.well_id_for(r, c, cols)
                    species = self.metadata.get(well_id, "unknown")
                    prefix = f"{species}_{self.media_name}_{well_id}"
                    if self._write_png(out_path / f"{prefix}.png", roi):
                        export_count += 1
        else:
            if not self.box_layout:
                QtWidgets.QMessageBox.warning(self, "No boxes", "No box layout is defined.")
                return
            rows = self.box_layout.rows
            cols = self.box_layout.cols
            self.box_layout.ensure_size()
            for idx in range(rows * cols):
                roi = self.canvas.get_box_roi(idx, require_enclosed=True)
                if roi is None:
                    skipped_open += 1
                    continue
                r = idx // cols
                c = idx % cols
                well_id = self.well_id_for(r, c, cols)
                species = self.metadata.get(well_id, "unknown")
                prefix = f"{species}_{self.media_name}_{well_id}"
                if self._write_png(out_path / f"{prefix}.png", roi):
                    export_count += 1
        self.statusBar().showMessage(
            f"Exported {export_count} wells to {out_path} (skipped {skipped_open} non-enclosed wells)."
        )

    def keyPressEvent(self, event):
        key = event.key()
        step = 40
        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()
        if key == QtCore.Qt.Key_Left:
            hbar.setValue(hbar.value() - step)
        elif key == QtCore.Qt.Key_Right:
            hbar.setValue(hbar.value() + step)
        elif key == QtCore.Qt.Key_Up:
            vbar.setValue(vbar.value() - step)
        elif key == QtCore.Qt.Key_Down:
            vbar.setValue(vbar.value() + step)
        else:
            super().keyPressEvent(event)


def main():
    configure_qt_environment()
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error in main():", e)
        print(runtime_diagnostics())
        traceback.print_exc()
