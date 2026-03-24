"""
GridShredder

GridShredder is a Qt-based GUI for interactively defining a grid over
a plate image (e.g. rectangular agar plate), adjusting grid lines, loading
metadata, and exporting per‑well image crops with consistent well IDs.

Major components:
- Environment setup and imports.
- WellPlateConfig: normalized grid line positions and geometry.
- ImageCanvas: QLabel-based widget to display the plate and interactive grid.
- MetadataDialog: dialog to load CSV metadata mapping well_id -> species.
- MainWindow: main application window with toolbars and export logic.
- main(): Qt application bootstrap and entry point.
"""

import sys
import os
import csv

import numpy as np
import cv2
import rawpy
from PyQt5 import QtCore, QtGui, QtWidgets

# ------------------------------------------------------------------
# Environment fixes to avoid OpenCV Qt plugin conflicts
# ------------------------------------------------------------------
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------
class WellPlateConfig:
    """
    Stores grid definition in normalized coordinates.

    Attributes
    ----------
    rows : int
        Logical rows (e.g. 8 for a 96‑well plate).
    cols : int
        Logical columns (e.g. 12 for a 96‑well plate).
    row_lines : list[float]
        Sorted normalized horizontal boundaries in [0, 1], length rows+1.
    col_lines : list[float]
        Sorted normalized vertical boundaries in [0, 1], length cols+1.
    """

    def __init__(self, rows=8, cols=12):
        self.rows = rows
        self.cols = cols
        self.row_lines = np.linspace(0, 1, rows + 1).tolist()
        self.col_lines = np.linspace(0, 1, cols + 1).tolist()

    @property
    def n_row_segments(self):
        """Current number of row segments (can exceed rows if lines added)."""
        return len(self.row_lines) - 1

    @property
    def n_col_segments(self):
        """Current number of column segments (can exceed cols if lines added)."""
        return len(self.col_lines) - 1


# ------------------------------------------------------------------
# Image canvas with grid
# ------------------------------------------------------------------
class ImageCanvas(QtWidgets.QLabel):
    """
    QLabel-based canvas that shows the plate image and interactive grid lines.

    Responsibilities
    ----------------
    - Manage loading and displaying images without quality loss.
    - Keep track of normalized grid line coordinates.
    - Draw grid lines over the image.
    - Handle mouse interaction for dragging grid lines to refine well borders.
    """

    grid_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(QtCore.Qt.AlignCenter)
        # Do NOT let the pixmap dictate the widget size
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                           QtWidgets.QSizePolicy.Ignored)

        # Core image state
        self.image = None        # OpenCV BGR image
        self.qpix = None         # QPixmap for Qt drawing
        self.plate_config = WellPlateConfig()

        # Mouse interaction for dragging lines
        self.dragging_line = None   # ('row', idx) or ('col', idx)
        self.drag_threshold = 5     # pixels

        # Geometry and zoom
        self._scaled_rect = None    # QRect of image inside label
        self.zoom = 1.0             # zoom factor

        # Visual background
        self.setStyleSheet("background-color: #202020;")

    # ---------------- Image I/O with no quality loss ----------------
    def load_image(self, path: str):
        """
        Load image from disk.

        For RAW (e.g. NEF, CR2, ARW, DNG) uses rawpy for high-quality decoding.
        For standard formats uses cv2.imread.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in [".nef", ".cr2", ".cr3", ".arw", ".dng"]:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(output_bps=16, no_auto_bright=True)
                if rgb.dtype != np.uint16:
                    rgb = rgb.astype(np.uint16)
                rgb8 = (rgb / 256).astype(np.uint8)
                img = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)

        if img is None:
            raise RuntimeError(f"Could not load image: {path}")

        # Ensure 3-channel BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        self.image = img
        self._update_qpix_from_image()
        self._update_scaled_pixmap()
        self.grid_changed.emit()

    def _update_qpix_from_image(self):
        """Convert the internal OpenCV BGR image to a QPixmap for display."""
        if self.image is None:
            self.qpix = None
            self.setPixmap(QtGui.QPixmap())
            self._scaled_rect = None
            return

        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(
            rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        # Deep copy so backing buffer remains valid
        qimg = qimg.copy()
        self.qpix = QtGui.QPixmap.fromImage(qimg)

    def _update_scaled_pixmap(self):
        """
        Compute and set the scaled pixmap according to current zoom and
        the available widget size, maintaining aspect ratio, without
        forcing the window to resize.
        """
        if self.qpix is None:
            return

        # Base on *image* size
        img_w = self.qpix.width()
        img_h = self.qpix.height()

        # Desired size based on zoom
        desired_w = int(img_w * self.zoom)
        desired_h = int(img_h * self.zoom)

        # Available drawing area (current label size)
        avail_w = max(1, self.width())
        avail_h = max(1, self.height())

        # Safety cap to avoid absurdly huge pixmaps if zoom is large
        max_factor = 4
        w = min(desired_w, avail_w * max_factor)
        h = min(desired_h, avail_h * max_factor)

        w = max(1, w)
        h = max(1, h)

        scaled = self.qpix.scaled(
            w, h,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )

        # Center in current label rect
        x_off = (self.width() - scaled.width()) // 2
        y_off = (self.height() - scaled.height()) // 2
        self.setPixmap(scaled)
        self._scaled_rect = QtCore.QRect(
            x_off, y_off, scaled.width(), scaled.height()
        )

    def resizeEvent(self, event):
        """Handle widget resize by updating the scaled pixmap."""
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def set_zoom(self, factor: float):
        """
        Set zoom factor (clamped between 0.1 and 8.0) and update drawing.
        """
        self.zoom = max(0.1, min(8.0, factor))
        self._update_scaled_pixmap()

    def zoom_in(self):
        """Increase zoom by 25%."""
        self.set_zoom(self.zoom * 1.25)

    def zoom_out(self):
        """Decrease zoom by 25%."""
        self.set_zoom(self.zoom / 1.25)

    # ---------------- Coordinate transforms ----------------
    def _image_coords_to_screen(self, x_norm: float, y_norm: float):
        """
        Convert normalized image coordinates (0..1, 0..1) to screen coordinates.
        """
        if self._scaled_rect is None:
            return None
        x = self._scaled_rect.left() + int(x_norm * self._scaled_rect.width())
        y = self._scaled_rect.top() + int(y_norm * self._scaled_rect.height())
        return QtCore.QPoint(x, y)

    def _screen_to_image_norm(self, x: int, y: int):
        """
        Convert screen point to normalized image coordinates (0..1, 0..1).
        """
        if self._scaled_rect is None:
            return None, None
        rel_x = (x - self._scaled_rect.left()) / self._scaled_rect.width()
        rel_y = (y - self._scaled_rect.top()) / self._scaled_rect.height()
        rel_x = min(max(rel_x, 0), 1)
        rel_y = min(max(rel_y, 0), 1)
        return rel_x, rel_y

    # ---------------- Grid config ----------------
    def set_plate_config(self, config: WellPlateConfig):
        """
        Replace current grid configuration with a new WellPlateConfig.
        """
        self.plate_config = config
        self.update()
        self.grid_changed.emit()

    # ---------------- Painting ----------------
    def paintEvent(self, event):
        """
        Custom paint event to draw the semi-transparent background, the image
        pixmap, and the current grid lines.
        """
        super().paintEvent(event)
        if self.image is None or self._scaled_rect is None:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # Dimmed background
        overlay_color = QtGui.QColor(0, 0, 0, 60)
        painter.fillRect(self.rect(), overlay_color)

        # Draw pixmap
        if self.pixmap():
            painter.drawPixmap(self._scaled_rect, self.pixmap())

        # Grid lines
        pen = QtGui.QPen(QtCore.Qt.green, 1, QtCore.Qt.SolidLine)
        painter.setPen(pen)

        # Vertical lines
        for x_norm in self.plate_config.col_lines:
            p1 = self._image_coords_to_screen(x_norm, 0)
            p2 = self._image_coords_to_screen(x_norm, 1)
            if p1 and p2:
                painter.drawLine(p1, p2)

        # Horizontal lines
        for y_norm in self.plate_config.row_lines:
            p1 = self._image_coords_to_screen(0, y_norm)
            p2 = self._image_coords_to_screen(1, y_norm)
            if p1 and p2:
                painter.drawLine(p1, p2)

        painter.end()

    # ---------------- Mouse interaction for dragging lines ----------------
    def mousePressEvent(self, event):
        """
        Detect whether the mouse press is close to a grid line and
        start dragging that line.
        """
        if self.image is None or self._scaled_rect is None:
            return
        pos = event.pos()
        x = pos.x()
        y = pos.y()

        nearest = None
        nearest_dist = self.drag_threshold + 1

        # Vertical lines
        for idx, x_norm in enumerate(self.plate_config.col_lines):
            p = self._image_coords_to_screen(x_norm, 0.5)
            if p:
                dist = abs(p.x() - x)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = ("col", idx)

        # Horizontal lines
        for idx, y_norm in enumerate(self.plate_config.row_lines):
            p = self._image_coords_to_screen(0.5, y_norm)
            if p:
                dist = abs(p.y() - y)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = ("row", idx)

        self.dragging_line = nearest if nearest_dist <= self.drag_threshold else None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        When dragging_line is set, update the corresponding grid line
        position based on mouse movement while enforcing minimal spacing.
        """
        if self.dragging_line is None or self._scaled_rect is None:
            super().mouseMoveEvent(event)
            return

        x = event.pos().x()
        y = event.pos().y()
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
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Stop dragging when the mouse button is released."""
        self.dragging_line = None
        super().mouseReleaseEvent(event)

    # ---------------- Well ROI extraction ----------------
    def get_well_roi(self, row: int, col: int, require_enclosed: bool = True):
        """
        Return an OpenCV image for well at (row, col) using current grid.

        Returns
        -------
        np.ndarray or None
            BGR image containing the well ROI, or None if invalid or open.
        """
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

        if require_enclosed:
            if (
                x1_clamped == 0
                or y1_clamped == 0
                or x2_clamped == w
                or y2_clamped == h
            ):
                return None

        return self.image[y1_clamped:y2_clamped, x1_clamped:x2_clamped].copy()


# ------------------------------------------------------------------
# Metadata dialog
# ------------------------------------------------------------------
class MetadataDialog(QtWidgets.QDialog):
    """
    Dialog to load CSV metadata and preview mapping: well_id -> species.

    Expected columns (case-insensitive, flexible):
    - well_id / WellID / well / Well
    - species / Species / strain / Strain
    """

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

        self.metadata = {}  # dict well_id -> species

    def browse_csv(self):
        """Open a file dialog to choose a CSV file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select metadata CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self.csv_path_edit.setText(path)

    def load_csv(self):
        """Parse the CSV file and populate the internal metadata dict."""
        path = self.csv_path_edit.text().strip()
        if not path or not os.path.exists(path):
            QtWidgets.QMessageBox.warning(self, "Error", "CSV path invalid.")
            return

        meta = {}
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    well_id = (
                        row.get("well_id")
                        or row.get("WellID")
                        or row.get("well")
                        or row.get("Well")
                    )
                    species = (
                        row.get("species")
                        or row.get("Species")
                        or row.get("strain")
                        or row.get("Strain")
                    )
                    if not well_id or not species:
                        continue
                    meta[well_id.strip()] = species.strip()
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to read CSV:\n{e}"
            )
            return

        if not meta:
            QtWidgets.QMessageBox.warning(
                self,
                "No metadata",
                "No valid well_id/species rows were found.\n"
                "Check column names (e.g. well_id, species).",
            )
            return

        self.metadata = meta
        self.populate_table()

    def populate_table(self):
        """Fill the table widget with the loaded metadata."""
        self.table.setRowCount(len(self.metadata))
        for row_idx, (well_id, species) in enumerate(self.metadata.items()):
            self.table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(well_id))
            self.table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(species))


# ------------------------------------------------------------------
# Main window
# ------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for GridShredder.

    Responsibilities
    ----------------
    - Host the ImageCanvas as central widget.
    - Provide toolbars for:
      * Opening images.
      * Loading metadata.
      * Setting media labels.
      * Resetting and manipulating grid lines.
      * Auto‑adjusting grid based on plate/colony detection.
      * Zoom and label orientation.
      * Exporting well images.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GridShredder")
        self.resize(1400, 900)

        self.canvas = ImageCanvas()
        self.setCentralWidget(self.canvas)

        self.plate_config = self.canvas.plate_config
        self.metadata = {}  # well_id -> species
        self.media_name = "media"
        self.flip_label = False  # column numbering orientation

        # Stacks of indices of added lines (for proper undo)
        self._added_vertical_indices = []
        self._added_horizontal_indices = []

        self._create_toolbar()
        self._create_statusbar()

        self.canvas.grid_changed.connect(self._on_grid_changed)

        self._apply_dark_palette()

    # ---------------- UI styling ----------------
    def _apply_dark_palette(self):
        """Apply a dark visual theme with contrasting accent buttons."""
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#303030"))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#202020"))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#252525"))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#F0F0F0"))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#F0F0F0"))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#404040"))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#F0F0F0"))
        self.setPalette(palette)

        # Toolbar and buttons: high‑contrast teal accent on dark background
        self.setStyleSheet(
            """
            QToolBar {
                spacing: 6px;
                padding: 4px;
                background-color: #303030;
            }
            QToolButton {
                padding: 4px 8px;
                border: 1px solid #101010;
                border-radius: 3px;
                color: #F0F0F0;
                background-color: #0097A7;          /* bright teal */
            }
            QToolButton:hover {
                background-color: #00BCD4;          /* lighter teal */
            }
            QToolButton:pressed {
                background-color: #006978;          /* darker teal */
                padding-top: 5px;
                padding-left: 9px;
            }
            QStatusBar {
                background-color: #303030;
                color: #F0F0F0;
            }
            QTableWidget {
                gridline-color: #555555;
            }
            """
        )

    # ---------------- Toolbar and status bar ----------------
    def _create_toolbar(self):
        """Create and populate the main toolbar."""
        tb = QtWidgets.QToolBar("Main")
        tb.setIconSize(QtCore.QSize(20, 20))
        self.addToolBar(tb)

        # Image and metadata
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
        export_action.setToolTip(
            "Export enclosed wells as individual PNGs using current grid"
        )
        export_action.triggered.connect(self.export_wells)
        tb.addAction(export_action)

        tb.addSeparator()

        # Grid manipulation
        grid_action = QtWidgets.QAction("Reset grid", self)
        grid_action.setToolTip("Reset grid to default 8×12")
        grid_action.triggered.connect(self.reset_grid)
        tb.addAction(grid_action)

        add_col_action = QtWidgets.QAction("Add vertical", self)
        add_col_action.setToolTip("Insert extra vertical gridline")
        add_col_action.triggered.connect(self.add_vertical_line)
        tb.addAction(add_col_action)

        add_row_action = QtWidgets.QAction("Add horizontal", self)
        add_row_action.setToolTip("Insert extra horizontal gridline")
        add_row_action.triggered.connect(self.add_horizontal_line)
        tb.addAction(add_row_action)

        remove_lines_action = QtWidgets.QAction("Remove last lines", self)
        remove_lines_action.setToolTip(
            "Remove the most recently added vertical and horizontal gridlines"
        )
        remove_lines_action.triggered.connect(self.remove_last_added_lines)
        tb.addAction(remove_lines_action)

        tb.addSeparator()

        # Auto‑adjust
        auto_adjust_action = QtWidgets.QAction("Auto‑adjust", self)
        auto_adjust_action.setToolTip(
            "Auto‑adjust grid using detected plate border / colony area"
        )
        auto_adjust_action.triggered.connect(self.auto_adjust_grid_via_circle)
        tb.addAction(auto_adjust_action)

        tb.addSeparator()

        # Zoom controls
        zoom_in_action = QtWidgets.QAction("+", self)
        zoom_in_action.setToolTip("Zoom in")
        zoom_in_action.triggered.connect(self.canvas.zoom_in)
        tb.addAction(zoom_in_action)

        zoom_out_action = QtWidgets.QAction("−", self)
        zoom_out_action.setToolTip("Zoom out")
        zoom_out_action.triggered.connect(self.canvas.zoom_out)
        tb.addAction(zoom_out_action)

        tb.addSeparator()

        # Flip label toggle
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

    def _create_statusbar(self):
        """Initialize the status bar."""
        self.statusBar().showMessage("Ready")

    # ---------------- Callbacks for canvas grid changes ----------------
    def _on_grid_changed(self):
        """Update status bar whenever the grid configuration is modified."""
        rows = self.canvas.plate_config.n_row_segments
        cols = self.canvas.plate_config.n_col_segments
        self.statusBar().showMessage(
            f"Grid updated: {rows} rows × {cols} columns"
        )

    # ---------------- Image and metadata handling ----------------
    def open_image(self):
        """Open an image file from disk and display it on the canvas."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open plate image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif "
            "*.nef *.NEF *.cr2 *.CR2 *.cr3 *.CR3 *.arw *.ARW *.dng *.DNG);;"
            "All files (*)",
        )
        if not path:
            return
        try:
            self.canvas.load_image(path)
            self.statusBar().showMessage(f"Loaded image: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load image:\n{e}"
            )

    def load_metadata(self):
        """Launch MetadataDialog to load a CSV and update metadata."""
        dlg = MetadataDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.metadata = dlg.metadata
            self.statusBar().showMessage(
                f"Loaded metadata for {len(self.metadata)} wells"
            )

    def set_media(self):
        """Ask the user for a media name string, used in exported filenames."""
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Set media name",
            "Media label (e.g. YPD, PDA):",
            text=self.media_name,
        )
        if ok and text.strip():
            self.media_name = text.strip()
            self.statusBar().showMessage(f"Media set to: {self.media_name}")

    # ---------------- Grid operations ----------------
    def reset_grid(self):
        """Reset grid to default uniform 8×12 layout."""
        self.plate_config = WellPlateConfig()
        self.canvas.set_plate_config(self.plate_config)
        # Reset undo stacks as well
        self._added_vertical_indices.clear()
        self._added_horizontal_indices.clear()
        self.statusBar().showMessage("Grid reset to default 8×12.")

    def add_vertical_line(self):
        """Insert an extra vertical gridline midway inside the widest column."""
        lines = self.plate_config.col_lines
        if len(lines) < 2:
            return
        widths = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
        idx = int(np.argmax(widths))
        new_x = (lines[idx] + lines[idx + 1]) / 2.0
        insert_pos = idx + 1
        lines.insert(insert_pos, new_x)

        # Remember which line we added
        self._added_vertical_indices.append(insert_pos)

        self.canvas.update()
        self.canvas.grid_changed.emit()
        self.statusBar().showMessage("Inserted extra vertical gridline.")

    def add_horizontal_line(self):
        """Insert an extra horizontal gridline midway inside the tallest row."""
        lines = self.plate_config.row_lines
        if len(lines) < 2:
            return
        heights = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
        idx = int(np.argmax(heights))
        new_y = (lines[idx] + lines[idx + 1]) / 2.0
        insert_pos = idx + 1
        lines.insert(insert_pos, new_y)

        # Remember which line we added
        self._added_horizontal_indices.append(insert_pos)

        self.canvas.update()
        self.canvas.grid_changed.emit()
        self.statusBar().showMessage("Inserted extra horizontal gridline.")

    def remove_last_added_lines(self):
        """
        Remove the most recently added vertical and horizontal gridlines, if present.

        Uses stacks of indices recorded when lines were added so that the
        correct lines (in reverse insertion order) are removed.
        """
        removed_any = False

        # Remove last added vertical line, if any
        if self._added_vertical_indices:
            idx = self._added_vertical_indices.pop()
            # ensure it's still an inner line
            if 0 < idx < len(self.plate_config.col_lines) - 1:
                self.plate_config.col_lines.pop(idx)
                removed_any = True

        # Remove last added horizontal line, if any
        if self._added_horizontal_indices:
            idx = self._added_horizontal_indices.pop()
            if 0 < idx < len(self.plate_config.row_lines) - 1:
                self.plate_config.row_lines.pop(idx)
                removed_any = True

        if removed_any:
            self.canvas.update()
            self.canvas.grid_changed.emit()
            self.statusBar().showMessage(
                "Removed last added gridlines (if any)."
            )
        else:
            self.statusBar().showMessage(
                "No added gridlines left to remove."
            )

    # ---------------- Auto‑adjust via circle/contour detection ----------------
    def auto_adjust_grid_via_circle(self):
        """
        Detect the plate region or dominant colony area and adjust the grid.

        Strategy
        --------
        1. Grayscale + blur.
        2. HoughCircles for circular plates.
        3. Adaptive threshold + morphology + contour detection.
        4. Choose circle bounding box if present, otherwise largest contour.
        5. Uniformly redistribute grid lines inside the chosen bounding box.
        """
        if self.canvas.image is None:
            QtWidgets.QMessageBox.warning(
                self, "No image", "Load an image first."
            )
            return

        img = self.canvas.image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)

        h, w = gray.shape[:2]

        circle_bbox = None
        try:
            circles = cv2.HoughCircles(
                gray_blur,
                cv2.HOUGH_GRADIENT,
                dp=1.0,
                minDist=min(h, w) // 2,
                param1=100,
                param2=40,
                minRadius=min(h, w) // 6,
                maxRadius=min(h, w) // 2,
            )
        except cv2.error:
            circles = None

        if circles is not None:
            circles = np.uint16(np.around(circles))
            x_c, y_c, r = circles[0][0]
            left_c = max(0, x_c - r)
            right_c = min(w - 1, x_c + r)
            top_c = max(0, y_c - r)
            bottom_c = min(h - 1, y_c + r)
            circle_bbox = (left_c, top_c, right_c, bottom_c)

        thr = cv2.adaptiveThreshold(
            gray_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51,
            5,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        morph = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_bbox = None
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_idx = int(np.argmax(areas))
            max_area = areas[max_idx]
            if max_area > 0.05 * (h * w):
                x, y, bw, bh = cv2.boundingRect(contours[max_idx])
                contour_bbox = (x, y, x + bw, y + bh)

        if circle_bbox is None and contour_bbox is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Auto‑adjust",
                "No suitable plate/colony region detected.\n"
                "Try increasing contrast or adjusting the image.",
            )
            return

        if circle_bbox is not None:
            left, top, right, bottom = circle_bbox
        else:
            left, top, right, bottom = contour_bbox

        pad_x = int(0.05 * (right - left))
        pad_y = int(0.05 * (bottom - top))
        left = max(0, left - pad_x)
        right = min(w - 1, right + pad_x)
        top = max(0, top - pad_y)
        bottom = min(h - 1, bottom + pad_y)

        x0_norm = left / float(w)
        x1_norm = right / float(w)
        y0_norm = top / float(h)
        y1_norm = bottom / float(h)

        rows = self.plate_config.rows
        cols = self.plate_config.cols

        self.plate_config.col_lines = np.linspace(
            x0_norm, x1_norm, cols + 1
        ).tolist()
        self.plate_config.row_lines = np.linspace(
            y0_norm, y1_norm, rows + 1
        ).tolist()

        self.canvas.update()
        self.canvas.grid_changed.emit()
        self.statusBar().showMessage(
            "Grid auto‑adjusted around detected plate/colony region."
        )

    # ---------------- Label orientation and export ----------------
    def set_flip_labels(self, checked: bool):
        """
        Toggle label orientation (left→right or right→left) used in well IDs.
        """
        self.flip_label = checked
        if self.flip_label:
            self.flip_label_action.setText("Right→Left labels (ON)")
            self.statusBar().showMessage(
                "Right→Left label orientation active (A01 at right)."
            )
        else:
            self.flip_label_action.setText("Left→Right labels (ON)")
            self.statusBar().showMessage(
                "Left→Right label orientation active (A01 at left)."
            )

    def well_id_for(self, row: int, col: int) -> str:
        """
        Convert (row, col) to well ID like A01, B06, H12.

        Column numbering can be flipped depending on self.flip_label.
        """
        row_letter = chr(ord("A") + row)
        n_cols = self.canvas.plate_config.n_col_segments
        num = col + 1 if not self.flip_label else n_cols - col
        return f"{row_letter}{num:02d}"

    def export_wells(self):
        """
        Export each enclosed well as a PNG image to a user-selected folder.

        Filenames: {species}_{media_name}_{well_id}.png
        """
        if self.canvas.image is None:
            QtWidgets.QMessageBox.warning(
                self, "No image", "Load an image first."
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm export",
            "Export wells using current grid lines?\n"
            "Only wells fully enclosed by grid (not touching image edges)\n"
            "will be exported.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder"
        )
        if not out_dir:
            return

        rows = self.canvas.plate_config.n_row_segments
        cols = self.canvas.plate_config.n_col_segments

        export_count = 0
        skipped_open = 0

        for r in range(rows):
            for c in range(cols):
                roi = self.canvas.get_well_roi(r, c, require_enclosed=True)
                if roi is None:
                    skipped_open += 1
                    continue

                well_id = self.well_id_for(r, c)
                species = self.metadata.get(well_id, "unknown")

                prefix = f"{species}_{self.media_name}_{well_id}"
                fname = os.path.join(out_dir, f"{prefix}.png")

                ok, buf = cv2.imencode(".png", roi)
                if ok:
                    buf.tofile(fname)
                    export_count += 1

        self.statusBar().showMessage(
            f"Exported {export_count} wells to {out_dir} "
            f"(skipped {skipped_open} non‑enclosed wells)."
        )


# ------------------------------------------------------------------
# Application entry point
# ------------------------------------------------------------------
def main():
    """
    Configure high‑DPI settings and launch the Qt application.
    """
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_EnableHighDpiScaling, True
    )
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_UseHighDpiPixmaps, True
    )

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print("Fatal error in main():", e)
        traceback.print_exc()
