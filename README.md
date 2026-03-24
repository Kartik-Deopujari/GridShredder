# GridShredder

GridShredder is an interactive desktop tool for turning messy plate photos into clean, analysis‑ready well images. Built with Qt, it lets you load an image of a rectangular agar plate (for example, a 96‑well or custom grid layout), overlay a virtual grid, and adjust each grid line by eye so wells match the actual colonies on the plate. Once the grid looks right, GridShredder automatically crops out each well and saves it as an individual image with filenames based on your metadata. 

For biologists, this means you can go from “one big plate photo” to “one image per colony/tile, correctly named and ready for analysis” in a single step. You can import a simple CSV linking well IDs to strain names, species, or treatments, and GridShredder will bake that information directly into the filenames. This is particularly useful for high‑throughput screens, growth assays, or imaging‑based phenotyping, where you might later quantify colony size, colour, or texture using ImageJ, Python scripts, or machine learning models—and you don’t want to waste time manually tracking isolate and ID.

---

## Features

- Interactive grid overlay on plate images (default 8 × 12 layout).
- Click‑and‑drag adjustment of grid lines with spacing constraints.
- Support for common image formats (PNG, JPG, TIFF, BMP, GIF) and RAW formats (NEF, CR2/CR3, ARW, DNG) via `rawpy`.
- Metadata import from CSV to map `well_id → species` (or strain).
- Media label support (e.g. `YPD`, `PDA`) embedded in filenames.
- Auto‑adjust grid using plate/colony detection:
  - Circular plate detection via Hough transform.
- Left→right or right→left column labelling (`A01` at left or right).
- Export of **only fully enclosed wells** as individual PNG files:
  - Filenames: `{species}_{media}_{well_id}.png`.
- Zoomable view with arrow‑key scrolling to inspect wells at high magnification.

---
## Dependencies

GridShredder requires:

- Python 3.8 or newer
- PyQt5
- numpy
- opencv-python
- rawpy
  
---

## Recommended Conda environment

To create an isolated environment for GridShredder with all required dependencies:

conda create -n gridshredder python=3.10

conda activate gridshredder

conda install -c conda-forge pyqt opencv numpy rawpy

---

## Usage

### 1. Launch and load an image

In a terminal:

```bash
python GridShredder.py
```

Then use the **“Open image”** toolbar button to load a plate image.  
GridShredder supports standard image formats (PNG, JPG/JPEG, TIFF) and RAW formats (e.g. NEF) via `rawpy`.

### 2. Adjust the grid

- Click **“Auto‑adjust”** to automatically align the grid to the plate/colony area (Experimental).  
- Drag grid lines directly to fine‑tune their positions.  
- Click **“Reset grid”** to revert all grid changes.

If your image contains more colonies or objects than a standard 96‑well layout, you can insert additional grid lines using **“Add vertical”** and **“Add horizontal”**.

### 3. Load sample metadata

Well IDs (e.g. `A01`, `B02`) and corresponding isolate IDs can be imported from a CSV file.  
The CSV should include the following headers:

```text
well_id,species
```

After selecting the metadata file, click **“Load”** in the dialog to import it.  
After cropping, each image is saved using the metadata information:

```text
isolateID_media_positionID
```

You can specify the **media** label (e.g. `YPD`, `PDA`) using the **“Set media”** option.

If your plate layout starts at the right (i.e. `A01` is on the right edge), toggle the **“Left→Right labels”** button to flip the column labels to a right‑to‑left orientation.

### 4. Export wells

Use the **“Export wells”** button to export all cropped wells to a chosen output folder.  
