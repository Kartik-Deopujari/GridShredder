# GridShredder2

GridShredder is an interactive desktop tool for turning messy plate photos into clean, analysis‑ready well images. Built with Qt, it lets you load an image of a rectangular agar plate (for example, a 96‑well or custom grid layout), overlay a virtual grid or boxes, and adjust each grid line/box by eye so positions match the actual colonies on the plate. Once the setup looks right, GridShredder automatically crops out each part and saves it as an individual image with filenames based on your metadata. 

For biologists, this means you can go from “one big plate photo” to “one image per colony/tile, correctly named and ready for analysis” in a single step. You can import a simple CSV linking well IDs to strain names, species, or treatments, and GridShredder will bake that information directly into the filenames. This is particularly useful for high‑throughput screens, growth assays, or imaging‑based phenotyping, where you might later quantify colony size, colour, or texture using ImageJ, Python scripts, or machine learning models—and you don’t want to waste time manually tracking isolate and ID.

<img width="1864" height="1047" alt="Screenshot from 2026-03-24 22-36-35" src="https://github.com/user-attachments/assets/bf074712-ef88-4740-8552-66237b6d8013" />
AI Generated image.

---

## Features

- **Custom grid size** (default 8 × 12 layout), change as per sample requirement.
- **New box mode!** Useful when colonies are not perfectly separated using grids.
- Customizable box size to accomodate for varying colony shape and size.
- Interactive grid and box overlay on plate images.
- Support for common image formats (PNG, JPG, TIFF, BMP, GIF) and RAW formats (NEF, CR2/CR3, ARW, DNG) via `rawpy`.
- Metadata import from CSV to map `well_id → sample_ID`.
- Media label support (e.g. `YPD`, `PDA`) embedded in filenames.
- Left→right or right→left column labelling (`A01` at left or right).
- Export of **only fully enclosed wells** as individual PNG files:
  - Filenames: `{sample_ID}_{media}_{well_id}.png`.
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
<img width="1861" height="1046" alt="Screenshot from 2026-03-27 16-59-48" src="https://github.com/user-attachments/assets/e0d9b7bb-8cbf-4d62-9a4a-ca0aa20ed590" />

### 1. Launch and load an image

In a terminal:

```bash
python GridShredder2.py
```

Then use the **“Open image”** toolbar button to load a plate image.  
GridShredder supports standard image formats (PNG, JPG/JPEG, TIFF) and RAW formats (e.g. NEF) via `rawpy`.

### 2. Adjust the grid

- Click **“Set plate size”** button in the tool bar to change the grid size (default = 8x12).
- Drag grid lines directly to fine‑tune their positions.  
- Click **“Reset grid”** to revert all grid changes.

If your image contains more colonies or objects than a standard 96‑well layout, you can insert additional grid lines using **“Add vertical”** and **“Add horizontal”**.
<img width="1861" height="1046" alt="Screenshot from 2026-03-27 17-00-30" src="https://github.com/user-attachments/assets/d840e247-5c96-4a9c-a47a-fdb33db59a91" />

### 3. Box mode
- Click **"Mode:Box/Grid"** button to toggle between grid and box mode.
- Number of boxes = gride size determined at the start of the program.
- Boxes can be easily dragged and resized.
- New boxes can be added with **“Add box”** button and removed with **“Remove box”** button (experimental).
<img width="1861" height="1046" alt="Screenshot from 2026-03-27 17-02-36" src="https://github.com/user-attachments/assets/cb577e24-e5e4-4611-b5a0-82ee88f99bfc" />


### 4. Load sample metadata

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

### 5. Export wells

Use the **“Export wells”** button to export all cropped wells to a chosen output folder.  

--
Developed by Kartik Deopujari with assistance from AI coding tools; final design and implementation curated and tested by the author.
