# 🌿 Rosa-sinensis (Hibiscus) Leaf Health Detection Web Application

> **Design and Development of a Model for Early Detection of Plant Health in *Rosa-Sinensis* using Leaf Vein Architecture**

A transparent, rule-based OpenCV/Python computer vision pipeline and clean, modern Single-Page Application (SPA) frontend for diagnosing nutritional deficiencies (Iron vs Nitrogen) and assessing vascular reticulation in *Hibiscus rosa-sinensis*.

---

## 📁 Project Structure

All files for the application reside directly inside the `FRONTEND` folder:

```
FRONTEND/
├── app.py              # FastAPI backend wrapping the OpenCV rule-based diagnostic pipeline
├── index.html          # Clean semantic botanical SPA user interface
├── styles.css          # Modernist dark-mode botanical design system (vanilla CSS)
├── app.js              # Interactive UI controller, drag & drop, API integration, and reports
├── requirements.txt    # Python dependencies
└── README.md           # Instructions & technical documentation
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have **Python 3.10+** installed. (FastAPI, OpenCV, NumPy, and Uvicorn are already installed in your current Python environment).

### 2. Navigate to the `FRONTEND` directory
Open PowerShell or your preferred terminal in this project directory:

```powershell
cd "c:\Users\THEJASWI NAYAK\Desktop\Web Dev\Frontend Major\FRONTEND"
```

*(Optional: if running on a fresh environment, install dependencies)*:
```powershell
pip install -r requirements.txt
```

### 3. Launch the Backend Server
Run `app.py`:

```powershell
python app.py
```

Or run via Uvicorn directly:

```powershell
uvicorn app:app --reload --port 8000
```

### 4. Open in Your Browser
Open your browser and visit:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## ✨ Key Features

1. **Clean Botanical Aesthetic**:
   - Deep forest green accents, glassmorphic cards, modern typography (`Plus Jakarta Sans`, `Outfit`, and `JetBrains Mono`).
   - Color-coded badges:
     - 🟢 **Healthy Foliage** (Green)
     - 🟡 **Mild Deficiency** (Amber/Yellow)
     - 🟠 **Moderate Deficiency** (Orange)
     - 🔴 **Severe Deficiency** (Red)
     - ⚪ **Non-Hibiscus Rejection** (Gray)

2. **Smart Upload & Instant Demo Presets**:
   - **Unified drag-and-drop dropzone** accepting leaf photos.
   - Optional secondary upload for backlit transmitted-light photos.
   - **5 One-Click Demo Specimens** (`Healthy`, `Mild Fe Chlorosis`, `Moderate Stress`, `Severe Bleaching`, and `Non-Hibiscus Leaf`) for instant testing without uploading files.

3. **Transparent Rule-Based Metrics Grid**:
   - **Vein Density ($V_\rho$)**: Normal healthy range 18.0% – 25.0%.
   - **Yellow Pixel Ratio ($Y_r$)**: Normal healthy threshold < 5.0%.
   - **Excess Green Index ($\text{ExG}$)**: Contrast index $2G - R - B$ (Healthy > 35.0).
   - **Dark Green Color Index ($\text{DGCI}$)**: Normalized spectrophotometric index (Healthy 0.55 – 0.75).
   - **Vascular Branch Points ($N_{bp}$)**: Topological bifurcations/trifurcations (Healthy 120 – 250 junctions).
   - Interactive gauge fill bars showing calibration threshold band comparisons.

4. **Visual Inspection Gallery**:
   - Side-by-side 4-channel view:
     - **Channel A**: Original Frontlit Photo
     - **Channel B**: Segmented Leaf Mask
     - **Channel C**: Extracted Vascular Vein Skeleton Overlay (Bright red overlay on tissue)
     - **Channel D**: Highlighted Unhealthy Chlorotic Regions (Gold/amber heatmap)
   - Click any image for **high-resolution lightbox zoom**.

5. **All 12 Preprocessing Pipeline Stages (Collapsible Drawer)**:
   - Stage 1: Raw Backlit Leaf (Transmitted Illumination)
   - Stage 2: Grayscale & Luminance Normalization
   - Stage 3: CLAHE (Contrast-Limited Adaptive Histogram Equalization)
   - Stage 4: Otsu Leaf Contour Segmentation & Background Isolation
   - Stage 5: Masked Lamina Tissue Extraction
   - Stage 6: Multi-Scale Frangi Vesselness Filter Approximation
   - Stage 7: Binarized Vascular Network (Adaptive Thresholding)
   - Stage 8: Morphological Vein Skeletonization (Medial Axis)
   - Stage 9: Vascular Branch Points & Junction Nodes
   - Stage 10: Frontlit Leaf Normalization & Color Space Deconvolution
   - Stage 11: ExG & DGCI Color Space Spatial Map
   - Stage 12: Interveinal Chlorosis Segmentation & Vein Proximity Map

6. **Reporting & Actions**:
   - **Download Full Diagnostic Report**: Generates and downloads a timestamped `.txt` diagnostic dossier.
   - **Print Diagnostic Sheet**: Formatted clean print view for laboratory records.
   - **Analyze Another Leaf**: One-click reset to test another specimen.
