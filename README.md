# OceanEmbed 🌊

**Satellite Embedding-Based Deep Learning Framework for 3D Subsurface Ocean Temperature Reconstruction**

> Submitted to **Smart India Hackathon (SIH) 2026**

OceanEmbed reconstructs the full 3-dimensional temperature structure of the North Indian Ocean at 15 depth levels — using only 7 daily surface satellite observations as input. A U-Net CNN with Self-Attention achieves **97.56% accuracy** on unseen test data (Oct–Dec 2023).

---

## 🗺️ Problem Domain

| Property | Value |
|---|---|
| Region | North Indian Ocean (5°N–30°N, 45°E–105°E) |
| Spatial Resolution | 0.25° × 0.25° |
| Temporal Resolution | Daily |
| Surface Inputs | SST, SSS, SSH, U/V Currents, U/V Winds (7 channels) |
| Output | Temperature at 15 depths: 0–1000 m |
| Training Data | GLORYS12V1 Reanalysis, 2023 (365 days) |
| Train/Test Split | Jan–Sep 2023 (train) · Oct–Dec 2023 (unseen test) |

---

## 📁 Repository Structure

```
OceanEmbed/
│
├── config.yaml                  # Hyperparameters and bounding box config
├── requirements.txt             # Python dependencies
├── .env                         # Copernicus & NASA API credentials (not committed)
├── .env.example                 # Credential template
│
├── data/
│   ├── raw/                     # Downloaded NetCDF files
│   └── processed/               # Regridded & normalized tensors (.pt)
│
├── models/
│   └── ocean_embed_cnn.pt       # Saved trained model weights
│
├── src/
│   ├── data/
│   │   ├── download.py          # Synthetic data generator (quick PoC)
│   │   ├── download_real.py     # Copernicus & NASA API downloader
│   │   ├── preprocess.py        # Grid alignment, interpolation, normalization
│   │   └── dataset.py           # PyTorch Dataset (+ feature engineering)
│   │
│   ├── models/
│   │   └── ocean_embed_cnn.py   # U-Net + Self-Attention architecture
│   │
│   ├── training/
│   │   └── train.py             # Training loop with chronological train/test split
│   │
│   ├── evaluation/
│   │   ├── metrics.py           # RMSE, Bias, Pearson Correlation, Accuracy %
│   │   └── validate_argo.py     # Independent validation against real ARGO floats
│   │
│   └── app/
│       └── main.py              # FastAPI backend — serves predictions & metrics
│
├── frontend/                    # React + Vite interactive dashboard
│   ├── src/
│   │   ├── App.tsx              # Main dashboard application
│   │   └── index.css            # Styling
│   ├── package.json
│   └── vite.config.ts           # Proxies /api/* → localhost:8000
│
├── notebooks/
│   └── demo_inference.py        # Standalone inference visualization script
│
├── results/
│   └── ocean_embed_inference.png # Generated inference comparison plot
│
└── PROJECT_EXPLAINED.md         # Complete plain-English system explanation
```

---

## 🚀 Running the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
cd frontend && npm install
```

### 2. Download & Preprocess Data

Register for free accounts at [Copernicus Marine](https://marine.copernicus.eu/) and [NASA Earthdata](https://urs.earthdata.nasa.gov/), then add your credentials to `.env`:

```bash
cp .env.example .env
# fill in COPERNICUS_USERNAME, COPERNICUS_PASSWORD, NASA_EARTHDATA_TOKEN
python src/data/download_real.py
python src/data/preprocess.py
```

### 3. Train the Model

```bash
python src/training/train.py
```

Training runs for 50 epochs. The model is saved to `models/ocean_embed_cnn.pt`. Chronological split is enforced — test set is the final 20% of days (Oct–Dec 2023) to prevent data leakage.

### 4. Launch the Interactive Dashboard

Start both servers (in separate terminals or let the IDE manage them):

```bash
# Terminal 1 — Backend API
python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open **[http://localhost:5173](http://localhost:5173)** in your browser.

---

## 📊 Dashboard Features

The interactive web dashboard provides a complete live demo:

| Feature | Description |
|---|---|
| **Deep Ocean Reconstruction** | Side-by-side maps: GLORYS Ground Truth · AI Prediction · Error Delta |
| **Surface Satellite Inputs** | All 7 input variable heatmaps (SST, SSS, SSH, U/V currents, U/V winds) |
| **Date Picker** | Navigate any day in 2023 with a calendar picker |
| **Depth Selector** | 15 clickable depth buttons (0 m → 1000 m) |
| **Coordinate Probe** | Click anywhere on the ocean → instant depth profile table at that point |
| **Map Pin** | White ✕ marker on all maps showing the probed location |
| **Export CSV** | Download current depth layer data (lat/lon/true/pred/error) |
| **Export Profile CSV** | Download full 15-depth profile at a probed coordinate |
| **Export Metrics JSON** | Machine-readable validation report |
| **Visualization Style** | Toggle between Heatmap and Contour |

---

## 🧠 Model Architecture

```
Input: (Batch, 11, 101, 241)
  → 7 satellite channels
  → 2 coordinate channels (lat, lon — CoordConv)
  → 2 temporal channels (sin/cos of day-of-year)

Encoder:  Conv2d blocks (64 → 128 → 256 channels) + MaxPool
Bottleneck: Self-Attention (ViT-style) for global spatial context
Decoder:  ConvTranspose2d blocks (256 → 128 → 64) + skip connections
Output: (Batch, 15, 101, 241) — temperature at 15 depth levels
```

---

## 📈 Validation Results

| Metric | Value |
|---|---|
| **Accuracy** | **97.56%** |
| **RMSE** | 0.7329 °C |
| **Mean Bias** | +0.0204 °C |
| **Pearson Correlation** | 0.9879 |

> Evaluated on Oct–Dec 2023 (unseen test set, never seen during training).

---

## 🔬 Independent ARGO Validation

Beyond GLORYS-based metrics, the model is cross-validated against **real ARGO float profiles** (`src/evaluation/validate_argo.py`). ARGO floats are autonomous robotic instruments that physically dive through the ocean and measure temperature — they are completely independent of the GLORYS training data.

---

## 💡 Why 2023 Data?

We train on 2023 because:
1. **Quality**: GLORYS "Multi-Year" reanalysis for 2023 is fully validated, cloud-gap-filled, and calibrated — unlike Near Real-Time (NRT) data which has raw sensor noise.
2. **Overlap**: All 6 satellite missions (SST, SSS, SSH, winds, currents, subsurface) have finalized reprocessed versions for 2023.
3. **No leakage**: The strict Oct–Dec 2023 test holdout proves generalization to genuinely unseen days.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | PyTorch 2.x |
| Data I/O | xarray, netCDF4, copernicusmarine |
| Backend API | FastAPI + Uvicorn |
| Frontend | React 18 + Vite + TypeScript |
| Visualization | Plotly.js (react-plotly.js) |
| Styling | Vanilla CSS + Tailwind CSS |
