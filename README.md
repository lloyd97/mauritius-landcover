---
title: Mauritius Land Cover Classification
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Mauritius Land Cover Classification

Deep learning-based land cover classification for Mauritius using Sentinel-2 satellite imagery and U-Net with ResNet50 encoder.

## Overview

This project uses a U-Net segmentation model with a ResNet50 encoder to classify land cover types across Mauritius from Sentinel-2 multispectral satellite imagery.

### Land Cover Classes (Apple Maps Style)

| Class | Color | RGB | Description |
|-------|-------|-----|-------------|
| Water | Soft Blue | (168, 216, 234) | Ocean, lagoons, rivers, reservoirs |
| Forest | Rich Green | (139, 195, 74) | Native forests, dense vegetation |
| Plantation | Light Green | (197, 225, 165) | Sugarcane fields, agricultural land |
| Urban | Warm Gray | (215, 204, 200) | Buildings, developed areas |
| Roads | Medium Gray | (158, 158, 158) | Highways, streets, paved surfaces |
| Bare Land | Sandy Cream | (239, 235, 233) | Quarries, beaches, cleared land |

### Model Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 73.50% |
| **Water Accuracy** | 99.77% |
| **Forest Accuracy** | 88.80% |
| **Urban Accuracy** | 85.29% |
| **Bare Land Accuracy** | 76.82% |
| **Plantation Accuracy** | 54.75% |
| **Roads Accuracy** | 37.65% |

## Features

- **Live Interactive Map**: Real-time land cover classification as you pan across Mauritius
- **9-Band Input**: Uses B2, B3, B4, B8, B11, B12 + NDVI, NDWI, NDBI indices
- **Apple Maps-inspired styling**: Clean, aesthetic color palette
- **Google Earth Engine integration**: Automatic Sentinel-2 imagery download

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mauritius-landcover.git
cd mauritius-landcover

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine
earthengine authenticate
```

## Usage

### Live Interactive Map

```bash
python src/web/live_interactive_map.py
# Open http://localhost:5003 in your browser
```

Pan around the map to automatically fetch Sentinel-2 imagery and classify land cover in real-time.

### Training Pipeline

```bash
# 1. Download training tiles from Google Earth Engine
python scripts/gather_targeted_samples.py

# 2. Auto-label tiles using spectral indices
python scripts/auto_label_tiles_fixed.py

# 3. Train model with class weights
python scripts/train_enhanced.py --epochs 50 --use-class-weights
```

## Project Structure

```
mauritius-landcover/
├── src/
│   ├── web/
│   │   └── live_interactive_map.py    # Flask web app with Leaflet.js
│   ├── models/
│   │   └── unet.py                     # U-Net model definition
│   └── data/
│       └── dataset.py                  # PyTorch dataset class
├── scripts/
│   ├── gather_targeted_samples.py      # Download training tiles from GEE
│   ├── auto_label_tiles_fixed.py       # Rule-based auto-labeling
│   ├── train_enhanced.py               # Training script with class weights
│   └── create_training_dataset.py      # Dataset creation utilities
├── checkpoints/                         # Model weights (not in repo)
├── data/
│   └── training/tiles/                  # Training data (not in repo)
└── requirements.txt
```

## Technical Details

### Input Data
- **6 Spectral Bands**: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)
- **3 Spectral Indices**: NDVI, NDWI, NDBI
- **Resolution**: 10m per pixel
- **Tile Size**: 256 x 256 pixels

### Model Architecture
- **Encoder**: ResNet50 (pretrained on ImageNet)
- **Decoder**: U-Net style upsampling
- **Input Channels**: 9 (modified first conv layer)
- **Output Classes**: 7 (including background)

### Training
- **Loss**: Class-weighted CrossEntropyLoss
- **Optimizer**: AdamW with discriminative learning rates
- **Augmentation**: Random flips, rotations, color jittering
- **Epochs**: 50

## Requirements

- Python 3.8+
- PyTorch 2.0+
- segmentation-models-pytorch
- Google Earth Engine Python API
- Flask
- rasterio
- numpy, Pillow

## License

MIT License

## Author

Lloyd Florens - PhD Research, Mauritius Land Cover Classification
