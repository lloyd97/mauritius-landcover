# Mauritius Land Cover Classification - Complete Guide

## 🎯 Project Overview

This project implements a deep learning system for land cover classification in Mauritius using Sentinel-2 satellite imagery. The system can detect and classify:

- **Water** (Blue) - Rivers, lakes, ocean
- **Forest** (Dark Green) - Dense vegetation
- **Plantation** (Light Green) - Agricultural land, sugar cane
- **Urban** (Brown) - Buildings and developed areas
- **Roads** (Grey) - Transportation infrastructure
- **Bare Land** (Tan) - Exposed soil, cleared areas

## 🏗️ Architecture

- **Model**: U-Net with ResNet50 encoder
- **Pre-training**: ImageNet weights (transfer learning)
- **Input**: 9-channel Sentinel-2 imagery (6 bands + 3 indices)
- **Output**: 7-class semantic segmentation
- **Framework**: PyTorch + segmentation-models-pytorch

## 📊 Data Pipeline

### 1. Data Acquisition
Real Sentinel-2 L2A imagery downloaded from Google Earth Engine across 8 locations in Mauritius:
- Port Louis (urban)
- Black River (coastal)
- Grand Bassin (forest)
- Pamplemousses (agricultural)
- Curepipe (urban)
- Riviere du Rempart (agricultural)
- Mahebourg (coastal)
- Central Plateau (mixed)

### 2. Automated Labeling
Rule-based classification using spectral indices:
- **NDVI** (vegetation)
- **NDWI** (water)
- **NDBI** (built-up areas)
- **MNDWI** (modified water index)
- **EVI** (enhanced vegetation index)
- **BSI** (bare soil index)

### 3. Training Data
- **32 tiles** (256x256 pixels each)
- **9 input channels**: B2, B3, B4, B8, B11, B12 + NDVI, NDWI, NDBI
- **Auto-labeled masks** for all 7 classes
- **80/20 train/val split**

## 🚀 Quick Start

### Run the Web Viewer

```bash
cd src/web
python live_map_app.py
```

Then open http://localhost:5000 in your browser

### Train the Model (Basic)

```bash
cd scripts
python train_model_pretrained.py \
  --data-dir ../data/training/tiles \
  --epochs 30 \
  --batch-size 2 \
  --lr 0.0001 \
  --output ../checkpoints/best_model.pth
```

### Train with Enhanced Features

```bash
python train_enhanced.py \
  --data-dir ../data/training/tiles \
  --epochs 50 \
  --batch-size 4 \
  --encoder-lr 0.00001 \
  --decoder-lr 0.0001 \
  --output ../checkpoints/enhanced_model.pth
```

## 📁 Project Structure

```
mauritius-landcover/
├── data/
│   ├── training/
│   │   ├── tiles/              # Training tiles (32 tiles)
│   │   │   ├── *_tile_*.npy    # 9-channel imagery
│   │   │   └── *_mask_*.npy    # Ground truth masks
│   │   └── *.tif               # Original Sentinel-2 downloads
│   └── live_imagery/           # Runtime downloaded imagery
├── checkpoints/
│   ├── best_model.pth          # Trained model
│   └── enhanced_model.pth      # Enhanced model (if trained)
├── src/
│   ├── data/
│   │   └── sentinel_downloader.py
│   ├── models/
│   │   ├── unet.py
│   │   └── segmentation.py
│   └── web/
│       ├── app.py              # Demo interface
│       └── live_map_app.py     # Live Sentinel-2 viewer
├── scripts/
│   ├── create_training_dataset.py  # Download & create tiles
│   ├── auto_label_tiles.py         # Automated labeling
│   ├── train_model_pretrained.py   # Basic training
│   └── train_enhanced.py           # Enhanced training
└── notebooks/
    └── exploration.ipynb
```

## 🔧 Key Scripts

### 1. Create Training Dataset
```bash
python scripts/create_training_dataset.py \
  --download \
  --output-dir ../data/training
```

Downloads Sentinel-2 imagery from 8 Mauritius locations and creates 256x256 tiles.

### 2. Auto-Label Tiles
```bash
python scripts/auto_label_tiles.py \
  --tiles-dir ../data/training/tiles
```

Automatically generates ground truth masks using spectral index rules.

### 3. Train Model
```bash
python scripts/train_model_pretrained.py \
  --data-dir ../data/training/tiles \
  --epochs 30 \
  --batch-size 2
```

Basic training with ImageNet pre-trained encoder.

### 4. Enhanced Training
```bash
python scripts/train_enhanced.py \
  --data-dir ../data/training/tiles \
  --epochs 50 \
  --encoder-lr 0.00001 \
  --decoder-lr 0.0001
```

Advanced training with:
- Discriminative learning rates
- LR scheduling
- Per-class IoU/F1 metrics
- Mixed precision (if CUDA available)

## 📈 Training Results

From the initial 30-epoch training run:

| Epoch | Train Loss | Val Loss | Status |
|-------|------------|----------|--------|
| 1     | 2.158      | 2.219    | Baseline |
| 10    | 1.363      | 1.202    | Improving |
| 20    | 0.982      | 0.871    | Good progress |
| 30    | 0.756      | 0.698    | Converged |

**Best Validation Loss**: 0.698

## 🌐 Web Interface

The live map viewer provides a 3-panel interface:

1. **Interactive Map** (left) - Click to select location
2. **Sentinel-2 Image** (center) - Real satellite imagery
3. **Classification** (right) - Color-coded land cover

### Features:
- Live Sentinel-2 data download
- Real-time classification
- Color-coded output matching requirements
- Coordinate-based image variation

## 🎨 Color Scheme

| Class | Color | Hex Code | RGB |
|-------|-------|----------|-----|
| Water | Blue | #0064FF | (0, 100, 255) |
| Forest | Dark Green | #006400 | (0, 100, 0) |
| Plantation | Light Green | #32CD32 | (50, 205, 50) |
| Urban | Brown | #8B4513 | (139, 69, 19) |
| Roads | Grey | #808080 | (128, 128, 128) |
| Bare Land | Tan | #D2B48C | (210, 180, 140) |

## 🔬 Technical Details

### Input Specifications
- **Bands**: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)
- **Indices**: NDVI, NDWI, NDBI
- **Resolution**: 10m per pixel (Sentinel-2)
- **Tile size**: 256x256 pixels
- **Input shape**: (9, 256, 256)

### Model Architecture
- **Encoder**: ResNet50 (ImageNet pre-trained)
- **Decoder**: U-Net upsampling path
- **Input adaptation**: Modified conv1 for 9 channels
- **Output**: 7-class logits
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam

### Augmentation
- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)
- Random 90° rotation (p=0.5)
- Shift/Scale/Rotate (enhanced version)

## 🎓 Inspiration & References

This implementation draws inspiration from:

1. **[pavlo-seimskyi/semantic-segmentation-satellite-imagery](https://github.com/pavlo-seimskyi/semantic-segmentation-satellite-imagery)**
   - U-Net + ResNet architecture
   - Discriminative learning rates
   - Progressive unfreezing strategy
   - Learning rate finder

2. **[souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch](https://github.com/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch)**
   - Configuration-driven approach
   - Per-class evaluation metrics
   - Promptable segmentation concept
   - LandCover.ai dataset handling

### Key Adaptations:
- ✅ Multi-channel input (9 bands) adapted from 3-channel RGB
- ✅ Automated spectral-based labeling for Mauritius
- ✅ Google Earth Engine integration
- ✅ Live web interface with Leaflet.js
- ✅ Discriminative LR (encoder: 1e-5, decoder: 1e-4)
- ✅ Per-class IoU and F1 metrics
- ✅ LR scheduling with ReduceLROnPlateau

## 🔮 Future Enhancements

1. **Temporal Analysis**
   - Compare 2015 vs 2024 imagery
   - Change detection algorithms
   - Time-series classification

2. **Model Improvements**
   - Attention mechanisms (U-Net++)
   - Multi-scale feature fusion
   - Test DeepLabV3+ architecture

3. **Data Enhancements**
   - More training tiles (currently 32)
   - Manual label refinement
   - Class balancing strategies

4. **Deployment**
   - Docker containerization
   - REST API for predictions
   - Cloud deployment (AWS/GCP)

## 📝 License

MIT License - See LICENSE file for details

## 👤 Author

**Lloyd Florens**
PhD Research Project - Mauritius Land Cover Classification

---

## 🆘 Troubleshooting

### Google Earth Engine Issues
```bash
# Authenticate
earthengine authenticate

# Set project
earthengine set_project YOUR_PROJECT_ID
```

### Training Memory Issues
Reduce batch size:
```bash
python train_model_pretrained.py --batch-size 1
```

### Web Viewer Not Updating
Restart the Flask server:
```bash
# Stop with Ctrl+C
# Restart
python src/web/live_map_app.py
```

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---
**Last Updated**: December 2024
**Status**: ✅ Functional - Model trained and deployed
