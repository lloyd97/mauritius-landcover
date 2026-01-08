# Mauritius Land Cover Analysis - Quick Start Guide

## ✅ Project Status: RUNNING!

Your Mauritius Land Cover Analysis system is now up and running!

## 🌐 Access the Web Interface

**Open your browser and visit:** http://localhost:5000

## 🎯 What You Can Do

The web interface has 3 main tabs:

### 1. **Classification Tab**
   - Click "Load Demo Data" to see a sample land cover classification
   - View color-coded land cover map with:
     - 🔲 **Roads** (Grey)
     - 💧 **Water/Rivers** (Blue)
     - 🌲 **Forest** (Dark Green)
     - 🌾 **Plantation/Sugar Cane** (Light Green)
     - 🏢 **Buildings** (Brown)
     - 🏜️ **Bare Land** (Tan)
   - See area statistics in hectares

### 2. **Change Detection Tab**
   - Select two time periods (e.g., 2015 → 2024)
   - Click "Analyze Changes"
   - See before/after maps and change statistics
   - Red areas show where land cover changed

### 3. **Time Series Tab**
   - View land cover changes over time (2015-2025)
   - Interactive chart showing area trends

## 📊 API Endpoints

- **GET /api/demo** - Get sample classification
- **POST /api/change_detection** - Analyze changes between years
- **GET /api/time_series** - Get time series data
- **GET /api/export/{format}** - Export results (geotiff, json, csv)

## 🔧 Project Architecture

```
┌─────────────────────────────────────────────────────┐
│        MAURITIUS LAND COVER ANALYZER                │
├─────────────────────────────────────────────────────┤
│                                                      │
│   Sentinel-2        Sentinel-2        Sentinel-2    │
│   2015-2017   →     2019-2021   →     2023-2025     │
│                          │                          │
│                          ▼                          │
│                   U-Net + LSTM                      │
│                   Deep Learning                     │
│                          │                          │
│                          ▼                          │
│              LAND COVER CLASSIFICATION              │
│   Roads │ Water │ Forest │ Crops │ Buildings       │
│   Grey  │ Blue  │ DkGreen│ Green │ Brown           │
│                          │                          │
│                          ▼                          │
│              CHANGE DETECTION MAP                   │
│              "What changed & when"                  │
└─────────────────────────────────────────────────────┘
```

## 📁 Project Structure

- `src/data/gee_download.py` - Download Sentinel-2 imagery
- `src/models/unet.py` - U-Net architecture
- `src/models/lstm_unet.py` - Temporal LSTM models
- `src/utils/visualization.py` - Color mapping & visualization
- `src/utils/change_detection.py` - Change analysis
- `src/web/app.py` - Web interface (currently running)
- `configs/config.yaml` - Configuration

## 🚀 Next Steps

### To Use Real Satellite Data:

1. **Set up Google Earth Engine:**
   ```bash
   py -m pip install earthengine-api geemap
   earthengine authenticate
   ```

2. **Download Sentinel-2 data:**
   ```bash
   py src/data/gee_download.py --config configs/config.yaml
   ```

3. **Train the model:**
   ```bash
   py src/models/train.py --config configs/config.yaml
   ```

### To Stop the Server:

Press `Ctrl+C` in the terminal or use:
```bash
# Find and kill the process
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *app.py*"
```

## 📝 Configuration

Edit `configs/config.yaml` to:
- Change time periods
- Adjust model architecture
- Modify training parameters
- Update class definitions

## 🎓 Research Context

This is a PhD research project for analyzing land cover changes in Mauritius over a 10-year period (2015-2025) using:
- Sentinel-2 satellite imagery (10m resolution)
- Deep learning (U-Net with ResNet50 encoder)
- Temporal analysis (LSTM for multi-temporal data)
- Post-classification change detection

## 📊 Current Demo Features

The demo shows synthetic data demonstrating:
- ✅ Color-coded land cover classification
- ✅ 10-year change detection (2015 → 2024)
- ✅ Statistical analysis
- ✅ Interactive visualization
- ✅ Export capabilities

Enjoy exploring your Mauritius Land Cover Analysis system! 🇲🇺
