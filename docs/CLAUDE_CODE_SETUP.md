# 🚀 Setting Up with Claude Code in VS Code Insiders

This guide helps you continue development of the Mauritius Land Cover Classification System using Claude Code in VS Code Insiders.

## Prerequisites

1. **VS Code Insiders** - Download from [code.visualstudio.com/insiders](https://code.visualstudio.com/insiders/)
2. **Claude Code Extension** - Install from VS Code marketplace
3. **Python 3.10+** - Install via [python.org](https://python.org) or conda
4. **CUDA** (optional) - For GPU acceleration

## Quick Setup

### Step 1: Clone/Download the Project

```bash
# If using Git
git clone https://github.com/yourusername/mauritius-landcover.git
cd mauritius-landcover

# Or extract the downloaded ZIP
unzip mauritius-landcover.zip
cd mauritius-landcover
```

### Step 2: Create Environment

```bash
# Using conda (recommended)
conda create -n landcover python=3.10
conda activate landcover

# Install GDAL (easier via conda)
conda install -c conda-forge gdal rasterio

# Install PyTorch with CUDA (if you have GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only
pip install torch torchvision

# Install project dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Step 3: Open in VS Code Insiders

```bash
code-insiders .
```

### Step 4: Using Claude Code

With Claude Code extension installed, you can:

1. **Ask Claude to help with code**
   - Press `Cmd/Ctrl + Shift + P` → "Claude: Ask Claude"
   - Or use the Claude sidebar

2. **Generate code snippets**
   - Highlight code and ask Claude to improve/fix it

3. **Get explanations**
   - Ask Claude to explain any part of the codebase

## Development Tasks to Continue

### 🔴 High Priority

1. **Set up Google Earth Engine authentication**
   ```bash
   earthengine authenticate
   ```
   Ask Claude: "Help me set up Google Earth Engine authentication and download Mauritius data"

2. **Create training labels**
   Ask Claude: "Generate training labels for Mauritius using OpenStreetMap data"

3. **Test the training pipeline**
   ```bash
   python src/models/train.py --config configs/config.yaml
   ```

### 🟡 Medium Priority

4. **Improve the web interface**
   Ask Claude: "Add a map viewer using Leaflet to the web interface"

5. **Add model export**
   Ask Claude: "Add ONNX export functionality for model deployment"

6. **Create evaluation scripts**
   Ask Claude: "Create a comprehensive evaluation script with confusion matrices"

### 🟢 Nice to Have

7. **Add Docker support**
   Ask Claude: "Create a Dockerfile for this project"

8. **Set up CI/CD**
   Ask Claude: "Create GitHub Actions workflow for testing"

## Project Structure Reference

```
mauritius-landcover/
├── src/
│   ├── data/
│   │   ├── gee_download.py      # ← Start here for data
│   │   ├── dataset.py           # PyTorch datasets
│   │   └── preprocessing.py     # (to create)
│   ├── models/
│   │   ├── unet.py              # U-Net architecture
│   │   ├── lstm_unet.py         # Temporal model
│   │   └── train.py             # Training script
│   ├── utils/
│   │   ├── visualization.py     # Plotting utilities
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── change_detection.py  # Change analysis
│   └── web/
│       └── app.py               # Flask web app
├── configs/
│   └── config.yaml              # Main configuration
└── requirements.txt
```

## Useful Claude Code Prompts

Here are some prompts to ask Claude Code:

### Data Acquisition
- "Download Sentinel-2 imagery for Mauritius from 2015 to 2024 using the gee_download.py script"
- "Create a data preprocessing pipeline that handles cloud masking"
- "Generate synthetic training labels using OpenStreetMap roads and buildings"

### Model Development
- "Add attention mechanisms to the U-Net model"
- "Implement test-time augmentation for better predictions"
- "Add mixed precision training support"

### Evaluation
- "Create a validation script that generates per-class accuracy reports"
- "Add confusion matrix visualization with proper labeling"
- "Implement cross-validation training"

### Web Interface
- "Add a side-by-side comparison slider to the web interface"
- "Create an interactive timeline for temporal analysis"
- "Add GeoJSON export functionality"

### PhD Specific
- "Generate LaTeX tables from the evaluation results"
- "Create publication-quality figures for the methodology section"
- "Write docstrings for all functions following Google style"

## Troubleshooting

### GDAL Installation Issues
```bash
# On Ubuntu
sudo apt-get install gdal-bin libgdal-dev
pip install GDAL==$(gdal-config --version)

# On macOS
brew install gdal
pip install GDAL==$(gdal-config --version)

# On Windows (recommended: use conda)
conda install -c conda-forge gdal
```

### GPU Memory Issues
```python
# In config.yaml, reduce batch size
training:
  batch_size: 8  # or 4
```

### Earth Engine Quota
- Use Copernicus Data Space as alternative
- Download data in smaller chunks

## Contact & Support

For project-specific questions, use Claude Code to get help directly in your IDE!

---

*This project is part of PhD research on land cover change detection in Mauritius.*
