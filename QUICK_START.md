# Quick Start - Get Training Data NOW

## Current Status

✅ **Web viewer running** - http://localhost:5000
✅ **Training scripts created**
✅ **Labeling tool created** - [data/training/labeling_tool.html](data/training/labeling_tool.html)
⏳ **Need:** Google Earth Engine authentication to download real data

---

## Option 1: Authenticate GEE (Recommended - Get Real Data)

### Step 1: Authenticate
```bash
earthengine authenticate
```

This will:
1. Open a browser
2. Ask you to sign in with Google
3. Give you a code to paste back

### Step 2: Download Training Data
```bash
cd scripts
python create_training_dataset.py --download --output-dir ../data/training
```

This downloads Sentinel-2 imagery from 8 locations in Mauritius.

### Step 3: Create Tiles
```bash
python create_training_dataset.py --create-tiles --output-dir ../data/training
```

Creates 256x256 training tiles.

---

## Option 2: Use Demo Data (Quick Test)

Since GEE isn't authenticated, here's how to create sample data to test the pipeline:

### Create Sample Dataset
```bash
cd scripts
python -c "
import numpy as np
from pathlib import Path

# Create sample tiles
tiles_dir = Path('../data/training/tiles')
tiles_dir.mkdir(parents=True, exist_ok=True)

for i in range(10):
    # Create random 9-channel tile
    tile = np.random.rand(9, 256, 256).astype(np.float32)
    np.save(tiles_dir / f'sample_tile_{i:04d}.npy', tile)

    # Create random mask (7 classes)
    mask = np.random.randint(0, 7, (256, 256), dtype=np.int64)
    np.save(tiles_dir / f'sample_mask_{i:04d}.npy', mask)

print('Created 10 sample tiles for testing')
"
```

### Train on Sample Data
```bash
python train_model_pretrained.py \
    --data-dir ../data/training/tiles \
    --epochs 10 \
    --batch-size 4 \
    --output ../checkpoints/test_model.pt
```

This tests that training works!

---

## Option 3: Manual Data Creation (Your Approach!)

Use real satellite imagery you already have:

### If you have Sentinel-2 GeoTIFFs:

```python
import rasterio
import numpy as np
from pathlib import Path

# Load your GeoTIFF
with rasterio.open('your_image.tif') as src:
    data = src.read()  # (bands, height, width)

# Create tiles
tile_size = 256
tiles_dir = Path('data/training/tiles')
tiles_dir.mkdir(parents=True, exist_ok=True)

count = 0
for y in range(0, data.shape[1] - tile_size, tile_size):
    for x in range(0, data.shape[2] - tile_size, tile_size):
        tile = data[:, y:y+tile_size, x:x+tile_size]

        # Save if it has 9 bands (or adapt your data)
        if tile.shape[0] >= 6:
            # Take first 6 bands
            bands_tile = tile[:6]

            # Compute indices (example)
            # You'll need to adapt based on your band order
            ndvi = (bands_tile[3] - bands_tile[2]) / (bands_tile[3] + bands_tile[2] + 1e-6)
            ndwi = (bands_tile[1] - bands_tile[3]) / (bands_tile[1] + bands_tile[3] + 1e-6)
            ndbi = (bands_tile[4] - bands_tile[3]) / (bands_tile[4] + bands_tile[3] + 1e-6)

            # Combine
            full_tile = np.vstack([bands_tile,
                                  ndvi[np.newaxis],
                                  ndwi[np.newaxis],
                                  ndbi[np.newaxis]])

            np.save(tiles_dir / f'tile_{count:04d}.npy', full_tile)
            count += 1

print(f'Created {count} tiles')
```

---

## Labeling Your Data

### Open the Labeling Tool

1. **Open:** [data/training/labeling_tool.html](data/training/labeling_tool.html) in your browser
2. **Load** a tile preview (create PNGs from your tiles)
3. **Paint** the classes:
   - Click class button
   - Paint on image
   - Save mask
4. **Repeat** for 20-30 tiles

---

## Training Pipeline

Once you have labeled data:

```bash
cd scripts

# Train with pre-trained ResNet50
python train_model_pretrained.py \
    --data-dir ../data/training/tiles \
    --architecture unet \
    --encoder resnet50 \
    --epochs 50 \
    --batch-size 8 \
    --output ../checkpoints/best.pt
```

---

## What Each File Does

- **[create_training_dataset.py](scripts/create_training_dataset.py)** - Downloads Sentinel-2 & creates tiles
- **[train_model_pretrained.py](scripts/train_model_pretrained.py)** - Trains model with pre-trained weights
- **[data/training/labeling_tool.html](data/training/labeling_tool.html)** - Interactive labeling interface
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Detailed instructions
- **[SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md)** - Overview

---

## Recommended Next Steps

### For Quick Results (Today):
1. Authenticate GEE: `earthengine authenticate`
2. Download 2-3 locations
3. Create 10-15 tiles
4. Quick label them
5. Train for 10 epochs
6. Test in web viewer!

### For Best Results (This Week):
1. Authenticate GEE
2. Download all 8 locations
3. Create 100+ tiles
4. Carefully label 30-50
5. Train for 50 epochs
6. Publication-quality results!

---

## Files in Your Workspace

```
mauritius-landcover/
├── data/
│   └── training/
│       ├── labeling_tool.html  ← Open this to label!
│       └── tiles/              ← Put your data here
│
├── scripts/
│   ├── create_training_dataset.py  ← Run this
│   └── train_model_pretrained.py   ← Then this
│
├── checkpoints/                ← Models save here
│
├── QUICK_START.md             ← You are here!
├── TRAINING_GUIDE.md          ← Detailed guide
└── SOLUTION_SUMMARY.md        ← Overview
```

---

## Need Help?

**GEE Authentication:**
```bash
earthengine authenticate
```

**Create Sample Data:**
```bash
cd scripts
python create_training_dataset.py --help
```

**Train Model:**
```bash
python train_model_pretrained.py --help
```

---

## The Bottom Line

**You have 3 choices:**

1. **Authenticate GEE** → Download real Sentinel-2 → Best results
2. **Use demo data** → Test the pipeline → Quick verification
3. **Use your own data** → Manual processing → Custom control

**Which do you want to do first?**
