# Google Earth Engine - Final Setup

## Current Status
✅ GEE is authenticated
❌ Need valid project ID

## Quick Solution - Create GEE Cloud Project (5 minutes)

### Step 1: Register for Earth Engine
1. Go to: https://signup.earthengine.google.com/
2. Click "Register a Noncommercial or Commercial Cloud project"
3. Select: **Noncommercial** (for academic/research use)
4. Fill in:
   - **Organization**: Your University/PhD Program
   - **Project**: "Mauritius Land Cover Classification"
   - **Use case**: "PhD research - land cover change detection"
5. Submit and wait for approval (usually instant)

### Step 2: Get Your Project ID
After approval:
1. Go to: https://console.cloud.google.com/
2. Look at the top - you'll see your project name/ID
3. Or go to: https://code.earthengine.google.com/
4. The project ID will be shown

### Step 3: Set the Project
```bash
earthengine set_project YOUR_PROJECT_ID_HERE
```

Example:
```bash
earthengine set_project ee-mauritiusresearch
# or
earthengine set_project phd-landcover-12345
```

---

## Alternative Option - Use Existing Project

If you already have a Google Cloud project with Earth Engine enabled:

### Check Your Projects:
1. Go to: https://console.cloud.google.com/
2. Click project dropdown at top
3. See list of your projects
4. Copy the Project ID (not the name!)

### Enable Earth Engine API:
1. Go to: https://console.cloud.google.com/apis/library/earthengine.googleapis.com
2. Click "Enable"
3. Wait 1-2 minutes

### Set the Project:
```bash
earthengine set_project YOUR_ACTUAL_PROJECT_ID
```

---

## Once Project is Set

### Test it works:
```bash
python -c "import ee; ee.Initialize(project='YOUR_PROJECT_ID'); print('SUCCESS!')"
```

### Download Mauritius data:
```bash
cd scripts
python create_training_dataset.py --download --output-dir ../data/training
```

---

## What Happens Next

Once GEE is working, the script will:
1. Download Sentinel-2 imagery from 8 locations in Mauritius
2. Each download takes ~2-3 minutes
3. Total: ~20-25 minutes for all data
4. You'll get real satellite imagery for training!

---

## If You Don't Want to Wait

### Option: Skip GEE for Now

You can train on synthetic/demo data to test the pipeline:

```bash
# Create sample data
cd scripts
python -c "
import numpy as np
from pathlib import Path

tiles_dir = Path('../data/training/tiles')
tiles_dir.mkdir(parents=True, exist_ok=True)

for i in range(30):
    tile = np.random.rand(9, 256, 256).astype(np.float32)
    mask = np.random.randint(0, 7, (256, 256), dtype=np.int64)
    np.save(tiles_dir / f'demo_tile_{i:04d}.npy', tile)
    np.save(tiles_dir / f'demo_mask_{i:04d}.npy', mask)

print('Created 30 demo tiles')
"

# Train on demo data
python train_model_pretrained.py --epochs 10 --batch-size 4
```

This tests the full pipeline without GEE!

---

## Summary

**Path A: Get Real Data (Recommended)**
1. Register at https://signup.earthengine.google.com/
2. Get project ID
3. Run: `earthengine set_project PROJECT_ID`
4. Download real Mauritius Sentinel-2 data

**Path B: Test Pipeline First**
1. Create demo data (script above)
2. Test training works
3. Set up GEE later for real data

---

**Which path do you want to take?**
