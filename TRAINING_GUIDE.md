# Complete Training Guide for Mauritius Land Cover Classification

## The Problem You Identified

You're absolutely correct! The issues are:

1. **Model isn't trained** → Random predictions
2. **No training data** → Need to create labeled dataset
3. **Need pre-trained weights** → Better starting point

## Solution: Complete Training Pipeline

### Step-by-Step Workflow

```
1. Download Sentinel-2 imagery from Mauritius
   ↓
2. Create training tiles
   ↓
3. Label the data (manual or semi-automatic)
   ↓
4. Train U-Net with ImageNet pre-trained encoder
   ↓
5. Fine-tune on Mauritius-specific data
   ↓
6. Deploy in web viewer
```

---

## Step 1: Authenticate Google Earth Engine

First, set up access to download real Sentinel-2 data:

```bash
earthengine authenticate
```

Follow the prompts:
- Browser will open
- Sign in with Google account
- Copy verification code
- Paste in terminal

---

## Step 2: Download Training Data

Download Sentinel-2 imagery from different areas of Mauritius:

```bash
cd scripts
python create_training_dataset.py --download --output-dir ../data/training
```

This downloads tiles from:
- Port Louis (urban)
- Black River (coastal)
- Grand Bassin (forest)
- Pamplemousses (agricultural)
- Curepipe (urban)
- And more...

Each area provides different land cover types for training.

---

## Step 3: Create Training Tiles

Split the large images into 256x256 training tiles:

```bash
python create_training_dataset.py --create-tiles --output-dir ../data/training
```

This creates:
- `data/training/tiles/*.npy` - Training data (9 bands)
- `data/training/tiles/*preview.png` - RGB previews for labeling

---

## Step 4: Label the Data

### Option A: Manual Labeling (Recommended for Start)

1. **Create the labeling tool:**
   ```bash
   python create_training_dataset.py --create-labeling-tool --output-dir ../data/training
   ```

2. **Open the tool:**
   Open `data/training/labeling_tool.html` in your browser

3. **Label tiles:**
   - Load a tile preview image
   - Select class (Roads, Water, Forest, Plantation, Buildings, Bare Land)
   - Paint on the image
   - Save mask
   - Repeat for 20-30 tiles minimum

4. **Save masks:**
   Save each mask as `*_mask_*.npy` in the same directory

### Option B: Semi-Automatic Labeling

Use color-based segmentation (like your web.py approach) to pre-label, then manually correct:

1. Create initial masks using HSV thresholding
2. Manually refine in labeling tool
3. Saves time while maintaining quality

---

## Step 5: Train the Model

Now train with pre-trained weights:

```bash
python train_model_pretrained.py \
    --data-dir ../data/training/tiles \
    --architecture unet \
    --encoder resnet50 \
    --epochs 50 \
    --batch-size 8 \
    --output ../checkpoints/best.pt
```

### What This Does:

1. **Loads ImageNet pre-trained ResNet50** encoder
2. **Adapts it for 9 input channels** (6 Sentinel-2 bands + 3 indices)
3. **Fine-tunes on your Mauritius data**
4. **Saves best model** to checkpoints/

### Training Options:

**Architectures:**
- `unet` - Standard U-Net (good balance)
- `unetplusplus` - U-Net++ (better but slower)
- `deeplabv3` - DeepLabV3 (great for semantic segmentation)
- `deeplabv3plus` - DeepLabV3+ (best quality, slower)

**Encoders:**
- `resnet50` - Good balance (recommended)
- `resnet101` - More capacity
- `efficientnet-b0` - Efficient, fast
- `efficientnet-b4` - Better quality

---

## Step 6: Test the Trained Model

The web viewer will automatically load your trained model:

```bash
cd src/web
python live_map_app.py
```

Visit: http://localhost:5000

Now when you click on the map:
- Downloads real Sentinel-2 imagery for that location
- Runs through YOUR trained model
- Shows accurate land cover classification!

---

## Quick Start (Minimal Dataset)

If you want to see results quickly:

### 1. Create minimal dataset (20 tiles):

```bash
# Download 2-3 locations
python scripts/create_training_dataset.py --download

# Create tiles
python scripts/create_training_dataset.py --create-tiles
```

### 2. Quick manual labeling:

- Label just 20 representative tiles
- Mix of urban, forest, agricultural, coastal
- 10-15 minutes per tile
- Total: 3-4 hours of labeling

### 3. Train:

```bash
python scripts/train_model_pretrained.py --epochs 30 --batch-size 4
```

### 4. Test:

```bash
cd src/web && python live_map_app.py
```

---

## Expected Results

### With Minimal Training (20 tiles, 30 epochs):
- **Accuracy**: 60-70%
- **Good at**: Major classes (water, forest, urban)
- **Weak at**: Roads, subtle differences

### With Good Training (100+ tiles, 50 epochs):
- **Accuracy**: 80-90%
- **Good at**: All classes
- **Weak at**: Edge cases, mixed pixels

### With Extensive Training (500+ tiles, 100 epochs):
- **Accuracy**: 90-95%
- **Publication quality**
- **Good generalization** across Mauritius

---

## Improving Results

### 1. Data Augmentation (Already included):
- Random flips, rotations
- Color jittering
- Gaussian noise/blur

### 2. Class Balancing:
- Ensure equal representation of all classes
- Use weighted loss if imbalanced

### 3. Multi-Scale Training:
- Train on different tile sizes
- Better at capturing context

### 4. Ensemble:
- Train multiple models
- Average predictions

---

## File Structure

```
mauritius-landcover/
├── data/
│   └── training/
│       ├── Port_Louis_sentinel2.tif
│       ├── tiles/
│       │   ├── tile_0001.npy
│       │   ├── tile_0001.png (preview)
│       │   ├── mask_0001.npy
│       │   └── ...
│       └── labeling_tool.html
│
├── checkpoints/
│   └── best.pt  ← Your trained model!
│
├── scripts/
│   ├── create_training_dataset.py
│   └── train_model_pretrained.py
│
└── src/
    └── web/
        └── live_map_app.py  ← Uses your model!
```

---

## Troubleshooting

### "No Google Earth Engine access"
```bash
earthengine authenticate
```

### "Not enough training data"
- Label at least 20 diverse tiles
- Mix different land cover types

### "Training is slow"
- Reduce batch size: `--batch-size 4`
- Use smaller encoder: `--encoder resnet34`
- Use CPU if GPU has issues

### "Poor accuracy"
- Label more data (aim for 50+ tiles)
- Train longer (50-100 epochs)
- Check label quality
- Try different architecture: `--architecture deeplabv3plus`

---

## Next Steps

1. **Authenticate GEE** (if not done)
2. **Download data** for 3-5 locations
3. **Label 20-30 tiles** manually
4. **Train initial model** (30 epochs)
5. **Test in web viewer**
6. **Iterate**: Add more data where model fails

---

## Summary

The key insight you had is correct:

✅ Use **pre-trained weights** (ImageNet/EuroSAT)
✅ Create **training dataset** from real Mauritius imagery
✅ **Fine-tune** on your specific classes
✅ Keep the **same web interface** layout

This approach gives you:
- Better starting point (pre-trained weights)
- Mauritius-specific accuracy
- Your custom color scheme
- Side-by-side visualization

Let me know which step you'd like to start with!
