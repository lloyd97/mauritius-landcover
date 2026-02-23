"""Quick analysis of spectral distribution in cached tiles to find correct NDBI/NDVI thresholds."""
import numpy as np
import torch
import segmentation_models_pytorch as smp
from pathlib import Path

# Load model
MODEL = smp.Unet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=7, activation=None)
first_conv = MODEL.encoder.conv1
MODEL.encoder.conv1 = torch.nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    MODEL.encoder.conv1.weight[:, :3, :, :] = first_conv.weight
    MODEL.encoder.conv1.weight[:, 3:, :, :] = first_conv.weight[:, :6, :, :] if first_conv.weight.shape[1] >= 6 else first_conv.weight.repeat(1, 3, 1, 1)[:, :6, :, :]

ckpt = torch.load('checkpoints/enhanced_model_old7class.pth', map_location='cpu', weights_only=False)
if 'model_state_dict' in ckpt:
    MODEL.load_state_dict(ckpt['model_state_dict'])
else:
    MODEL.load_state_dict(ckpt)
MODEL.eval()

# Load land mask result from ocean mask (we'll use all tiles, skip pure ocean ones)
tile_dir = Path('data/classification_cache/raw_tiles/2019')
tiles = sorted(tile_dir.glob('tile_*.npy'))

all_ndvi = []
all_ndbi = []
all_ndwi = []
all_brightness = []
all_model_class = []

print(f"Analyzing {len(tiles)} tiles...")
for i, tp in enumerate(tiles):
    bands = np.load(tp)  # (9, 256, 256)

    ndvi = bands[6]
    ndwi = bands[7]
    ndbi = bands[8]

    # Normalize if needed
    if np.abs(ndvi).max() > 1.5:
        ndvi = np.clip(ndvi / 10000.0, -1, 1)
    if np.abs(ndwi).max() > 1.5:
        ndwi = np.clip(ndwi / 10000.0, -1, 1)
    if np.abs(ndbi).max() > 1.5:
        ndbi = np.clip(ndbi / 10000.0, -1, 1)

    b, g, r = bands[0].copy(), bands[1].copy(), bands[2].copy()
    if b.max() > 100:
        b, g, r = b / 10000.0, g / 10000.0, r / 10000.0
    brightness = (b + g + r) / 3.0

    # Run model
    input_tensor = torch.from_numpy(bands).float().unsqueeze(0)
    with torch.no_grad():
        output = MODEL(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).numpy()

    # Only keep land pixels (skip pure water tiles)
    water_ratio = np.sum(pred == 1) / pred.size
    if water_ratio > 0.9:
        continue  # Skip ocean tiles

    # Flatten and store
    all_ndvi.append(ndvi.flatten())
    all_ndbi.append(ndbi.flatten())
    all_ndwi.append(ndwi.flatten())
    all_brightness.append(brightness.flatten())
    all_model_class.append(pred.flatten())

all_ndvi = np.concatenate(all_ndvi)
all_ndbi = np.concatenate(all_ndbi)
all_ndwi = np.concatenate(all_ndwi)
all_brightness = np.concatenate(all_brightness)
all_model_class = np.concatenate(all_model_class)

print(f"\nTotal pixels analyzed: {len(all_ndvi):,}")
print(f"Model class distribution:")
for c in range(7):
    count = np.sum(all_model_class == c)
    pct = count / len(all_model_class) * 100
    print(f"  Class {c}: {count:>10,} ({pct:.1f}%)")

# Focus on model class 2 (dominant vegetation) - this is what we need to split
mod2_mask = all_model_class == 2
mod2_ndvi = all_ndvi[mod2_mask]
mod2_ndbi = all_ndbi[mod2_mask]
mod2_brightness = all_brightness[mod2_mask]

print(f"\n=== Model Class 2 (Vegetation) Analysis ===")
print(f"Total mod2 pixels: {np.sum(mod2_mask):,} ({np.sum(mod2_mask)/len(all_model_class)*100:.1f}%)")
print(f"\nNDVI distribution (model class 2):")
for p in [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]:
    print(f"  P{p}: {np.percentile(mod2_ndvi, p):.4f}")

print(f"\nNDBI distribution (model class 2):")
for p in [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]:
    print(f"  P{p}: {np.percentile(mod2_ndbi, p):.4f}")

print(f"\nBrightness distribution (model class 2):")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  P{p}: {np.percentile(mod2_brightness, p):.4f}")

# Target: Forest ~27%, Wasteland ~26%, Plantation ~24%, Urban ~12%, Roads ~7%, Bare Land ~4%
# Since mod2 is the dominant class, we need to find thresholds that split it right
# Let's compute what % of mod2 falls into different NDBI/NDVI ranges

print(f"\n=== Threshold Analysis for Model Class 2 ===")
print(f"(Percentages are of LAND pixels, not just mod2)")

# Estimate non-water pixels
land_mask = all_model_class != 1  # Exclude water
land_total = np.sum(land_mask)

# Different NDBI thresholds for Forest
for ndbi_thresh in [-0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0]:
    forest_count = np.sum(mod2_mask & (all_ndvi >= 0.4) & (all_ndbi < ndbi_thresh))
    print(f"  NDBI < {ndbi_thresh:+.2f} & NDVI>=0.4 (Forest): {forest_count/land_total*100:.1f}%")

print()
for ndbi_thresh in [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05]:
    urban_count = np.sum(mod2_mask & (all_ndbi >= ndbi_thresh))
    print(f"  NDBI >= {ndbi_thresh:+.2f} (Urban): {urban_count/land_total*100:.1f}%")

print()
for ndvi_thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
    low_count = np.sum(mod2_mask & (all_ndvi < ndvi_thresh))
    print(f"  NDVI < {ndvi_thresh:.2f} (Low veg): {low_count/land_total*100:.1f}%")

# Also check model class 3
mod3_mask = all_model_class == 3
print(f"\n=== Model Class 3 Analysis ===")
print(f"Total mod3 pixels: {np.sum(mod3_mask):,} ({np.sum(mod3_mask)/len(all_model_class)*100:.1f}%)")
if np.sum(mod3_mask) > 0:
    print(f"NDVI range: {all_ndvi[mod3_mask].min():.4f} to {all_ndvi[mod3_mask].max():.4f}")
    print(f"NDBI range: {all_ndbi[mod3_mask].min():.4f} to {all_ndbi[mod3_mask].max():.4f}")

# Check model class 4 (low veg / plantation)
mod4_mask = all_model_class == 4
print(f"\n=== Model Class 4 Analysis ===")
print(f"Total mod4 pixels: {np.sum(mod4_mask):,} ({np.sum(mod4_mask)/len(all_model_class)*100:.1f}%)")

# Check model class 5 (Urban)
mod5_mask = all_model_class == 5
print(f"\n=== Model Class 5 (Urban) Analysis ===")
print(f"Total mod5 pixels: {np.sum(mod5_mask):,} ({np.sum(mod5_mask)/len(all_model_class)*100:.1f}%)")

print("\nDone!")
