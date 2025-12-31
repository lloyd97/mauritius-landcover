"""
Automated labeling for Mauritius land cover tiles
Uses rule-based classification with spectral indices to create training masks
"""

import numpy as np
import os
from pathlib import Path
from scipy import ndimage
import argparse

# Land cover class mapping
CLASSES = {
    'background': 0,
    'water': 1,          # Blue
    'forest': 2,         # Dark Green
    'plantation': 3,     # Light Green
    'urban': 4,          # Brown (buildings)
    'roads': 5,          # Grey
    'bare_land': 6       # Tan
}

def calculate_indices(tile_data):
    """
    Calculate spectral indices from Sentinel-2 bands

    Input: tile_data of shape (C, H, W) = (9, 256, 256)
    Input bands (9 channels):
    0: B2 (Blue)
    1: B3 (Green)
    2: B4 (Red)
    3: B8 (NIR)
    4: B11 (SWIR1)
    5: B12 (SWIR2)
    6: NDVI
    7: NDWI
    8: NDBI
    """
    indices = {}

    # Extract bands - tile_data is (C, H, W)
    blue = tile_data[0, :, :]
    green = tile_data[1, :, :]
    red = tile_data[2, :, :]
    nir = tile_data[3, :, :]
    swir1 = tile_data[4, :, :]
    swir2 = tile_data[5, :, :]

    # Pre-calculated indices
    indices['ndvi'] = tile_data[6, :, :]  # Already calculated
    indices['ndwi'] = tile_data[7, :, :]  # Already calculated
    indices['ndbi'] = tile_data[8, :, :]  # Already calculated

    # Additional indices
    # MNDWI (Modified NDWI) - better for water
    indices['mndwi'] = (green - swir1) / (green + swir1 + 1e-8)

    # EVI (Enhanced Vegetation Index)
    indices['evi'] = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

    # BSI (Bare Soil Index)
    indices['bsi'] = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-8)

    # UI (Urban Index)
    indices['ui'] = (swir2 - nir) / (swir2 + nir + 1e-8)

    # Brightness
    indices['brightness'] = (blue + green + red + nir) / 4.0

    return indices, {'blue': blue, 'green': green, 'red': red, 'nir': nir, 'swir1': swir1, 'swir2': swir2}

def classify_tile(tile_data):
    """
    Rule-based classification using spectral indices
    Returns a mask with class labels

    Args:
        tile_data: np.array of shape (C, H, W) = (9, 256, 256)
    Returns:
        mask: np.array of shape (H, W) = (256, 256)
    """
    C, H, W = tile_data.shape  # (9, 256, 256)
    mask = np.zeros((H, W), dtype=np.uint8)

    # Calculate indices
    indices, bands = calculate_indices(tile_data)

    ndvi = indices['ndvi']
    ndwi = indices['ndwi']
    ndbi = indices['ndbi']
    mndwi = indices['mndwi']
    evi = indices['evi']
    bsi = indices['bsi']
    ui = indices['ui']
    brightness = indices['brightness']

    nir = bands['nir']
    red = bands['red']
    swir1 = bands['swir1']

    # Normalize brightness for thresholding
    brightness_norm = (brightness - brightness.min()) / (brightness.max() - brightness.min() + 1e-8)

    # Priority-based classification (order matters!)

    # 1. WATER - highest priority (very distinctive)
    # Water has high NDWI/MNDWI, low NIR, low NDVI
    water_mask = (
        ((mndwi > 0.3) | (ndwi > 0.2)) &
        (ndvi < 0.3) &
        (nir < 0.15)
    )
    mask[water_mask] = CLASSES['water']

    # 2. DENSE FOREST - high NDVI, high NIR, moderate EVI
    # Dense vegetation with strong NIR response
    forest_mask = (
        (ndvi > 0.6) &
        (evi > 0.4) &
        (nir > 0.25) &
        (ndbi < 0.0) &
        (mask == 0)  # Not already classified
    )
    mask[forest_mask] = CLASSES['forest']

    # 3. PLANTATION/AGRICULTURAL - moderate NDVI, organized patterns
    # Less dense than forest, more uniform
    plantation_mask = (
        (ndvi > 0.35) &
        (ndvi <= 0.6) &
        (evi > 0.2) &
        (nir > 0.15) &
        (mask == 0)
    )
    mask[plantation_mask] = CLASSES['plantation']

    # 4. URBAN/BUILDINGS - high NDBI, low NDVI, high SWIR
    # Built-up areas with strong SWIR response
    urban_mask = (
        (ndbi > 0.1) &
        (ndvi < 0.3) &
        (ui > 0.0) &
        (swir1 > 0.15) &
        (mask == 0)
    )
    mask[urban_mask] = CLASSES['urban']

    # 5. ROADS - linear features, moderate brightness, low NDVI
    # Similar to urban but more linear and less bright
    roads_mask = (
        (ndvi < 0.2) &
        (ndbi > -0.1) &
        (brightness_norm > 0.3) &
        (brightness_norm < 0.7) &
        (mask == 0)
    )
    # Apply morphological operations to detect linear features
    roads_cleaned = detect_linear_features(roads_mask)
    mask[roads_cleaned] = CLASSES['roads']

    # 6. BARE LAND - high BSI, low NDVI, low water indices
    bare_mask = (
        (bsi > 0.0) &
        (ndvi < 0.25) &
        (mndwi < 0.0) &
        (mask == 0)
    )
    mask[bare_mask] = CLASSES['bare_land']

    # Apply smoothing to reduce noise
    mask = smooth_mask(mask)

    return mask

def detect_linear_features(binary_mask, min_length=10):
    """
    Detect linear features (roads) using morphological operations
    """
    if not binary_mask.any():
        return binary_mask

    # Use line structuring elements
    h_line = np.zeros((1, min_length), dtype=np.uint8)
    h_line[0, :] = 1
    v_line = np.zeros((min_length, 1), dtype=np.uint8)
    v_line[:, 0] = 1

    # Detect horizontal and vertical lines
    h_lines = ndimage.binary_opening(binary_mask, structure=h_line)
    v_lines = ndimage.binary_opening(binary_mask, structure=v_line)

    # Combine
    linear_features = h_lines | v_lines

    return linear_features

def smooth_mask(mask, kernel_size=3):
    """
    Smooth mask using majority filter to reduce noise
    """
    from scipy.ndimage import generic_filter

    def majority(values):
        """Return most common value"""
        unique, counts = np.unique(values, return_counts=True)
        return unique[np.argmax(counts)]

    # Apply majority filter
    smoothed = generic_filter(mask, majority, size=kernel_size, mode='constant', cval=0)

    return smoothed.astype(np.uint8)

def label_all_tiles(tiles_dir, output_dir=None):
    """
    Automatically label all tiles in the directory
    """
    tiles_dir = Path(tiles_dir)

    if output_dir is None:
        output_dir = tiles_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .npy tile files (not masks)
    tile_files = [f for f in tiles_dir.glob('*.npy') if '_mask' not in f.name]

    print(f"============================================================")
    print(f"Automated Tile Labeling")
    print(f"============================================================")
    print(f"Found {len(tile_files)} tiles to label\n")

    labeled_count = 0

    for tile_file in sorted(tile_files):
        print(f"Processing: {tile_file.name}")

        # Load tile data
        tile_data = np.load(tile_file)

        # Generate mask
        mask = classify_tile(tile_data)

        # Create mask filename
        mask_name = tile_file.stem + '_mask.npy'
        mask_path = output_dir / mask_name

        # Save mask
        np.save(mask_path, mask)

        # Print class statistics
        unique, counts = np.unique(mask, return_counts=True)
        class_names = {v: k for k, v in CLASSES.items()}

        print(f"  Classes found:")
        for class_id, count in zip(unique, counts):
            class_name = class_names.get(class_id, 'unknown')
            percentage = (count / mask.size) * 100
            print(f"    - {class_name}: {percentage:.1f}%")

        print(f"  Saved: {mask_path.name}\n")
        labeled_count += 1

    print(f"============================================================")
    print(f"Labeling Complete!")
    print(f"============================================================")
    print(f"Labeled {labeled_count} tiles")
    print(f"Masks saved to: {output_dir}")
    print(f"\nNext step: Train the model using:")
    print(f"  py scripts/train_model_pretrained.py --data-dir {output_dir}")

    return labeled_count

def main():
    parser = argparse.ArgumentParser(description='Automatically label land cover tiles')
    parser.add_argument('--tiles-dir', type=str,
                       default='data/training/tiles',
                       help='Directory containing tile .npy files')
    parser.add_argument('--output-dir', type=str,
                       default=None,
                       help='Output directory for masks (default: same as tiles-dir)')

    args = parser.parse_args()

    label_all_tiles(args.tiles_dir, args.output_dir)

if __name__ == '__main__':
    main()
