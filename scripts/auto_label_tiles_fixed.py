"""
Automated labeling for Mauritius land cover tiles
Uses rule-based classification with spectral indices to create training masks
FIXED: Thresholds corrected for raw Sentinel-2 values (0-10000 range)
"""

import numpy as np
import os
from pathlib import Path
from scipy import ndimage
import argparse

# Land cover class mapping
CLASSES = {
    'background': 0,
    'water': 1,
    'forest': 2,
    'plantation': 3,
    'urban': 4,
    'roads': 5,
    'bare_land': 6
}

def calculate_indices(tile_data):
    indices = {}
    blue = tile_data[0, :, :]
    green = tile_data[1, :, :]
    red = tile_data[2, :, :]
    nir = tile_data[3, :, :]
    swir1 = tile_data[4, :, :]
    swir2 = tile_data[5, :, :]

    indices['ndvi'] = tile_data[6, :, :]
    indices['ndwi'] = tile_data[7, :, :]
    indices['ndbi'] = tile_data[8, :, :]
    indices['mndwi'] = (green - swir1) / (green + swir1 + 1e-8)
    indices['evi'] = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    indices['bsi'] = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-8)
    indices['ui'] = (swir2 - nir) / (swir2 + nir + 1e-8)
    indices['brightness'] = (blue + green + red + nir) / 4.0

    return indices, {'blue': blue, 'green': green, 'red': red, 'nir': nir, 'swir1': swir1, 'swir2': swir2}

def classify_tile(tile_data):
    C, H, W = tile_data.shape
    mask = np.zeros((H, W), dtype=np.uint8)
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
    swir1 = bands['swir1']

    brightness_norm = (brightness - brightness.min()) / (brightness.max() - brightness.min() + 1e-8)

    # WATER - FIXED threshold: nir < 1500 (raw values)
    water_mask = (
        ((mndwi > 0.2) | (ndwi > 0.1)) &
        (ndvi < 0.3) &
        (nir < 1500)
    )
    mask[water_mask] = CLASSES['water']

    # FOREST - FIXED threshold: nir > 2500
    forest_mask = (
        (ndvi > 0.6) &
        (evi > 0.4) &
        (nir > 2500) &
        (ndbi < 0.0) &
        (mask == 0)
    )
    mask[forest_mask] = CLASSES['forest']

    # PLANTATION - FIXED threshold: nir > 1500
    plantation_mask = (
        (ndvi > 0.35) &
        (ndvi <= 0.6) &
        (evi > 0.2) &
        (nir > 1500) &
        (mask == 0)
    )
    mask[plantation_mask] = CLASSES['plantation']

    # URBAN - FIXED threshold: swir1 > 1500
    urban_mask = (
        (ndbi > 0.1) &
        (ndvi < 0.3) &
        (ui > 0.0) &
        (swir1 > 1500) &
        (mask == 0)
    )
    mask[urban_mask] = CLASSES['urban']

    # ROADS
    roads_mask = (
        (ndvi < 0.2) &
        (ndbi > -0.1) &
        (brightness_norm > 0.3) &
        (brightness_norm < 0.7) &
        (mask == 0)
    )
    mask[roads_mask] = CLASSES['roads']

    # BARE LAND
    bare_mask = (
        (bsi > 0.0) &
        (ndvi < 0.25) &
        (mndwi < 0.0) &
        (mask == 0)
    )
    mask[bare_mask] = CLASSES['bare_land']

    return mask

def label_all_tiles(tiles_dir, output_dir=None):
    tiles_dir = Path(tiles_dir)
    if output_dir is None:
        output_dir = tiles_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    tile_files = [f for f in tiles_dir.glob('*.npy') if '_mask' not in f.name]
    print(f"Found {len(tile_files)} tiles to label")

    labeled_count = 0
    water_detected_count = 0

    for tile_file in sorted(tile_files):
        print(f"Processing: {tile_file.name}")
        tile_data = np.load(tile_file)
        mask = classify_tile(tile_data)

        mask_name = tile_file.stem + '_mask.npy'
        mask_path = output_dir / mask_name
        np.save(mask_path, mask)

        unique, counts = np.unique(mask, return_counts=True)
        class_names = {v: k for k, v in CLASSES.items()}

        print(f"  Classes:")
        for class_id, count in zip(unique, counts):
            class_name = class_names.get(class_id, 'unknown')
            percentage = (count / mask.size) * 100
            print(f"    {class_name}: {percentage:.1f}%")
            if class_name == 'water' and percentage > 0:
                water_detected_count += 1

        labeled_count += 1

    print(f"\nLabeled {labeled_count} tiles")
    print(f"Tiles with water detected: {water_detected_count}")
    return labeled_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles-dir', type=str, default='data/training/tiles')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    label_all_tiles(args.tiles_dir, args.output_dir)

if __name__ == '__main__':
    main()
