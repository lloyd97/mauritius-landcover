"""
Historical Land Cover Change Detection for Mauritius
=====================================================

Compares land cover classification across multiple time periods
to detect and quantify changes (urban expansion, deforestation, etc.)

Time periods: 2016, 2019, 2022, 2025
"""

import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json
import argparse

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from PIL import Image

try:
    import ee
    import requests
    from rasterio.io import MemoryFile
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    print("ERROR: Missing dependencies")
    sys.exit(1)

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mauritius bounds - full island
MAURITIUS_BOUNDS = {
    'min_lon': 57.30,
    'max_lon': 57.82,
    'min_lat': -20.53,
    'max_lat': -19.98
}

# Time periods to analyze
# Note: Sentinel-2 launched June 2015, so we use Landsat for 2010-2014
TIME_PERIODS = [
    {'year': 2010, 'start': '2010-01-01', 'end': '2010-12-31', 'name': '2010', 'satellite': 'landsat'},
    {'year': 2013, 'start': '2013-01-01', 'end': '2013-12-31', 'name': '2013', 'satellite': 'landsat'},
    {'year': 2016, 'start': '2016-01-01', 'end': '2016-12-31', 'name': '2016', 'satellite': 'sentinel'},
    {'year': 2019, 'start': '2019-01-01', 'end': '2019-12-31', 'name': '2019', 'satellite': 'sentinel'},
    {'year': 2022, 'start': '2022-01-01', 'end': '2022-12-31', 'name': '2022', 'satellite': 'sentinel'},
    {'year': 2025, 'start': '2024-06-01', 'end': '2025-01-01', 'name': '2025', 'satellite': 'sentinel'},
]

# Land cover classes - Apple Maps style
CLASSES = {
    0: {'name': 'Background', 'color': [245, 243, 240]},
    1: {'name': 'Water', 'color': [168, 216, 234]},
    2: {'name': 'Forest', 'color': [139, 195, 74]},
    3: {'name': 'Plantation', 'color': [197, 225, 165]},
    4: {'name': 'Urban', 'color': [215, 204, 200]},
    5: {'name': 'Roads', 'color': [158, 158, 158]},
    6: {'name': 'Bare Land', 'color': [239, 235, 233]}
}

# Change types of interest
CHANGE_TYPES = {
    'urbanization': {'from': [2, 3, 6], 'to': 4, 'color': [255, 0, 0]},      # Red - concerning
    'deforestation': {'from': [2], 'to': [3, 4, 5, 6], 'color': [255, 165, 0]},  # Orange - concerning
    'reforestation': {'from': [3, 4, 6], 'to': 2, 'color': [0, 255, 0]},     # Green - positive
    'development': {'from': [6], 'to': [4, 5], 'color': [255, 255, 0]},      # Yellow
}


def load_model():
    """Load trained U-Net model"""
    print("Loading model...")
    model = smp.Unet(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=7,
        activation=None
    )

    # Modify first conv for 9 channels
    first_conv = model.encoder.conv1
    new_conv = nn.Conv2d(
        9, first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None
    )

    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = first_conv.weight[:, :3, :, :]
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
        new_conv.weight[:, 3:, :, :] *= 0.01

    model.encoder.conv1 = new_conv

    # Load checkpoint
    checkpoint_path = Path('checkpoints/enhanced_model.pth')
    if not checkpoint_path.exists():
        checkpoint_path = Path('checkpoints/best_model.pth')

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("ERROR: No model checkpoint found!")
        sys.exit(1)

    model.to(DEVICE)
    model.eval()
    return model


def download_landsat_composite(bounds, start_date, end_date):
    """
    Download cloud-free Landsat composite for a time period (pre-2015)
    Uses Landsat 7 (2010-2012) or Landsat 8 (2013+)

    Returns: numpy array of shape (9, H, W) with bands + indices
    """
    print(f"  Downloading Landsat for {start_date} to {end_date}...")

    geometry = ee.Geometry.Rectangle([
        bounds['min_lon'], bounds['min_lat'],
        bounds['max_lon'], bounds['max_lat']
    ])

    year = int(start_date[:4])

    # Choose Landsat collection based on year
    if year >= 2013:
        # Landsat 8
        collection = 'LANDSAT/LC08/C02/T1_L2'
        bands_map = {'blue': 'SR_B2', 'green': 'SR_B3', 'red': 'SR_B4',
                     'nir': 'SR_B5', 'swir1': 'SR_B6', 'swir2': 'SR_B7'}
        scale_factor = 0.0000275
        offset = -0.2
    else:
        # Landsat 7 (has SLC-off stripes after 2003, but still usable with compositing)
        collection = 'LANDSAT/LE07/C02/T1_L2'
        bands_map = {'blue': 'SR_B1', 'green': 'SR_B2', 'red': 'SR_B3',
                     'nir': 'SR_B4', 'swir1': 'SR_B5', 'swir2': 'SR_B7'}
        scale_factor = 0.0000275
        offset = -0.2

    # Get collection
    landsat = (ee.ImageCollection(collection)
               .filterBounds(geometry)
               .filterDate(start_date, end_date)
               .filter(ee.Filter.lt('CLOUD_COVER', 30)))

    count = landsat.size().getInfo()
    print(f"  Found {count} Landsat images")

    if count == 0:
        # Try with higher cloud tolerance
        landsat = (ee.ImageCollection(collection)
                   .filterBounds(geometry)
                   .filterDate(start_date, end_date)
                   .filter(ee.Filter.lt('CLOUD_COVER', 50)))
        count = landsat.size().getInfo()
        print(f"  Relaxed cloud filter: {count} images")

    if count == 0:
        return None

    # Create median composite and apply scaling
    def apply_scale(image):
        optical = image.select(['SR_B.*']).multiply(scale_factor).add(offset)
        return optical

    composite = landsat.map(apply_scale).median().clip(geometry)

    # Select and rename bands to match our format
    B2 = composite.select(bands_map['blue']).multiply(10000).rename('B2')  # Scale to match Sentinel-2
    B3 = composite.select(bands_map['green']).multiply(10000).rename('B3')
    B4 = composite.select(bands_map['red']).multiply(10000).rename('B4')
    B8 = composite.select(bands_map['nir']).multiply(10000).rename('B8')
    B11 = composite.select(bands_map['swir1']).multiply(10000).rename('B11')
    B12 = composite.select(bands_map['swir2']).multiply(10000).rename('B12')

    # Calculate indices (same as Sentinel-2)
    nir = composite.select(bands_map['nir'])
    red = composite.select(bands_map['red'])
    green = composite.select(bands_map['green'])
    swir1 = composite.select(bands_map['swir1'])

    NDVI = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    NDWI = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
    NDBI = swir1.subtract(nir).divide(swir1.add(nir)).rename('NDBI')

    # Combine all
    full_image = ee.Image.cat([B2, B3, B4, B8, B11, B12, NDVI, NDWI, NDBI])

    try:
        # Download as GeoTIFF (Landsat is 30m, so we'll get lower resolution)
        url = full_image.getDownloadURL({
            'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI'],
            'region': geometry,
            'scale': 30,  # Landsat native resolution
            'format': 'GEO_TIFF'
        })

        response = requests.get(url, timeout=300)

        if response.status_code != 200:
            print(f"  ERROR: Download failed with status {response.status_code}")
            return None

        with MemoryFile(response.content) as memfile:
            with memfile.open() as src:
                data = src.read().astype(np.float32)

        print(f"  Downloaded Landsat shape: {data.shape}")
        return data

    except Exception as e:
        print(f"  ERROR downloading Landsat: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_imagery_tiled(bounds, start_date, end_date, satellite='sentinel', grid_size=4):
    """
    Download imagery in tiles to avoid GEE size limits, then stitch together.

    Args:
        bounds: Dict with min_lon, max_lon, min_lat, max_lat
        start_date, end_date: Date range
        satellite: 'sentinel' or 'landsat'
        grid_size: Number of tiles in each direction (grid_size x grid_size)

    Returns: numpy array of shape (9, H, W)
    """
    print(f"  Downloading {satellite.upper()} for {start_date} to {end_date} in {grid_size}x{grid_size} tiles...")

    # Calculate tile bounds
    lon_step = (bounds['max_lon'] - bounds['min_lon']) / grid_size
    lat_step = (bounds['max_lat'] - bounds['min_lat']) / grid_size

    all_tiles = []
    tile_shapes = []

    for row in range(grid_size):
        row_tiles = []
        for col in range(grid_size):
            tile_bounds = {
                'min_lon': bounds['min_lon'] + col * lon_step,
                'max_lon': bounds['min_lon'] + (col + 1) * lon_step,
                'min_lat': bounds['min_lat'] + row * lat_step,
                'max_lat': bounds['min_lat'] + (row + 1) * lat_step
            }

            print(f"    Tile [{row},{col}]...", end=" ", flush=True)

            if satellite == 'landsat':
                tile_data = download_landsat_tile(tile_bounds, start_date, end_date)
            else:
                tile_data = download_sentinel2_tile(tile_bounds, start_date, end_date)

            if tile_data is None:
                print("FAILED")
                return None

            print(f"OK {tile_data.shape[1:]}")
            row_tiles.append(tile_data)

        all_tiles.append(row_tiles)

    # Stitch tiles together
    print("  Stitching tiles...")
    rows_stitched = []
    for row_tiles in all_tiles:
        # Concatenate horizontally (along width axis)
        row_stitched = np.concatenate(row_tiles, axis=2)
        rows_stitched.append(row_stitched)

    # Concatenate vertically (along height axis)
    full_image = np.concatenate(rows_stitched, axis=1)
    print(f"  Final stitched shape: {full_image.shape}")

    return full_image


def download_sentinel2_tile(bounds, start_date, end_date):
    """Download a single Sentinel-2 tile"""
    geometry = ee.Geometry.Rectangle([
        bounds['min_lon'], bounds['min_lat'],
        bounds['max_lon'], bounds['max_lat']
    ])

    # Try S2_SR_HARMONIZED first, fallback to S2_SR or S2
    collections = [
        'COPERNICUS/S2_SR_HARMONIZED',
        'COPERNICUS/S2_SR',
        'COPERNICUS/S2'
    ]

    s2 = None
    for coll in collections:
        s2 = (ee.ImageCollection(coll)
              .filterBounds(geometry)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))

        count = s2.size().getInfo()
        if count > 0:
            print(f"[{coll}: {count}]", end=" ", flush=True)
            break

    if s2 is None or s2.size().getInfo() == 0:
        # Last resort: try with very high cloud tolerance
        for coll in collections:
            s2 = (ee.ImageCollection(coll)
                  .filterBounds(geometry)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80)))
            if s2.size().getInfo() > 0:
                break

    if s2.size().getInfo() == 0:
        return None

    composite = s2.median().clip(geometry)

    B2 = composite.select('B2')
    B3 = composite.select('B3')
    B4 = composite.select('B4')
    B8 = composite.select('B8')
    B11 = composite.select('B11')
    B12 = composite.select('B12')

    NDVI = B8.subtract(B4).divide(B8.add(B4)).rename('NDVI')
    NDWI = B3.subtract(B8).divide(B3.add(B8)).rename('NDWI')
    NDBI = B11.subtract(B8).divide(B11.add(B8)).rename('NDBI')

    full_image = ee.Image.cat([B2, B3, B4, B8, B11, B12, NDVI, NDWI, NDBI])

    try:
        url = full_image.getDownloadURL({
            'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI'],
            'region': geometry,
            'scale': 10,
            'format': 'GEO_TIFF'
        })

        response = requests.get(url, timeout=180)
        if response.status_code != 200:
            print(f"[HTTP {response.status_code}]", end=" ")
            return None

        with MemoryFile(response.content) as memfile:
            with memfile.open() as src:
                return src.read().astype(np.float32)

    except requests.exceptions.Timeout:
        print("[TIMEOUT]", end=" ")
        return None
    except Exception as e:
        print(f"[ERR: {str(e)[:50]}]", end=" ")
        return None


def download_landsat_tile(bounds, start_date, end_date):
    """Download a single Landsat tile"""
    geometry = ee.Geometry.Rectangle([
        bounds['min_lon'], bounds['min_lat'],
        bounds['max_lon'], bounds['max_lat']
    ])

    year = int(start_date[:4])

    if year >= 2013:
        collection = 'LANDSAT/LC08/C02/T1_L2'
        bands_map = {'blue': 'SR_B2', 'green': 'SR_B3', 'red': 'SR_B4',
                     'nir': 'SR_B5', 'swir1': 'SR_B6', 'swir2': 'SR_B7'}
    else:
        collection = 'LANDSAT/LE07/C02/T1_L2'
        bands_map = {'blue': 'SR_B1', 'green': 'SR_B2', 'red': 'SR_B3',
                     'nir': 'SR_B4', 'swir1': 'SR_B5', 'swir2': 'SR_B7'}

    scale_factor = 0.0000275
    offset = -0.2

    landsat = (ee.ImageCollection(collection)
               .filterBounds(geometry)
               .filterDate(start_date, end_date)
               .filter(ee.Filter.lt('CLOUD_COVER', 40)))

    if landsat.size().getInfo() == 0:
        landsat = (ee.ImageCollection(collection)
                   .filterBounds(geometry)
                   .filterDate(start_date, end_date)
                   .filter(ee.Filter.lt('CLOUD_COVER', 60)))

    if landsat.size().getInfo() == 0:
        return None

    def apply_scale(image):
        optical = image.select(['SR_B.*']).multiply(scale_factor).add(offset)
        return optical

    composite = landsat.map(apply_scale).median().clip(geometry)

    B2 = composite.select(bands_map['blue']).multiply(10000).rename('B2')
    B3 = composite.select(bands_map['green']).multiply(10000).rename('B3')
    B4 = composite.select(bands_map['red']).multiply(10000).rename('B4')
    B8 = composite.select(bands_map['nir']).multiply(10000).rename('B8')
    B11 = composite.select(bands_map['swir1']).multiply(10000).rename('B11')
    B12 = composite.select(bands_map['swir2']).multiply(10000).rename('B12')

    nir = composite.select(bands_map['nir'])
    red = composite.select(bands_map['red'])
    green = composite.select(bands_map['green'])
    swir1 = composite.select(bands_map['swir1'])

    NDVI = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    NDWI = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
    NDBI = swir1.subtract(nir).divide(swir1.add(nir)).rename('NDBI')

    full_image = ee.Image.cat([B2, B3, B4, B8, B11, B12, NDVI, NDWI, NDBI])

    try:
        url = full_image.getDownloadURL({
            'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI'],
            'region': geometry,
            'scale': 30,
            'format': 'GEO_TIFF'
        })

        response = requests.get(url, timeout=120)
        if response.status_code != 200:
            return None

        with MemoryFile(response.content) as memfile:
            with memfile.open() as src:
                return src.read().astype(np.float32)

    except Exception as e:
        return None


def download_sentinel2_composite(bounds, start_date, end_date, tile_size=512):
    """Wrapper that uses tiled download"""
    return download_imagery_tiled(bounds, start_date, end_date, satellite='sentinel', grid_size=4)


def classify_image(model, bands_data, tile_size=256):
    """
    Run classification on large image by tiling

    Args:
        model: Trained model
        bands_data: (9, H, W) numpy array
        tile_size: Size of tiles for inference

    Returns:
        Classification map of shape (H, W)
    """
    _, h, w = bands_data.shape
    prediction = np.zeros((h, w), dtype=np.uint8)

    # Process in tiles with overlap
    stride = tile_size // 2  # 50% overlap

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = bands_data[:, y_start:y_end, x_start:x_end]

            # Pad if needed
            if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                padded = np.zeros((9, tile_size, tile_size), dtype=np.float32)
                padded[:, :tile.shape[1], :tile.shape[2]] = tile
                tile = padded

            # Run inference
            input_tensor = torch.from_numpy(tile).float().unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)
                pred_tile = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # Place in output (only the valid center region for overlap)
            if stride < tile_size:
                # Use center region to avoid edge artifacts
                margin = stride // 2
                pred_region = pred_tile[margin:-margin, margin:-margin] if margin > 0 else pred_tile
                y_dst = y_start + margin
                x_dst = x_start + margin
                y_dst_end = min(y_dst + pred_region.shape[0], h)
                x_dst_end = min(x_dst + pred_region.shape[1], w)
                prediction[y_dst:y_dst_end, x_dst:x_dst_end] = pred_region[:y_dst_end-y_dst, :x_dst_end-x_dst]
            else:
                prediction[y_start:y_end, x_start:x_end] = pred_tile[:y_end-y_start, :x_end-x_start]

    return prediction


def calculate_statistics(classification):
    """Calculate area statistics for each class"""
    unique, counts = np.unique(classification, return_counts=True)
    total_pixels = classification.size

    # Pixel size at 10m resolution
    pixel_area_km2 = (10 * 10) / 1e6  # 0.0001 km²

    stats = {}
    for class_id, count in zip(unique, counts):
        if class_id in CLASSES:
            stats[CLASSES[class_id]['name']] = {
                'pixels': int(count),
                'percentage': round((count / total_pixels) * 100, 2),
                'area_km2': round(count * pixel_area_km2, 2)
            }

    return stats


def detect_changes(classification_t1, classification_t2):
    """
    Detect changes between two classification maps

    Returns:
        change_map: Map showing change types
        change_matrix: Transition matrix
        change_stats: Statistics about changes
    """
    h, w = classification_t1.shape
    change_map = np.zeros((h, w, 3), dtype=np.uint8)
    change_map[:] = [200, 200, 200]  # Gray for no change

    # Create transition matrix
    num_classes = 7
    transition_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(num_classes):
        for j in range(num_classes):
            mask = (classification_t1 == i) & (classification_t2 == j)
            transition_matrix[i, j] = mask.sum()

    # Highlight significant changes
    changes_detected = {}

    # Urbanization (Forest/Plantation/Bare → Urban)
    urban_mask = ((classification_t1 == 2) | (classification_t1 == 3) | (classification_t1 == 6)) & (classification_t2 == 4)
    change_map[urban_mask] = [255, 0, 0]  # Red
    changes_detected['urbanization'] = int(urban_mask.sum())

    # Deforestation (Forest → anything else)
    deforest_mask = (classification_t1 == 2) & (classification_t2 != 2) & (classification_t2 != 1)
    change_map[deforest_mask] = [255, 165, 0]  # Orange
    changes_detected['deforestation'] = int(deforest_mask.sum())

    # Reforestation (Non-forest → Forest)
    reforest_mask = (classification_t1 != 2) & (classification_t1 != 1) & (classification_t2 == 2)
    change_map[reforest_mask] = [0, 255, 0]  # Green
    changes_detected['reforestation'] = int(reforest_mask.sum())

    # New development (Bare land → Urban/Roads)
    develop_mask = (classification_t1 == 6) & ((classification_t2 == 4) | (classification_t2 == 5))
    change_map[develop_mask] = [255, 255, 0]  # Yellow
    changes_detected['new_development'] = int(develop_mask.sum())

    # Water changes
    water_loss = (classification_t1 == 1) & (classification_t2 != 1)
    water_gain = (classification_t1 != 1) & (classification_t2 == 1)
    change_map[water_loss] = [139, 69, 19]  # Brown
    change_map[water_gain] = [0, 191, 255]  # Deep sky blue
    changes_detected['water_loss'] = int(water_loss.sum())
    changes_detected['water_gain'] = int(water_gain.sum())

    # Calculate pixel area
    pixel_area_km2 = (10 * 10) / 1e6

    change_stats = {
        'transition_matrix': transition_matrix.tolist(),
        'changes': {k: {'pixels': v, 'area_km2': round(v * pixel_area_km2, 4)}
                   for k, v in changes_detected.items()}
    }

    return change_map, transition_matrix, change_stats


def create_colored_map(classification):
    """Convert classification to RGB image"""
    h, w = classification.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, info in CLASSES.items():
        mask = classification == class_id
        colored[mask] = info['color']

    return colored


def save_results(output_dir, year, classification, colored_map, stats):
    """Save classification results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save classification as numpy
    np.save(output_dir / f'classification_{year}.npy', classification)

    # Save colored map as PNG
    Image.fromarray(colored_map).save(output_dir / f'landcover_{year}.png')

    # Save statistics as JSON
    with open(output_dir / f'stats_{year}.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved results for {year}")


def main():
    parser = argparse.ArgumentParser(description='Historical Land Cover Change Detection')
    parser.add_argument('--output-dir', type=str, default='data/historical_analysis',
                       help='Output directory')
    parser.add_argument('--years', type=str, default='2016,2019,2022,2025',
                       help='Years to analyze (comma-separated)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download, use existing data')

    args = parser.parse_args()

    # Initialize GEE
    try:
        ee.Initialize(project='ee-lloydflorens12111997')
        print("Google Earth Engine initialized")
    except Exception as e:
        print(f"ERROR: GEE initialization failed: {e}")
        return

    # Parse years
    years_to_analyze = [int(y.strip()) for y in args.years.split(',')]
    periods = [p for p in TIME_PERIODS if p['year'] in years_to_analyze]

    print(f"\n{'='*60}")
    print("HISTORICAL LAND COVER CHANGE DETECTION - MAURITIUS")
    print(f"{'='*60}")
    print(f"Time periods: {[p['name'] for p in periods]}")
    print(f"Output directory: {args.output_dir}")

    # Load model
    model = load_model()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each time period
    classifications = {}
    all_stats = {}

    for period in periods:
        print(f"\n{'='*60}")
        print(f"Processing: {period['name']}")
        print(f"{'='*60}")

        # Check for existing data
        existing_file = output_dir / f"classification_{period['name']}.npy"

        if args.skip_download and existing_file.exists():
            print(f"  Loading existing classification...")
            classification = np.load(existing_file)
        else:
            # Download imagery based on satellite type using tiled approach
            satellite = period.get('satellite', 'sentinel')

            # Use more tiles for Sentinel-2 (10m) vs Landsat (30m) due to GEE size limits
            # Sentinel at 10m is 3x higher res than Landsat, need ~9x more tiles
            grid_size = 8 if satellite == 'sentinel' else 4  # 8x8=64 tiles for Sentinel, 4x4=16 for Landsat

            bands_data = download_imagery_tiled(
                MAURITIUS_BOUNDS,
                period['start'],
                period['end'],
                satellite=satellite,
                grid_size=grid_size
            )

            if bands_data is None:
                print(f"  ERROR: Could not download data for {period['name']}")
                continue

            # Classify
            print(f"  Running classification...")
            classification = classify_image(model, bands_data)

        # Calculate statistics
        stats = calculate_statistics(classification)
        print(f"\n  Land Cover Statistics for {period['name']}:")
        for class_name, class_stats in stats.items():
            print(f"    {class_name}: {class_stats['percentage']}% ({class_stats['area_km2']} km²)")

        # Create colored map
        colored_map = create_colored_map(classification)

        # Save results
        save_results(output_dir, period['name'], classification, colored_map, stats)

        classifications[period['name']] = classification
        all_stats[period['name']] = stats

    # Perform change detection between consecutive periods
    if len(classifications) >= 2:
        print(f"\n{'='*60}")
        print("CHANGE DETECTION ANALYSIS")
        print(f"{'='*60}")

        years = sorted(classifications.keys())
        all_changes = {}

        for i in range(len(years) - 1):
            year1 = years[i]
            year2 = years[i + 1]

            print(f"\n  Comparing {year1} → {year2}...")

            change_map, transition_matrix, change_stats = detect_changes(
                classifications[year1],
                classifications[year2]
            )

            # Save change map
            Image.fromarray(change_map).save(output_dir / f'changes_{year1}_to_{year2}.png')

            # Save change statistics
            with open(output_dir / f'change_stats_{year1}_to_{year2}.json', 'w') as f:
                json.dump(change_stats, f, indent=2)

            all_changes[f"{year1}_to_{year2}"] = change_stats

            # Print summary
            print(f"\n  Changes {year1} → {year2}:")
            for change_type, data in change_stats['changes'].items():
                if data['pixels'] > 0:
                    print(f"    {change_type}: {data['area_km2']} km²")

        # Overall change (first to last)
        if len(years) > 2:
            year1 = years[0]
            year2 = years[-1]
            print(f"\n  Overall comparison {year1} → {year2}...")

            change_map, _, change_stats = detect_changes(
                classifications[year1],
                classifications[year2]
            )

            Image.fromarray(change_map).save(output_dir / f'changes_{year1}_to_{year2}_overall.png')

            with open(output_dir / f'change_stats_{year1}_to_{year2}_overall.json', 'w') as f:
                json.dump(change_stats, f, indent=2)

            print(f"\n  Total Changes {year1} → {year2}:")
            for change_type, data in change_stats['changes'].items():
                if data['pixels'] > 0:
                    print(f"    {change_type}: {data['area_km2']} km²")

    # Save overall summary
    summary = {
        'time_periods': [p['name'] for p in periods],
        'statistics_by_year': all_stats,
        'bounds': MAURITIUS_BOUNDS
    }

    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
