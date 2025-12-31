"""
Gather Targeted Training Samples for Urban and Water Classes
=============================================================

This script specifically targets:
1. Urban areas (Port Louis, Curepipe, Quatre Bornes, etc.)
2. Water bodies (coast, rivers, lakes, lagoons)

Generates more training tiles to balance the dataset.
"""

import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import argparse

sys.path.append(str(Path(__file__).parent.parent))

try:
    import ee
    import requests
    from rasterio.io import MemoryFile
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    print("ERROR: Missing dependencies. Install with: pip install earthengine-api rasterio requests")
    sys.exit(1)

# TARGETED LOCATIONS - Heavy focus on Urban and Water
URBAN_LOCATIONS = [
    # Port Louis - Main city
    {"name": "Port_Louis_Central", "lat": -20.1609, "lon": 57.5012},
    {"name": "Port_Louis_North", "lat": -20.1500, "lon": 57.5000},
    {"name": "Port_Louis_Harbor", "lat": -20.1650, "lon": 57.5100},

    # Curepipe - Urban center
    {"name": "Curepipe_Center", "lat": -20.3167, "lon": 57.5167},
    {"name": "Curepipe_South", "lat": -20.3250, "lon": 57.5150},

    # Quatre Bornes - Dense urban
    {"name": "Quatre_Bornes", "lat": -20.2650, "lon": 57.4800},

    # Vacoas - Urban
    {"name": "Vacoas", "lat": -20.2983, "lon": 57.4783},

    # Rose Hill - Urban
    {"name": "Rose_Hill", "lat": -20.2167, "lon": 57.4833},

    # Beau Bassin - Urban
    {"name": "Beau_Bassin", "lat": -20.2333, "lon": 57.4667},

    # Phoenix - Industrial/Urban
    {"name": "Phoenix", "lat": -20.2500, "lon": 57.5000},
]

WATER_LOCATIONS = [
    # Coastal waters - North
    {"name": "Grand_Baie_Coast", "lat": -20.0100, "lon": 57.5800},
    {"name": "Cap_Malheureux", "lat": -19.9833, "lon": 57.6167},
    {"name": "Trou_aux_Biches", "lat": -20.0400, "lon": 57.5500},

    # Coastal waters - East
    {"name": "Belle_Mare_Lagoon", "lat": -20.2000, "lon": 57.7700},
    {"name": "Trou_d_Eau_Douce", "lat": -20.2400, "lon": 57.7867},
    {"name": "Blue_Bay", "lat": -20.4433, "lon": 57.7033},

    # Coastal waters - West
    {"name": "Flic_en_Flac_Coast", "lat": -20.2833, "lon": 57.3667},
    {"name": "Tamarin_Bay", "lat": -20.3217, "lon": 57.3700},
    {"name": "Le_Morne_Lagoon", "lat": -20.4667, "lon": 57.3167},

    # Coastal waters - South
    {"name": "Mahebourg_Bay", "lat": -20.4100, "lon": 57.7050},
    {"name": "Souillac_Coast", "lat": -20.5167, "lon": 57.5167},

    # Inland water
    {"name": "Mare_aux_Vacoas", "lat": -20.2917, "lon": 57.4600},
    {"name": "La_Nicoliere_Reservoir", "lat": -20.2167, "lon": 57.6000},
]

def download_sentinel2_tile(lat, lon, name, output_dir, tile_size=256):
    """Download Sentinel-2 tile with 9 bands + indices"""

    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"Location: ({lat:.4f}, {lon:.4f})")
    print(f"{'='*60}")

    # Create bounds
    # For 256x256 at 10m resolution = 2.56km x 2.56km
    meters_per_degree = 111000  # approximate
    km_size = (tile_size * 10) / 1000  # tile_size pixels * 10m resolution
    buffer_deg = (km_size / 2) / (meters_per_degree * np.cos(np.radians(lat)))

    bounds = ee.Geometry.Rectangle([
        lon - buffer_deg, lat - buffer_deg,
        lon + buffer_deg, lat + buffer_deg
    ])

    # Get recent cloud-free Sentinel-2
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(bounds)
          .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)))

    if s2.size().getInfo() == 0:
        print(f"  WARNING: No cloud-free imagery found for {name}, trying with higher cloud tolerance...")
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(bounds)
              .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))

    composite = s2.median().clip(bounds)

    # Calculate spectral indices
    B2 = composite.select('B2')  # Blue
    B3 = composite.select('B3')  # Green
    B4 = composite.select('B4')  # Red
    B8 = composite.select('B8')  # NIR
    B11 = composite.select('B11')  # SWIR1
    B12 = composite.select('B12')  # SWIR2

    # NDVI = (NIR - Red) / (NIR + Red)
    NDVI = B8.subtract(B4).divide(B8.add(B4)).rename('NDVI')

    # NDWI = (Green - NIR) / (Green + NIR)
    NDWI = B3.subtract(B8).divide(B3.add(B8)).rename('NDWI')

    # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    NDBI = B11.subtract(B8).divide(B11.add(B8)).rename('NDBI')

    # Combine all bands
    full_image = ee.Image.cat([B2, B3, B4, B8, B11, B12, NDVI, NDWI, NDBI])

    try:
        # Download as GeoTIFF
        bands_url = full_image.getDownloadURL({
            'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI'],
            'region': bounds,
            'scale': 10,
            'format': 'GEO_TIFF'
        })

        print(f"  Downloading from GEE...")
        response = requests.get(bands_url, timeout=60)

        if response.status_code != 200:
            print(f"  ERROR: Download failed with status {response.status_code}")
            return None

        # Read with rasterio
        with MemoryFile(response.content) as memfile:
            with memfile.open() as src:
                bands_data = src.read().astype(np.float32)

        print(f"  Downloaded shape: {bands_data.shape}")
        print(f"  Data range: min={bands_data.min():.2f}, max={bands_data.max():.2f}")

        # Resize to exact tile_size if needed
        if bands_data.shape[1] != tile_size or bands_data.shape[2] != tile_size:
            print(f"  Resizing from {bands_data.shape[1:]} to ({tile_size}, {tile_size})...")
            from PIL import Image
            resized = np.zeros((9, tile_size, tile_size), dtype=np.float32)
            for i in range(9):
                band_img = Image.fromarray(bands_data[i])
                resized[i] = np.array(band_img.resize((tile_size, tile_size)))
            bands_data = resized

        # Save tile
        output_path = output_dir / f"{name}_tile_001.npy"
        np.save(output_path, bands_data)
        print(f"  ✓ Saved to: {output_path}")

        return bands_data

    except Exception as e:
        print(f"  ERROR downloading {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Gather targeted samples for urban and water')
    parser.add_argument('--output-dir', type=str, default='data/training/tiles',
                       help='Output directory for tiles')
    parser.add_argument('--urban-only', action='store_true',
                       help='Only download urban samples')
    parser.add_argument('--water-only', action='store_true',
                       help='Only download water samples')
    parser.add_argument('--tile-size', type=int, default=256,
                       help='Tile size in pixels')

    args = parser.parse_args()

    # Initialize GEE
    try:
        ee.Initialize(project='ee-lloydflorens12111997')
        print("SUCCESS: Google Earth Engine initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize GEE: {e}")
        print("Run: earthengine authenticate")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print("# TARGETED SAMPLE COLLECTION - Urban & Water Focus")
    print(f"{'#'*60}")
    print(f"Output directory: {output_dir}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")

    # Determine which samples to download
    locations_to_download = []

    if args.water_only:
        locations_to_download = WATER_LOCATIONS
        print(f"\nMode: Water samples only ({len(WATER_LOCATIONS)} locations)")
    elif args.urban_only:
        locations_to_download = URBAN_LOCATIONS
        print(f"\nMode: Urban samples only ({len(URBAN_LOCATIONS)} locations)")
    else:
        locations_to_download = URBAN_LOCATIONS + WATER_LOCATIONS
        print(f"\nMode: Both urban and water")
        print(f"  Urban locations: {len(URBAN_LOCATIONS)}")
        print(f"  Water locations: {len(WATER_LOCATIONS)}")
        print(f"  Total: {len(locations_to_download)}")

    # Download all samples
    success_count = 0
    fail_count = 0

    for i, loc in enumerate(locations_to_download, 1):
        print(f"\n[{i}/{len(locations_to_download)}]")
        result = download_sentinel2_tile(
            loc['lat'], loc['lon'], loc['name'],
            output_dir, args.tile_size
        )

        if result is not None:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total tiles in {output_dir}: {len(list(output_dir.glob('*_tile_*.npy')))}")
    print(f"\nNext steps:")
    print(f"1. Auto-label tiles: python scripts/auto_label_tiles.py --tiles-dir {output_dir}")
    print(f"2. Train model: python scripts/train_enhanced.py --data-dir {output_dir} --epochs 50")


if __name__ == '__main__':
    main()
