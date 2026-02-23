"""
Gather Targeted Training Samples for Plantation (Sugarcane) and Ocean Classes
==============================================================================

This script specifically targets:
1. Sugarcane plantation areas across Mauritius
2. Ocean/lagoon/coastal areas for water class balance

Generates training tiles to improve accuracy for these underrepresented classes.
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

# TARGETED LOCATIONS - Sugarcane Plantations
# Mauritius has ~54,000 hectares of sugarcane covering ~80% of agricultural land
PLANTATION_LOCATIONS = [
    # Northern plains - major sugarcane belt
    {"name": "Plantation_Beau_Plan", "lat": -20.0900, "lon": 57.5800},
    {"name": "Plantation_Mon_Loisir", "lat": -20.0700, "lon": 57.6100},
    {"name": "Plantation_Labourdonnais", "lat": -20.0600, "lon": 57.5500},
    {"name": "Plantation_Belle_Vue", "lat": -20.1100, "lon": 57.6200},

    # Eastern sugarcane areas
    {"name": "Plantation_FUEL_Flacq", "lat": -20.1900, "lon": 57.7200},
    {"name": "Plantation_Constance", "lat": -20.2200, "lon": 57.7400},
    {"name": "Plantation_Deep_River", "lat": -20.2500, "lon": 57.7000},
    {"name": "Plantation_Queen_Victoria", "lat": -20.2700, "lon": 57.7300},

    # Southern plantations
    {"name": "Plantation_Bel_Ombre", "lat": -20.4700, "lon": 57.4000},
    {"name": "Plantation_Riche_en_Eau", "lat": -20.4400, "lon": 57.5900},
    {"name": "Plantation_Rose_Belle", "lat": -20.3800, "lon": 57.5800},
    {"name": "Plantation_Mon_Desert", "lat": -20.3600, "lon": 57.5400},

    # Western plantations
    {"name": "Plantation_Medine", "lat": -20.2900, "lon": 57.3800},
    {"name": "Plantation_Yemen", "lat": -20.3200, "lon": 57.4100},
    {"name": "Plantation_Casela", "lat": -20.2800, "lon": 57.4200},
]

# Ocean/Lagoon/Reef areas - to give model clear ocean training data
OCEAN_LOCATIONS = [
    # Lagoon areas with clear water
    {"name": "Ocean_Blue_Bay_Lagoon", "lat": -20.4450, "lon": 57.7100},
    {"name": "Ocean_Grand_Baie_Lagoon", "lat": -19.9900, "lon": 57.5800},
    {"name": "Ocean_Trou_aux_Biches_Lagoon", "lat": -20.0300, "lon": 57.5400},
    {"name": "Ocean_Flic_en_Flac_Lagoon", "lat": -20.2950, "lon": 57.3550},
    {"name": "Ocean_Belle_Mare_Lagoon", "lat": -20.1850, "lon": 57.7850},

    # Deeper ocean areas
    {"name": "Ocean_North_Reef", "lat": -19.9600, "lon": 57.6000},
    {"name": "Ocean_East_Reef", "lat": -20.2000, "lon": 57.8200},
    {"name": "Ocean_Southwest_Open", "lat": -20.5000, "lon": 57.3500},
]


def download_sentinel2_tile(lat, lon, name, output_dir, tile_size=256):
    """Download Sentinel-2 tile with 9 bands + indices"""

    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"Location: ({lat:.4f}, {lon:.4f})")
    print(f"{'='*60}")

    # Create bounds
    # For 256x256 at 10m resolution = 2.56km x 2.56km
    km_per_degree = 111.0
    km_size = (tile_size * 10) / 1000
    buffer_deg = (km_size / 2) / (km_per_degree * np.cos(np.radians(lat)))

    bounds = ee.Geometry.Rectangle([
        lon - buffer_deg, lat - buffer_deg,
        lon + buffer_deg, lat + buffer_deg
    ])

    # Get recent cloud-free Sentinel-2
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(bounds)
          .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15)))

    if s2.size().getInfo() == 0:
        print(f"  WARNING: No cloud-free imagery found, trying higher cloud tolerance...")
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(bounds)
              .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))

    if s2.size().getInfo() == 0:
        print(f"  ERROR: No imagery found for {name}")
        return None

    composite = s2.median().clip(bounds)

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

        output_path = output_dir / f"{name}_tile_001.npy"
        np.save(output_path, bands_data)
        print(f"  Saved to: {output_path}")

        return bands_data

    except Exception as e:
        print(f"  ERROR downloading {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Gather targeted samples for plantation and ocean')
    parser.add_argument('--output-dir', type=str, default='data/training/tiles',
                       help='Output directory for tiles')
    parser.add_argument('--plantation-only', action='store_true',
                       help='Only download plantation samples')
    parser.add_argument('--ocean-only', action='store_true',
                       help='Only download ocean samples')
    parser.add_argument('--tile-size', type=int, default=256,
                       help='Tile size in pixels')

    args = parser.parse_args()

    try:
        ee.Initialize(project='ee-lloydflorens12111997')
        print("SUCCESS: Google Earth Engine initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize GEE: {e}")
        print("Run: earthengine authenticate")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print("# TARGETED SAMPLE COLLECTION - Plantation & Ocean Focus")
    print(f"{'#'*60}")
    print(f"Output directory: {output_dir}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")

    locations_to_download = []

    if args.ocean_only:
        locations_to_download = OCEAN_LOCATIONS
        print(f"\nMode: Ocean samples only ({len(OCEAN_LOCATIONS)} locations)")
    elif args.plantation_only:
        locations_to_download = PLANTATION_LOCATIONS
        print(f"\nMode: Plantation samples only ({len(PLANTATION_LOCATIONS)} locations)")
    else:
        locations_to_download = PLANTATION_LOCATIONS + OCEAN_LOCATIONS
        print(f"\nMode: Both plantation and ocean")
        print(f"  Plantation locations: {len(PLANTATION_LOCATIONS)}")
        print(f"  Ocean locations: {len(OCEAN_LOCATIONS)}")
        print(f"  Total: {len(locations_to_download)}")

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

    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total tiles in {output_dir}: {len(list(output_dir.glob('*_tile_*.npy')))}")
    print(f"\nNext steps:")
    print(f"1. Auto-label tiles: py scripts/auto_label_tiles_fixed.py --tiles-dir {output_dir}")
    print(f"2. Retrain model: py scripts/retrain_9class_v2.py")


if __name__ == '__main__':
    main()
