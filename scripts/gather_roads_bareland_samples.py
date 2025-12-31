"""
Gather Targeted Training Samples for Roads and Bare Land Classes
=================================================================

This script specifically targets:
1. Major roads and highways (M1, M2, A1-A10)
2. Bare land areas (quarries, construction sites, cleared land)

Generates more training tiles to improve accuracy for these classes.
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

# TARGETED LOCATIONS - Heavy focus on Roads and Bare Land

# Major road locations - along highways and main roads
ROAD_LOCATIONS = [
    # M1 Motorway (Port Louis to Airport)
    {"name": "M1_PortLouis_Exit", "lat": -20.1750, "lon": 57.4950},
    {"name": "M1_Ebene_Junction", "lat": -20.2450, "lon": 57.4850},
    {"name": "M1_Phoenix_Section", "lat": -20.2750, "lon": 57.4950},
    {"name": "M1_Highlands_Section", "lat": -20.3100, "lon": 57.5050},
    {"name": "M1_Wooton_Section", "lat": -20.3400, "lon": 57.5200},
    {"name": "M1_Airport_Approach", "lat": -20.4100, "lon": 57.6700},

    # M2 Motorway (Terre Rouge - Verdun)
    {"name": "M2_Terre_Rouge", "lat": -20.1350, "lon": 57.5200},
    {"name": "M2_Pamplemousses", "lat": -20.1000, "lon": 57.5700},
    {"name": "M2_Forbach", "lat": -20.0800, "lon": 57.5900},

    # A1 Road (Port Louis - Grand Baie)
    {"name": "A1_Baie_du_Tombeau", "lat": -20.1200, "lon": 57.5100},
    {"name": "A1_Triolet", "lat": -20.0550, "lon": 57.5450},
    {"name": "A1_Grand_Baie_Road", "lat": -20.0200, "lon": 57.5650},

    # B Road intersections
    {"name": "B2_Moka_Junction", "lat": -20.2200, "lon": 57.4950},
    {"name": "B3_Curepipe_Road", "lat": -20.3000, "lon": 57.5100},
    {"name": "B7_Flacq_Road", "lat": -20.2300, "lon": 57.7100},

    # Coastal roads
    {"name": "Coastal_Road_West", "lat": -20.3500, "lon": 57.3700},
    {"name": "Coastal_Road_South", "lat": -20.4800, "lon": 57.5500},
    {"name": "Coastal_Road_East", "lat": -20.3500, "lon": 57.7500},
]

# Bare land / Quarry / Construction locations
BARE_LAND_LOCATIONS = [
    # Quarries and mining areas
    {"name": "Basalt_Quarry_Midlands", "lat": -20.2800, "lon": 57.5400},
    {"name": "Quarry_Mare_Chicose", "lat": -20.3600, "lon": 57.5800},
    {"name": "Quarry_Nouvelle_France", "lat": -20.4200, "lon": 57.5600},

    # Industrial/cleared areas
    {"name": "Industrial_Zone_Phoenix", "lat": -20.2600, "lon": 57.4900},
    {"name": "Industrial_Coromandel", "lat": -20.2100, "lon": 57.5250},
    {"name": "Jin_Fei_Zone", "lat": -20.0600, "lon": 57.5300},

    # Airport area (large bare/paved areas)
    {"name": "SSR_Airport_Runway", "lat": -20.4300, "lon": 57.6800},
    {"name": "Airport_Cargo_Area", "lat": -20.4250, "lon": 57.6650},

    # Construction/development areas
    {"name": "Smart_City_Moka", "lat": -20.2150, "lon": 57.5000},
    {"name": "Cote_dOr_Development", "lat": -20.2750, "lon": 57.5650},
    {"name": "Port_Extension", "lat": -20.1550, "lon": 57.5050},

    # Beach/sandy areas (similar spectral signature)
    {"name": "Mont_Choisy_Beach", "lat": -20.0350, "lon": 57.5550},
    {"name": "Belle_Mare_Beach", "lat": -20.1950, "lon": 57.7800},
    {"name": "Le_Morne_Beach", "lat": -20.4550, "lon": 57.3200},

    # Cleared agricultural land
    {"name": "Cleared_Land_Flacq", "lat": -20.1800, "lon": 57.7000},
    {"name": "Cleared_Land_Pamplemousses", "lat": -20.0850, "lon": 57.5450},
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
    start_date = end_date - timedelta(days=90)  # Extended to 90 days for more options

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(bounds)
          .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15)))

    if s2.size().getInfo() == 0:
        print(f"  WARNING: No cloud-free imagery found for {name}, trying with higher cloud tolerance...")
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(bounds)
              .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))

    if s2.size().getInfo() == 0:
        print(f"  ERROR: No imagery found for {name}")
        return None

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
        print(f"  Saved to: {output_path}")

        return bands_data

    except Exception as e:
        print(f"  ERROR downloading {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Gather targeted samples for roads and bare land')
    parser.add_argument('--output-dir', type=str, default='data/training/tiles',
                       help='Output directory for tiles')
    parser.add_argument('--roads-only', action='store_true',
                       help='Only download road samples')
    parser.add_argument('--bareland-only', action='store_true',
                       help='Only download bare land samples')
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
    print("# TARGETED SAMPLE COLLECTION - Roads & Bare Land Focus")
    print(f"{'#'*60}")
    print(f"Output directory: {output_dir}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")

    # Determine which samples to download
    locations_to_download = []

    if args.bareland_only:
        locations_to_download = BARE_LAND_LOCATIONS
        print(f"\nMode: Bare land samples only ({len(BARE_LAND_LOCATIONS)} locations)")
    elif args.roads_only:
        locations_to_download = ROAD_LOCATIONS
        print(f"\nMode: Road samples only ({len(ROAD_LOCATIONS)} locations)")
    else:
        locations_to_download = ROAD_LOCATIONS + BARE_LAND_LOCATIONS
        print(f"\nMode: Both roads and bare land")
        print(f"  Road locations: {len(ROAD_LOCATIONS)}")
        print(f"  Bare land locations: {len(BARE_LAND_LOCATIONS)}")
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
    print(f"1. Auto-label tiles: python scripts/auto_label_tiles_fixed.py --tiles-dir {output_dir}")
    print(f"2. Retrain model: python scripts/train_enhanced.py --epochs 50 --use-class-weights")


if __name__ == '__main__':
    main()
