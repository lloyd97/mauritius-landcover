"""
Live Interactive Map Viewer for Mauritius Land Cover Classification
Fetches and classifies Sentinel-2/Landsat imagery in real-time as you pan the map
Supports historical imagery from 2010-present with change detection charts
"""

import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
import segmentation_models_pytorch as smp
from PIL import Image
import io
import base64
import ee
import requests
from datetime import datetime, timedelta
import rasterio
from rasterio.io import MemoryFile
import json
from scipy.ndimage import median_filter

app = Flask(__name__)

# Historical time periods configuration
TIME_PERIODS = {
    'current': {'start': None, 'end': None, 'satellite': 'sentinel', 'label': 'Current (Last 60 days)'},
    '2022': {'start': '2022-01-01', 'end': '2022-12-31', 'satellite': 'sentinel', 'label': '2022'},
    '2019': {'start': '2019-01-01', 'end': '2019-12-31', 'satellite': 'sentinel', 'label': '2019'},
    '2016': {'start': '2016-01-01', 'end': '2016-12-31', 'satellite': 'sentinel', 'label': '2016'},
    '2013': {'start': '2013-01-01', 'end': '2013-12-31', 'satellite': 'landsat', 'label': '2013'},
    '2010': {'start': '2010-01-01', 'end': '2010-12-31', 'satellite': 'landsat', 'label': '2010'},
}

# Store historical statistics for change detection charts
HISTORICAL_STATS = {}

# Cache for full island classification results
MAURITIUS_CLASSIFICATION_CACHE = {}

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None
DATA_DIR = Path('data/live_imagery')
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Mauritius bounds
MAURITIUS_BOUNDS = {
    'min_lon': 57.30,
    'max_lon': 57.82,
    'min_lat': -20.53,
    'max_lat': -19.98,
    'center_lat': -20.25,
    'center_lon': 57.55
}

# Land cover classes - Apple Maps inspired color palette
# Designed to match Apple Maps aesthetic: soft blues, natural greens, warm grays
CLASSES = {
    0: {'name': 'Clouds', 'color': [255, 255, 255]},          # White for clouds
    1: {'name': 'Water', 'color': [168, 216, 234]},           # Soft light blue - inland water (lakes, rivers)
    2: {'name': 'Forest', 'color': [34, 139, 34]},            # Forest green - dense vegetation (NDVI > 0.7)
    3: {'name': 'Plantation', 'color': [218, 165, 32]},       # Goldenrod/yellow - agricultural (sugarcane, crops)
    4: {'name': 'Urban', 'color': [215, 204, 200]},           # Warm gray/tan (Apple Maps buildings)
    5: {'name': 'Roads', 'color': [158, 158, 158]},           # Medium gray (Apple Maps roads)
    6: {'name': 'Bare Land', 'color': [239, 235, 233]},       # Sandy cream (Apple Maps bare areas)
    7: {'name': 'Ocean', 'color': [135, 190, 220]},           # Darker blue - ocean/sea water
    8: {'name': 'Wasteland', 'color': [189, 183, 107]}        # Khaki/olive - sparse vegetation, scrubland
}

# Initialize Earth Engine
GEE_AVAILABLE = False
try:
    # Check for EARTHENGINE_TOKEN environment variable (for cloud deployment)
    ee_token = os.environ.get('EARTHENGINE_TOKEN')
    if ee_token:
        credentials_dict = json.loads(ee_token)

        if 'private_key' in credentials_dict:
            # Service account credentials
            credentials = ee.ServiceAccountCredentials(None, key_data=credentials_dict)
            ee.Initialize(credentials)
        else:
            # OAuth refresh token - write to EE's expected credentials location
            ee_credentials_dir = os.path.expanduser('~/.config/earthengine')
            os.makedirs(ee_credentials_dir, exist_ok=True)
            credentials_path = os.path.join(ee_credentials_dir, 'credentials')
            with open(credentials_path, 'w') as f:
                json.dump(credentials_dict, f)
            print(f"Wrote credentials to {credentials_path}")

            # Initialize with project
            project = credentials_dict.get('project')
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
        print("SUCCESS: Google Earth Engine initialized from EARTHENGINE_TOKEN")
    else:
        ee.Initialize()
        print("SUCCESS: Google Earth Engine initialized from local credentials")
    GEE_AVAILABLE = True
except Exception as e:
    print(f"WARNING: GEE initialization failed: {e}")


# Mauritius land mask - cached at module level
MAURITIUS_LAND_MASK = None

def get_mauritius_land_mask():
    """
    Get a land mask for Mauritius using GEE's country boundaries.
    Returns an ee.Image where land=1 and ocean=0.
    """
    global MAURITIUS_LAND_MASK

    if MAURITIUS_LAND_MASK is not None:
        return MAURITIUS_LAND_MASK

    if not GEE_AVAILABLE:
        return None

    try:
        # Use LSIB (Large Scale International Boundary) dataset for country boundaries
        # Filter for Mauritius
        countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
        mauritius = countries.filter(ee.Filter.eq('country_na', 'Mauritius'))

        # Create a mask image: 1 inside Mauritius, 0 outside
        MAURITIUS_LAND_MASK = ee.Image.constant(1).clip(mauritius).unmask(0)
        print("SUCCESS: Loaded Mauritius land boundary mask")
        return MAURITIUS_LAND_MASK
    except Exception as e:
        print(f"WARNING: Could not load Mauritius land mask: {e}")
        return None


def apply_ocean_mask_composite(composite, bounds):
    """
    Apply ocean mask to the full composite classification image.
    Water pixels outside Mauritius land boundary become 'Ocean' (class 7).

    Uses chunked processing to avoid GEE's sampleRectangle pixel limit (262144).

    Args:
        composite: numpy array of class predictions (full mosaic)
        bounds: dict with north, south, east, west coordinates (full Mauritius bounds)

    Returns:
        Modified composite with ocean pixels marked as class 7
    """
    if not GEE_AVAILABLE:
        print("    GEE not available, skipping ocean mask")
        return composite

    land_mask_ee = get_mauritius_land_mask()
    if land_mask_ee is None:
        print("    Land mask not available, skipping ocean mask")
        return composite

    try:
        h, w = composite.shape
        print(f"    Composite shape: {h}x{w} = {h*w} pixels")

        # GEE limit is 262144 pixels per sampleRectangle call
        # Use 450x450 chunks (202500 pixels) to stay safely under the limit
        chunk_size = 450
        n_chunks_y = (h + chunk_size - 1) // chunk_size
        n_chunks_x = (w + chunk_size - 1) // chunk_size

        print(f"    Processing land mask in {n_chunks_y}x{n_chunks_x} = {n_chunks_y * n_chunks_x} chunks")

        lat_range = bounds['north'] - bounds['south']
        lon_range = bounds['east'] - bounds['west']

        land_mask_np = np.zeros((h, w), dtype=np.uint8)

        for cy in range(n_chunks_y):
            for cx in range(n_chunks_x):
                # Pixel coordinates for this chunk
                y_start = cy * chunk_size
                y_end = min((cy + 1) * chunk_size, h)
                x_start = cx * chunk_size
                x_end = min((cx + 1) * chunk_size, w)

                chunk_h = y_end - y_start
                chunk_w = x_end - x_start

                # Geographic bounds for this chunk
                # Note: y=0 is top (north), y=h is bottom (south)
                chunk_north = bounds['north'] - (y_start / h) * lat_range
                chunk_south = bounds['north'] - (y_end / h) * lat_range
                chunk_west = bounds['west'] + (x_start / w) * lon_range
                chunk_east = bounds['west'] + (x_end / w) * lon_range

                try:
                    region = ee.Geometry.Rectangle([chunk_west, chunk_south, chunk_east, chunk_north])

                    # Scale to match chunk resolution
                    scale = (chunk_north - chunk_south) * 111000 / chunk_h

                    chunk_mask = land_mask_ee.reproject(crs='EPSG:4326', scale=scale).sampleRectangle(
                        region=region,
                        defaultValue=0
                    ).get('constant').getInfo()

                    chunk_mask = np.array(chunk_mask)

                    # Resize if needed
                    if chunk_mask.shape != (chunk_h, chunk_w):
                        chunk_pil = Image.fromarray(chunk_mask.astype(np.uint8))
                        chunk_pil = chunk_pil.resize((chunk_w, chunk_h), Image.NEAREST)
                        chunk_mask = np.array(chunk_pil)

                    land_mask_np[y_start:y_end, x_start:x_end] = chunk_mask

                except Exception as chunk_e:
                    print(f"      Chunk [{cy},{cx}] error: {chunk_e}")
                    # Leave as 0 (ocean) if can't get mask

        # Mark ALL pixels outside land boundary as Ocean (class 7)
        # This is critical: the model classifies pure ocean tiles as Forest, Wasteland, etc.
        # We must override ALL non-land pixels, not just Water-classed ones
        outside_land = land_mask_np == 0

        ocean_count = np.sum(outside_land)
        land_count = np.sum(land_mask_np == 1)
        print(f"    Pixels outside land boundary (-> Ocean): {ocean_count}")
        print(f"    Pixels inside land boundary (kept): {land_count}")

        composite_masked = composite.copy()
        composite_masked[outside_land] = 7  # Mark ALL non-land as Ocean

        return composite_masked

    except Exception as e:
        print(f"    Warning: Could not apply ocean mask: {e}")
        import traceback
        traceback.print_exc()
        return composite


def load_model():
    """Load trained U-Net model (7-class model, remapped to 9 display classes)"""
    global MODEL

    print("Loading model...")
    MODEL = smp.Unet(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=7,
        activation=None
    )

    # Modify first conv for 9 channels
    first_conv = MODEL.encoder.conv1
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

    MODEL.encoder.conv1 = new_conv

    # Load the 7-class model that produced good results on Jan 24/25
    checkpoint_path = Path('checkpoints/enhanced_model_old7class.pth')
    if not checkpoint_path.exists():
        checkpoint_path = Path('checkpoints/enhanced_model_backup_before_wasteland.pth')
    if not checkpoint_path.exists():
        checkpoint_path = Path('checkpoints/improved_model.pth')
    if not checkpoint_path.exists():
        checkpoint_path = Path('checkpoints/best_model.pth')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        print(f"SUCCESS: Loaded model from {checkpoint_path}")
    else:
        print("WARNING: No checkpoint found")

    MODEL.to(DEVICE)
    MODEL.eval()


def download_sentinel2_for_location(lat, lon, size_km=2):
    """
    Download Sentinel-2 imagery for a specific location

    Args:
        lat: Latitude
        lon: Longitude
        size_km: Size of area to download in km (default 2km x 2km)

    Returns:
        dict with 'rgb_image' and 'bands_data'
    """
    if not GEE_AVAILABLE:
        return create_fallback_image(lat, lon)

    try:
        print(f"Downloading Sentinel-2 for: {lat:.4f}, {lon:.4f}")

        # Calculate bounds (approximately size_km x size_km)
        # 1 degree latitude ~ 111 km
        lat_offset = (size_km / 111.0) / 2
        lon_offset = (size_km / (111.0 * np.cos(np.radians(lat)))) / 2

        bounds = ee.Geometry.Rectangle([
            lon - lon_offset,
            lat - lat_offset,
            lon + lon_offset,
            lat + lat_offset
        ])

        # Get recent Sentinel-2 imagery (last 60 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)

        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(bounds) \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])

        # Get median composite
        image = s2.median().clip(bounds)

        # Calculate indices
        def calc_ndvi(img):
            return img.normalizedDifference(['B8', 'B4']).rename('NDVI')

        def calc_ndwi(img):
            return img.normalizedDifference(['B3', 'B8']).rename('NDWI')

        def calc_ndbi(img):
            return img.normalizedDifference(['B11', 'B8']).rename('NDBI')

        ndvi = calc_ndvi(image)
        ndwi = calc_ndwi(image)
        ndbi = calc_ndbi(image)

        # Combine all bands
        full_image = image.addBands([ndvi, ndwi, ndbi])

        # Get RGB thumbnail URL
        rgb_url = image.getThumbURL({
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000,
            'dimensions': 256,
            'format': 'png'
        })

        # Download RGB
        rgb_response = requests.get(rgb_url, timeout=30)
        rgb_image = np.array(Image.open(io.BytesIO(rgb_response.content)))

        # Get 9-band data as GeoTIFF using rasterio
        bands_url = full_image.getDownloadURL({
            'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI'],
            'region': bounds,
            'scale': 10,
            'format': 'GEO_TIFF'
        })

        # Download bands
        bands_response = requests.get(bands_url, timeout=30)

        # Use rasterio to read the GeoTIFF
        with MemoryFile(bands_response.content) as memfile:
            with memfile.open() as src:
                bands_data = src.read().astype(np.float32)  # Returns (bands, height, width)

        print(f"Downloaded bands data shape: {bands_data.shape}")
        print(f"GEE data range BEFORE normalization: min={bands_data.min():.4f}, max={bands_data.max():.4f}, mean={bands_data.mean():.4f}")

        # DO NOT NORMALIZE - training data is in raw Sentinel-2 values (range: -0.8 to 6592)
        # Keeping data in raw format to match training data
        # for i in range(bands_data.shape[0]):
        #     band = bands_data[i]
        #     if band.max() > 1.0:
        #         bands_data[i] = band / 10000.0  # Sentinel-2 scale factor

        # Ensure correct size
        if rgb_image.shape[:2] != (256, 256):
            rgb_image = np.array(Image.fromarray(rgb_image).resize((256, 256)))

        if bands_data.shape[1:] != (256, 256):
            resized_bands = []
            for i in range(bands_data.shape[0]):
                band_img = Image.fromarray(bands_data[i])
                band_resized = np.array(band_img.resize((256, 256), Image.BILINEAR))
                resized_bands.append(band_resized)
            bands_data = np.stack(resized_bands, axis=0)

        print(f"Successfully downloaded: RGB {rgb_image.shape}, Bands {bands_data.shape}")

        return {
            'rgb_image': rgb_image,
            'bands_data': bands_data
        }

    except Exception as e:
        print(f"Error downloading Sentinel-2: {e}")
        import traceback
        traceback.print_exc()
        return create_fallback_image(lat, lon)


def download_landsat_for_location(lat, lon, year, size_km=2):
    """
    Download Landsat imagery for a specific location and year (for 2010-2013)

    Args:
        lat: Latitude
        lon: Longitude
        year: Year string (e.g., '2010', '2013')
        size_km: Size of area in km

    Returns:
        dict with 'rgb_image' and 'bands_data'
    """
    if not GEE_AVAILABLE:
        return create_fallback_image(lat, lon)

    try:
        print(f"Downloading Landsat {year} for: {lat:.4f}, {lon:.4f}")

        period = TIME_PERIODS[year]
        start_date = period['start']
        end_date = period['end']

        # Calculate bounds
        lat_offset = (size_km / 111.0) / 2
        lon_offset = (size_km / (111.0 * np.cos(np.radians(lat)))) / 2

        bounds = ee.Geometry.Rectangle([
            lon - lon_offset,
            lat - lat_offset,
            lon + lon_offset,
            lat + lat_offset
        ])

        # Use Landsat 7 for 2010-2013 (Landsat 8 starts 2013)
        if int(year) <= 2012:
            collection = 'LANDSAT/LE07/C02/T1_L2'
            bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
            rgb_bands = ['SR_B3', 'SR_B2', 'SR_B1']
            nir_band = 'SR_B4'
            swir1_band = 'SR_B5'
            swir2_band = 'SR_B7'
            green_band = 'SR_B2'
        else:
            collection = 'LANDSAT/LC08/C02/T1_L2'
            bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
            rgb_bands = ['SR_B4', 'SR_B3', 'SR_B2']
            nir_band = 'SR_B5'
            swir1_band = 'SR_B6'
            swir2_band = 'SR_B7'
            green_band = 'SR_B3'

        landsat = ee.ImageCollection(collection) \
            .filterBounds(bounds) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', 30)) \
            .select(bands)

        if landsat.size().getInfo() == 0:
            print(f"No Landsat imagery found for {year}, trying higher cloud tolerance...")
            landsat = ee.ImageCollection(collection) \
                .filterBounds(bounds) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUD_COVER', 50)) \
                .select(bands)

        if landsat.size().getInfo() == 0:
            print(f"Still no imagery for {year}")
            return create_fallback_image(lat, lon)

        image = landsat.median().clip(bounds)

        # Calculate indices using Landsat bands
        # Rename bands to generic names for index calculation
        if int(year) <= 2012:
            image = image.rename(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
        else:
            image = image.rename(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])

        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        # NDWI = (Green - NIR) / (Green + NIR)
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')

        full_image = image.addBands([ndvi, ndwi, ndbi])

        # Get RGB thumbnail
        rgb_url = image.getThumbURL({
            'bands': ['B4', 'B3', 'B2'],
            'min': 7000,
            'max': 20000,
            'dimensions': 256,
            'format': 'png'
        })

        rgb_response = requests.get(rgb_url, timeout=30)
        rgb_image = np.array(Image.open(io.BytesIO(rgb_response.content)))

        # Get 9-band data
        bands_url = full_image.getDownloadURL({
            'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI'],
            'region': bounds,
            'scale': 30,  # Landsat resolution
            'format': 'GEO_TIFF'
        })

        bands_response = requests.get(bands_url, timeout=30)

        with MemoryFile(bands_response.content) as memfile:
            with memfile.open() as src:
                bands_data = src.read().astype(np.float32)

        # Scale Landsat values to match Sentinel-2 range for model compatibility
        # Landsat L2 surface reflectance scale factor is 0.0000275, offset -0.2
        for i in range(6):  # First 6 bands are reflectance
            bands_data[i] = bands_data[i] * 0.0000275 - 0.2
            bands_data[i] = bands_data[i] * 10000  # Scale to match Sentinel-2

        print(f"Landsat data shape: {bands_data.shape}")

        # Resize if needed
        if rgb_image.shape[:2] != (256, 256):
            rgb_image = np.array(Image.fromarray(rgb_image).resize((256, 256)))

        if bands_data.shape[1:] != (256, 256):
            resized_bands = []
            for i in range(bands_data.shape[0]):
                band_img = Image.fromarray(bands_data[i])
                band_resized = np.array(band_img.resize((256, 256), Image.BILINEAR))
                resized_bands.append(band_resized)
            bands_data = np.stack(resized_bands, axis=0)

        return {
            'rgb_image': rgb_image,
            'bands_data': bands_data
        }

    except Exception as e:
        print(f"Error downloading Landsat: {e}")
        import traceback
        traceback.print_exc()
        return create_fallback_image(lat, lon)


def download_imagery_for_year(lat, lon, year='current', size_km=2):
    """
    Download satellite imagery for a specific year
    Routes to appropriate satellite (Sentinel-2 or Landsat)
    """
    if year == 'current':
        return download_sentinel2_for_location(lat, lon, size_km)

    period = TIME_PERIODS.get(year)
    if not period:
        print(f"Unknown year: {year}, using current")
        return download_sentinel2_for_location(lat, lon, size_km)

    if period['satellite'] == 'landsat':
        return download_landsat_for_location(lat, lon, year, size_km)
    else:
        return download_sentinel2_for_year(lat, lon, year, size_km)


def download_sentinel2_for_year(lat, lon, year, size_km=2):
    """Download Sentinel-2 imagery for a specific year"""
    if not GEE_AVAILABLE:
        return create_fallback_image(lat, lon)

    try:
        print(f"Downloading Sentinel-2 {year} for: {lat:.4f}, {lon:.4f}")

        period = TIME_PERIODS[year]
        start_date = period['start']
        end_date = period['end']

        lat_offset = (size_km / 111.0) / 2
        lon_offset = (size_km / (111.0 * np.cos(np.radians(lat)))) / 2

        bounds = ee.Geometry.Rectangle([
            lon - lon_offset,
            lat - lat_offset,
            lon + lon_offset,
            lat + lat_offset
        ])

        # Try different Sentinel-2 collections based on year
        # S2_SR_HARMONIZED: 2019+
        # S2_SR: 2017+
        # S2: 2015+ (TOA, not surface reflectance)
        collections_to_try = []
        year_int = int(year)

        if year_int >= 2019:
            collections_to_try = ['COPERNICUS/S2_SR_HARMONIZED', 'COPERNICUS/S2_SR', 'COPERNICUS/S2']
        elif year_int >= 2017:
            collections_to_try = ['COPERNICUS/S2_SR', 'COPERNICUS/S2']
        else:
            collections_to_try = ['COPERNICUS/S2']  # Only TOA available for 2015-2016

        s2 = None
        used_collection = None

        for collection in collections_to_try:
            print(f"  Trying {collection}...")
            s2_test = ee.ImageCollection(collection) \
                .filterBounds(bounds) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
                .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])

            count = s2_test.size().getInfo()
            print(f"    Found {count} images")

            if count > 0:
                s2 = s2_test
                used_collection = collection
                break

        # If no imagery found with 30% cloud, try with 50%
        if s2 is None or s2.size().getInfo() == 0:
            for collection in collections_to_try:
                print(f"  Trying {collection} with relaxed cloud filter...")
                s2_test = ee.ImageCollection(collection) \
                    .filterBounds(bounds) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)) \
                    .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])

                count = s2_test.size().getInfo()
                if count > 0:
                    s2 = s2_test
                    used_collection = collection
                    print(f"    Found {count} images")
                    break

        if s2 is None or s2.size().getInfo() == 0:
            print(f"  No Sentinel-2 imagery found for {year}")
            return create_fallback_image(lat, lon)

        print(f"  Using {used_collection} with {s2.size().getInfo()} images")
        image = s2.median().clip(bounds)

        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')

        full_image = image.addBands([ndvi, ndwi, ndbi])

        rgb_url = image.getThumbURL({
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000,
            'dimensions': 256,
            'format': 'png'
        })

        rgb_response = requests.get(rgb_url, timeout=30)
        rgb_image = np.array(Image.open(io.BytesIO(rgb_response.content)))

        bands_url = full_image.getDownloadURL({
            'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI'],
            'region': bounds,
            'scale': 10,
            'format': 'GEO_TIFF'
        })

        bands_response = requests.get(bands_url, timeout=30)

        with MemoryFile(bands_response.content) as memfile:
            with memfile.open() as src:
                bands_data = src.read().astype(np.float32)

        if rgb_image.shape[:2] != (256, 256):
            rgb_image = np.array(Image.fromarray(rgb_image).resize((256, 256)))

        if bands_data.shape[1:] != (256, 256):
            resized_bands = []
            for i in range(bands_data.shape[0]):
                band_img = Image.fromarray(bands_data[i])
                band_resized = np.array(band_img.resize((256, 256), Image.BILINEAR))
                resized_bands.append(band_resized)
            bands_data = np.stack(resized_bands, axis=0)

        return {
            'rgb_image': rgb_image,
            'bands_data': bands_data
        }

    except Exception as e:
        print(f"Error downloading Sentinel-2 for {year}: {e}")
        import traceback
        traceback.print_exc()
        return create_fallback_image(lat, lon)


def create_fallback_image(lat, lon):
    """Create fallback synthetic image when GEE fails"""
    print(f"Creating fallback image for {lat:.4f}, {lon:.4f}")

    # Use training tiles as fallback
    tiles_dir = Path('data/training/tiles')
    all_npys = list(tiles_dir.glob('*.npy'))

    # Filter out mask files - only keep tile files
    all_tiles = [f for f in all_npys if '_mask' not in f.name and '_tile_' in f.name]

    if all_tiles:
        # Pick a random tile based on coordinates
        idx = int(abs(lat * 1000 + lon * 1000)) % len(all_tiles)
        tile_path = all_tiles[idx]
        tile_data = np.load(tile_path).astype(np.float32)

        print(f"Using fallback tile: {tile_path.name}, shape: {tile_data.shape}")

        # Validate shape
        if tile_data.ndim != 3 or tile_data.shape[0] != 9:
            print(f"ERROR: Invalid tile shape {tile_data.shape}, expected (9, H, W)")
            # Create synthetic data as ultimate fallback
            rgb_image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            bands_data = np.random.rand(9, 256, 256).astype(np.float32)
            return {'rgb_image': rgb_image, 'bands_data': bands_data}

        # Create RGB from tile
        blue = tile_data[0, :, :]
        green = tile_data[1, :, :]
        red = tile_data[2, :, :]

        def normalize(band):
            vmin, vmax = np.percentile(band, [2, 98])
            band = np.clip((band - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255)
            return band.astype(np.uint8)

        rgb_image = np.stack([normalize(red), normalize(green), normalize(blue)], axis=-1)

        return {
            'rgb_image': rgb_image,
            'bands_data': tile_data
        }

    # Ultimate fallback: synthetic data
    rgb_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    bands_data = np.random.rand(9, 256, 256).astype(np.float32)

    return {
        'rgb_image': rgb_image,
        'bands_data': bands_data
    }


def classify_image(bands_data, apply_postprocessing=False, year='current'):
    """
    Run 7-class model inference, then remap to 9 display classes.

    Model classes (trained labels vs actual spectral meaning):
        0=Background (mixed), 1=Roads (actually water), 2=Water (actually forest),
        3=Forest (actually moderate veg), 4=Plantation (low veg), 5=Buildings (urban), 6=Bare Land

    Display classes:
        0=Clouds, 1=Water, 2=Forest, 3=Plantation, 4=Urban, 5=Roads, 6=Bare Land, 7=Ocean, 8=Wasteland
    """
    input_tensor = torch.from_numpy(bands_data).float().unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        output = MODEL(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze(0)
        prediction = prediction.cpu().numpy()

    # Debug: show model output before remapping
    unique, counts = np.unique(prediction, return_counts=True)
    print(f"Model output classes (7-class): {dict(zip(unique, counts))}")

    # Remap 7 model classes → 9 display classes using spectral indices
    display = remap_7_to_9(prediction, bands_data)

    # Smooth the classification to remove salt-and-pepper noise
    display = median_filter(display.astype(np.int32), size=5).astype(np.int64)

    unique2, counts2 = np.unique(display, return_counts=True)
    print(f"Display classes (9-class): {dict(zip(unique2, counts2))}")

    return display


def remap_7_to_9(prediction, bands_data):
    """
    Remap 7-class model output to 9 display classes.

    Uses spectral indices (NDVI, NDWI, NDBI) from bands_data to split
    the model's coarse classes into finer land cover types.

    Thresholds calibrated from spectral analysis of 195 Mauritius tiles.
    Target: Forest ~27%, Wasteland ~26%, Plantation ~24%, Urban ~12%, Roads ~7%, Bare Land ~4%

    bands_data shape: (9, H, W) - B2, B3, B4, B8, B11, B12, NDVI, NDWI, NDBI
    """
    display = np.zeros_like(prediction)

    # Extract spectral indices
    ndvi = bands_data[6]
    ndwi = bands_data[7]
    ndbi = bands_data[8]
    blue, green, red = bands_data[0], bands_data[1], bands_data[2]

    # Normalize if in raw Sentinel-2 scale
    if np.abs(ndvi).max() > 1.5:
        ndvi = np.clip(ndvi / 10000.0, -1, 1)
    if np.abs(ndwi).max() > 1.5:
        ndwi = np.clip(ndwi / 10000.0, -1, 1)
    if np.abs(ndbi).max() > 1.5:
        ndbi = np.clip(ndbi / 10000.0, -1, 1)

    # Brightness for cloud detection
    b, g, r = blue.copy(), green.copy(), red.copy()
    if b.max() > 100:
        b, g, r = b / 10000.0, g / 10000.0, r / 10000.0
    brightness = (b + g + r) / 3.0

    # ── Model class 0 (Background) → Bare Land / Clouds / Water ──
    bg = prediction == 0
    display[bg] = 6  # Default: Bare Land
    display[bg & (brightness > 0.25) & (ndwi <= 0.0)] = 0    # Clouds (bright, not water)
    display[bg & (ndwi > 0.3)] = 7                             # Ocean (strong water signal)
    display[bg & (ndwi > 0.0) & (ndwi <= 0.3)] = 1            # Shallow water
    display[bg & (ndwi <= 0.0) & (brightness <= 0.25) & (ndvi >= 0.3)] = 8  # Vegetated bg → Wasteland

    # ── Model class 1 (Water in model) → Display Water ──
    display[prediction == 1] = 1

    # ── Model class 2 (dominant vegetation, ~71% of land) ──
    # Split using NDBI (calibrated from P25=-0.37, P50=-0.29, P75=-0.21 of mod2)
    # Forest: very low NDBI = dense canopy blocking SWIR
    # Plantation: moderate NDBI = sugarcane rows with soil exposure
    # Urban: high NDBI = built-up surfaces (concrete, metal roofs)
    # Wasteland: catch-all for moderate vegetation
    mod2 = prediction == 2
    display[mod2] = 8  # Default: Wasteland (catch-all)
    display[mod2 & (ndvi >= 0.4) & (ndbi < -0.33)] = 2    # Forest (~25% of land)
    display[mod2 & (ndvi >= 0.4) & (ndbi >= -0.33) & (ndbi < -0.24)] = 3  # Plantation (~22%)
    display[mod2 & (ndvi < 0.15) & (ndbi < -0.15)] = 6    # Bare Land (very low NDVI)
    display[mod2 & (ndbi >= -0.15)] = 4                     # Urban (~9%, high SWIR = built-up)

    # ── Model class 3 (moderate vegetation, ~19.5% of land) ──
    mod3 = prediction == 3
    display[mod3] = 8  # Default: Wasteland
    display[mod3 & (ndvi >= 0.5) & (ndbi < -0.35)] = 2    # Dense veg → Forest
    display[mod3 & (ndvi >= 0.35) & (ndbi >= -0.35) & (ndbi < -0.24)] = 3  # Plantation
    display[mod3 & (ndbi >= -0.15)] = 4                     # Urban
    display[mod3 & (ndvi < 0.2)] = 5                        # Low NDVI in mod3 → Roads

    # ── Model class 4 (low/sparse vegetation, ~4.5% of land) → Roads ──
    mod4 = prediction == 4
    display[mod4] = 5  # Default: Roads
    display[mod4 & (ndvi >= 0.4)] = 8                       # High NDVI → actually Wasteland
    display[mod4 & (ndbi >= -0.1)] = 4                      # High NDBI → Urban

    # ── Model class 5 (Buildings) → Urban ──
    display[prediction == 5] = 4

    # ── Model class 6 (Bare Land) → Bare Land ──
    display[prediction == 6] = 6

    return display



def create_colored_mask(prediction):
    """Convert class indices to RGB"""
    h, w = prediction.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, info in CLASSES.items():
        mask = prediction == class_id
        colored[mask] = info['color']

    return colored


def get_class_statistics(prediction):
    """Calculate land-only class distribution (exclude Clouds, Water, Ocean)"""
    unique, counts = np.unique(prediction, return_counts=True)
    count_map = dict(zip(unique, counts))

    # Exclude non-land classes: Clouds(0), Water(1), Ocean(7)
    excluded = {0, 1, 7}
    land_pixels = sum(c for cid, c in count_map.items() if cid not in excluded and cid in CLASSES)

    if land_pixels == 0:
        land_pixels = 1  # Avoid division by zero

    stats = []
    for class_id, count in count_map.items():
        if class_id in CLASSES and class_id not in excluded:
            percentage = (count / land_pixels) * 100
            stats.append({
                'name': CLASSES[class_id]['name'],
                'color': CLASSES[class_id]['color'],
                'percentage': round(percentage, 2),
                'pixels': int(count)
            })

    stats.sort(key=lambda x: x['percentage'], reverse=True)
    return stats


def array_to_base64(img_array):
    """Convert numpy array to base64"""
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Live Interactive Map - Mauritius Land Cover</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow-x: auto;
            overflow-y: auto;
        }

        #header {
            background: linear-gradient(135deg, #2d5016 0%, #4a7c24 100%);
            color: white;
            padding: 15px 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            z-index: 1000;
            position: relative;
        }

        #header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }

        #header p {
            font-size: 13px;
            opacity: 0.95;
        }

        #info-bar {
            background: #f8f9fa;
            padding: 10px 30px;
            border-bottom: 2px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
        }

        #coordinates {
            font-size: 14px;
            font-weight: 500;
            color: #333;
        }

        .year-selector-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .year-selector-container label {
            font-size: 14px;
            font-weight: 600;
            color: #333;
        }

        #year-selector {
            padding: 8px 15px;
            font-size: 14px;
            border: 2px solid #4a7c24;
            border-radius: 8px;
            background: white;
            color: #333;
            cursor: pointer;
            font-weight: 500;
            min-width: 180px;
        }

        #year-selector:hover {
            border-color: #2d5016;
            background: #f0f7e6;
        }

        #year-selector:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(74, 124, 36, 0.2);
        }

        #status {
            font-size: 13px;
            padding: 5px 15px;
            border-radius: 20px;
            background: #e8f5e9;
            color: #2e7d32;
        }

        #status.loading {
            background: #fff3e0;
            color: #f57c00;
        }

        #container {
            display: grid;
            grid-template-columns: 320px 1fr 1fr 280px 320px;
            min-height: calc(100vh - 120px);
            gap: 0;
            overflow-x: auto;
        }

        #change-chart-container {
            padding: 10px;
            height: 100%;
        }

        #change-chart-container canvas {
            max-height: 250px;
        }

        .chart-title {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }

        .historical-summary {
            margin-top: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 12px;
        }

        .historical-summary h4 {
            margin-bottom: 8px;
            color: #2d5016;
        }

        .change-indicator {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            border-bottom: 1px solid #eee;
        }

        .change-indicator:last-child {
            border-bottom: none;
        }

        .change-positive {
            color: #2e7d32;
        }

        .change-negative {
            color: #c62828;
        }

        .panel {
            background: white;
            border-right: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
            min-width: 180px;
        }

        .panel:last-child {
            border-right: none;
        }

        .panel:nth-child(2),
        .panel:nth-child(3) {
            min-width: 250px;
        }

        .panel-header {
            padding: 15px 20px;
            font-weight: 600;
            font-size: 15px;
            border-bottom: 2px solid #f0f0f0;
            background: #fafafa;
        }

        .panel-content {
            flex: 1;
            overflow: auto;
            padding: 15px;
        }

        #map {
            height: 100%;
            width: 100%;
        }

        #mauritius-map {
            height: 100%;
            width: 100%;
            min-height: 200px;
            background-color: #a8d8ea;  /* Light blue for ocean */
        }

        .dual-image-container {
            display: flex;
            flex-direction: column;
            gap: 8px;
            height: 100%;
            width: 100%;
        }

        .image-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }

        .image-label {
            text-align: center;
            font-size: 11px;
            font-weight: 600;
            color: #666;
            padding: 3px 0;
            background: #f5f5f5;
            border-radius: 4px 4px 0 0;
        }

        .image-container {
            text-align: center;
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #fafafa;
            border-radius: 0 0 4px 4px;
        }

        .image-container img {
            max-width: 100%;
            max-height: 100%;
            border: 2px solid #e0e0e0;
            border-radius: 4px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            margin: 12px 0;
            padding: 10px 12px;
            background: #f8f9fa;
            border-radius: 6px;
            transition: transform 0.2s;
        }

        .legend-item:hover {
            transform: translateX(5px);
            background: #e9ecef;
        }

        .legend-color {
            width: 32px;
            height: 32px;
            border-radius: 6px;
            margin-right: 15px;
            border: 2px solid #ddd;
        }

        .legend-text {
            flex: 1;
            font-size: 14px;
            font-weight: 500;
        }

        .legend-percentage, .legend-value {
            font-weight: 700;
            font-size: 14px;
            color: #333;
            min-width: 55px;
            text-align: right;
        }

        .legend-bar-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .legend-bar-bg {
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }

        .legend-bar {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        #loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            display: none;
            z-index: 9999;
            align-items: center;
            justify-content: center;
        }

        .loading-content {
            background: white;
            padding: 40px 60px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }

        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #4CAF50;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            font-size: 18px;
            color: #333;
            margin-bottom: 10px;
        }

        .loading-subtext {
            font-size: 14px;
            color: #666;
        }

        /* Responsive Design */
        @media (max-width: 1600px) {
            #container {
                grid-template-columns: 250px 1fr 1fr 220px 280px;
            }
        }

        @media (max-width: 1400px) {
            #container {
                grid-template-columns: 220px 1fr 1fr 200px 250px;
            }

            .panel-header {
                padding: 12px 15px;
                font-size: 14px;
            }
        }

        @media (max-width: 1200px) {
            #container {
                grid-template-columns: 180px 1fr 1fr 180px 220px;
            }

            #header h1 {
                font-size: 20px;
            }

            #info-bar {
                padding: 8px 15px;
                gap: 15px;
            }

            .legend-item {
                margin: 6px 0;
                padding: 6px 8px;
            }

            .legend-color {
                width: 24px;
                height: 24px;
                margin-right: 10px;
            }

            .legend-text {
                font-size: 12px;
            }

            .legend-percentage {
                font-size: 12px;
            }
        }

        @media (max-width: 1024px) {
            #container {
                grid-template-columns: 1fr 1fr;
                grid-template-rows: auto auto auto auto;
                height: auto;
                min-height: calc(100vh - 120px);
            }

            .panel {
                border-right: none;
                border-bottom: 1px solid #e0e0e0;
            }

            .panel:nth-child(1) { /* Satellite panel */
                order: 3;
                grid-column: 1;
                min-height: 300px;
            }

            .panel:nth-child(2) { /* Map panel */
                order: 1;
                grid-column: 1 / -1;
                min-height: 350px;
            }

            .panel:nth-child(3) { /* Mauritius Map panel */
                order: 2;
                grid-column: 1 / -1;
                min-height: 300px;
            }

            .panel:nth-child(4) { /* Legend panel */
                order: 4;
                grid-column: 2;
                min-height: 300px;
            }

            .panel:nth-child(5) { /* Historical panel */
                order: 5;
                grid-column: 1 / -1;
            }

            #header h1 {
                font-size: 18px;
            }

            #info-bar {
                flex-wrap: wrap;
                gap: 10px;
                padding: 10px 15px;
            }
        }

        @media (max-width: 768px) {
            #container {
                grid-template-columns: 1fr;
            }

            .panel:nth-child(1),
            .panel:nth-child(4),
            .panel:nth-child(5) {
                grid-column: 1;
            }

            .panel:nth-child(2) {
                min-height: 300px;
            }

            .panel:nth-child(3) {
                min-height: 280px;
            }

            #header {
                padding: 12px 15px;
            }

            #header h1 {
                font-size: 16px;
                line-height: 1.3;
            }

            #header p {
                font-size: 11px;
            }

            #info-bar {
                flex-direction: column;
                align-items: stretch;
                gap: 8px;
            }

            .year-selector-container {
                justify-content: space-between;
            }

            #year-selector {
                min-width: 150px;
            }

            #status {
                text-align: center;
            }

            .panel-header {
                padding: 10px 12px;
                font-size: 13px;
            }

            .panel-content {
                padding: 10px;
            }

            .legend-item {
                margin: 6px 0;
                padding: 6px 8px;
            }

            .legend-color {
                width: 20px;
                height: 20px;
                margin-right: 8px;
            }

            .legend-text {
                font-size: 11px;
            }

            .legend-percentage {
                font-size: 11px;
                min-width: 40px;
            }

            .loading-content {
                padding: 25px 30px;
                margin: 15px;
            }

            .spinner {
                width: 40px;
                height: 40px;
            }

            .loading-text {
                font-size: 16px;
            }
        }

        @media (max-width: 480px) {
            #header h1 {
                font-size: 14px;
            }

            #header p {
                font-size: 10px;
            }

            .panel:nth-child(2) {
                min-height: 250px;
            }

            .panel:nth-child(3) {
                min-height: 220px;
            }

            #year-selector {
                min-width: 120px;
                padding: 6px 10px;
                font-size: 13px;
            }

            .year-selector-container label {
                font-size: 12px;
            }

            .dual-image-container {
                gap: 5px;
            }

            .image-label {
                font-size: 10px;
            }
        }
    </style>
</head>
<body>
    <div id="header">
        <h1>🇲🇺 Live Interactive Map - Mauritius Land Cover Classification</h1>
        <p>Pan around the map to automatically fetch and classify Sentinel-2 imagery</p>
    </div>

    <div id="info-bar">
        <div id="coordinates">Lat: --, Lon: --</div>
        <div class="year-selector-container">
            <label for="year-selector">Imagery Year:</label>
            <select id="year-selector">
                <option value="current">Current (Last 60 days)</option>
                <option value="2022">2022</option>
                <option value="2019">2019</option>
                <option value="2016">2016</option>
                <option value="2013">2013 (Landsat)</option>
                <option value="2010">2010 (Landsat)</option>
            </select>
        </div>
        <div id="status">Ready</div>
    </div>

    <div id="loading-overlay">
        <div class="loading-content">
            <div class="spinner"></div>
            <div class="loading-text">Fetching Sentinel-2 Imagery...</div>
            <div class="loading-subtext">Downloading and classifying</div>
        </div>
    </div>

    <div id="container">
        <div class="panel">
            <div class="panel-header">🗺️ Interactive Map</div>
            <div class="panel-content" style="padding: 0;">
                <div id="map"></div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">🛰️ Satellite Image & Classification</div>
            <div class="panel-content">
                <div class="dual-image-container">
                    <div class="image-wrapper">
                        <div class="image-label">Sentinel-2</div>
                        <div class="image-container" id="satellite-container">
                            <p style="color: #999;">Pan the map to load imagery</p>
                        </div>
                    </div>
                    <div class="image-wrapper">
                        <div class="image-label">Classification</div>
                        <div class="image-container" id="classification-container">
                            <p style="color: #999;">Classification will appear here</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">🗺️ Mauritius Classification</div>
            <div class="panel-content" style="padding: 0; position: relative; height: calc(100% - 40px);">
                <div id="mauritius-map"></div>
                <div id="mauritius-loading" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); display: none; background: rgba(255,255,255,0.9); padding: 15px 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); z-index: 1000;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div class="spinner" style="width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #4a7c24; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                        <span>Classifying Mauritius...</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">📊 Class Distribution</div>
            <div class="panel-content">
                <div id="legend">
                    <p style="color: #999;">Statistics will appear here</p>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">📈 Historical Change Detection</div>
            <div class="panel-content">
                <div id="change-chart-container">
                    <div class="chart-title">Land Cover Changes Over Time</div>
                    <canvas id="changeChart"></canvas>
                    <div id="historical-summary" class="historical-summary">
                        <h4>Change Summary (2010 vs Current)</h4>
                        <p style="color: #999; font-size: 11px;">Click "Load Historical Comparison" to analyze changes</p>
                    </div>
                    <button id="load-historical-btn" style="margin-top: 10px; padding: 8px 16px; background: #4a7c24; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; width: 100%;">
                        Load Historical Comparison
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize map centered on Mauritius
        const map = L.map('map').setView([{{ center_lat }}, {{ center_lon }}], 11);

        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18
        }).addTo(map);

        // Add marker for current location
        let marker = L.marker([{{ center_lat }}, {{ center_lon }}]).addTo(map);

        let isLoading = false;
        let lastFetchedLat = null;
        let lastFetchedLon = null;
        let lastFetchedYear = 'current';
        let changeChart = null;
        let historicalData = {};

        // Class colors for chart
        const classColors = {
            'Water': 'rgba(168, 216, 234, 0.8)',
            'Forest': 'rgba(34, 139, 34, 0.8)',
            'Plantation': 'rgba(218, 165, 32, 0.8)',
            'Urban': 'rgba(215, 204, 200, 0.8)',
            'Roads': 'rgba(158, 158, 158, 0.8)',
            'Bare Land': 'rgba(239, 235, 233, 0.8)',
            'Clouds': 'rgba(255, 255, 255, 0.8)',
            'Ocean': 'rgba(135, 190, 220, 0.8)',
            'Wasteland': 'rgba(189, 183, 107, 0.8)'
        };

        // Initialize Mauritius overview map (no base map - classification only)
        const mauritiusMap = L.map('mauritius-map', {
            zoomControl: false,
            attributionControl: false
        }).setView([-20.25, 57.55], 10);

        // Force map to recalculate size after DOM is ready
        setTimeout(() => mauritiusMap.invalidateSize(), 100);

        let mauritiusOverlay = null;
        let currentMauritiusYear = null;

        // Function to load full island classification
        async function loadMauritiusClassification(year, force = false) {
            console.log('loadMauritiusClassification called with year:', year, 'current:', currentMauritiusYear, 'force:', force);

            if (!force && currentMauritiusYear === year) {
                console.log('Skipping - already loaded year:', year);
                return;
            }

            document.getElementById('mauritius-loading').style.display = 'block';

            try {
                console.log('Fetching classification for year:', year);
                const response = await fetch('/api/classify_mauritius', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ year: year })
                });

                const data = await response.json();
                console.log('Received data for year:', year, 'statistics:', data.statistics);

                if (data.classification_image) {
                    // Remove old overlay
                    if (mauritiusOverlay) {
                        mauritiusMap.removeLayer(mauritiusOverlay);
                    }

                    // Add new classification overlay
                    const bounds = [
                        [data.bounds.south, data.bounds.west],
                        [data.bounds.north, data.bounds.east]
                    ];

                    mauritiusOverlay = L.imageOverlay(
                        'data:image/png;base64,' + data.classification_image,
                        bounds,
                        { opacity: 1.0 }
                    ).addTo(mauritiusMap);

                    currentMauritiusYear = year;

                    // Update Class Distribution panel with island-wide stats
                    if (data.statistics) {
                        console.log('Updating legend with statistics:', data.statistics);
                        updateIslandLegend(data.statistics, year);
                    }

                    // Store for historical comparison
                    historicalData[year] = data.statistics;
                }
            } catch (error) {
                console.error('Error loading Mauritius classification:', error);
            }

            document.getElementById('mauritius-loading').style.display = 'none';
        }

        // Function to update legend with island-wide statistics
        function updateIslandLegend(statistics, year) {
            const legendDiv = document.getElementById('legend');
            const yearLabel = year === 'current' ? 'Current' : year;

            // Filter out Clouds, Water, and Ocean (show land classes only), sort by percentage descending
            const landStats = statistics
                .filter(stat => stat.class !== 'Clouds' && stat.class !== 'Water' && stat.class !== 'Ocean')
                .sort((a, b) => b.percentage - a.percentage);

            // Find max percentage for bar scaling (so largest bar is 100% width)
            const maxPercentage = Math.max(...landStats.map(s => s.percentage));

            let html = `<div style="font-size: 12px; color: #333; margin-bottom: 12px; text-align: center; font-weight: 600;">🏝️ Mauritius Land Cover</div>`;
            html += `<div style="font-size: 10px; color: #888; margin-bottom: 15px; text-align: center;">${yearLabel} Classification</div>`;

            landStats.forEach(stat => {
                const color = classColors[stat.class] || 'rgba(200,200,200,0.8)';
                const rgbColor = color.replace('0.8)', '1)');
                const barColor = color.replace('rgba', 'rgb').replace(', 0.8)', ')');
                // Scale bar width relative to max, so bars are visible
                const barWidth = (stat.percentage / maxPercentage) * 100;
                html += `
                    <div class="legend-item" style="flex-direction: column; align-items: stretch; padding: 8px 12px;">
                        <div style="display: flex; align-items: center; margin-bottom: 6px;">
                            <div class="legend-color" style="background: ${rgbColor}; width: 24px; height: 24px; margin-right: 10px;"></div>
                            <div class="legend-text" style="flex: 1;">${stat.class}</div>
                            <div class="legend-value">${stat.percentage.toFixed(1)}%</div>
                        </div>
                        <div class="legend-bar-bg">
                            <div class="legend-bar" style="width: ${barWidth}%; background: ${barColor};"></div>
                        </div>
                    </div>
                `;
            });

            legendDiv.innerHTML = html;
        }

        // Load initial classification based on dropdown selection
        setTimeout(() => {
            const initialYear = document.getElementById('year-selector').value;
            loadMauritiusClassification(initialYear, true);
        }, 1000);

        // Function to fetch and classify
        function fetchAndClassify(lat, lon, forceRefresh = false) {
            const selectedYear = document.getElementById('year-selector').value;

            // Don't fetch if already loading
            if (isLoading) return;

            // Don't fetch if coordinates and year haven't changed much
            if (!forceRefresh && lastFetchedLat !== null && lastFetchedLon !== null) {
                const latDiff = Math.abs(lat - lastFetchedLat);
                const lonDiff = Math.abs(lon - lastFetchedLon);
                if (latDiff < 0.01 && lonDiff < 0.01 && selectedYear === lastFetchedYear) {
                    return;
                }
            }

            isLoading = true;
            lastFetchedLat = lat;
            lastFetchedLon = lon;
            lastFetchedYear = selectedYear;

            // Update UI
            document.getElementById('coordinates').textContent =
                `Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`;
            document.getElementById('status').textContent = 'Fetching ' + (selectedYear === 'current' ? 'current' : selectedYear) + ' imagery...';
            document.getElementById('status').className = 'loading';
            document.getElementById('loading-overlay').style.display = 'flex';

            // Update loading text
            const satellite = ['2010', '2013'].includes(selectedYear) ? 'Landsat' : 'Sentinel-2';
            document.querySelector('.loading-text').textContent = `Fetching ${satellite} ${selectedYear === 'current' ? '' : selectedYear} Imagery...`;

            // Update marker
            marker.setLatLng([lat, lon]);

            // Fetch from backend with year parameter
            fetch('/api/classify_location', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lat, lon, year: selectedYear })
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }

                // Update panel header to show satellite type
                const panelHeader = document.querySelector('.panel:nth-child(2) .panel-header');
                if (['2010', '2013'].includes(selectedYear)) {
                    panelHeader.textContent = '🛰️ Landsat ' + selectedYear + ' Satellite Image';
                } else if (selectedYear === 'current') {
                    panelHeader.textContent = '🛰️ Sentinel-2 Satellite Image (Current)';
                } else {
                    panelHeader.textContent = '🛰️ Sentinel-2 ' + selectedYear + ' Satellite Image';
                }

                // Update images
                document.getElementById('satellite-container').innerHTML =
                    '<img src="data:image/png;base64,' + data.satellite + '" alt="Satellite">';

                document.getElementById('classification-container').innerHTML =
                    '<img src="data:image/png;base64,' + data.classification + '" alt="Classification">';

                // Store statistics for historical comparison (don't update legend - keep island-wide stats)
                historicalData[selectedYear] = data.statistics;

                // Update status
                document.getElementById('status').textContent = 'Ready';
                document.getElementById('status').className = '';
                document.getElementById('loading-overlay').style.display = 'none';
                isLoading = false;
            })
            .catch(err => {
                console.error(err);
                document.getElementById('status').textContent = 'Error: ' + err.message;
                document.getElementById('status').className = '';
                document.getElementById('loading-overlay').style.display = 'none';
                isLoading = false;
            });
        }

        // Year selector change handler
        document.getElementById('year-selector').addEventListener('change', function() {
            const selectedYear = this.value;
            console.log('Year selector changed to:', selectedYear);
            // Update local tile classification
            if (lastFetchedLat !== null && lastFetchedLon !== null) {
                fetchAndClassify(lastFetchedLat, lastFetchedLon, true);
            }
            // Update full island classification - force reload
            loadMauritiusClassification(selectedYear, true);
        });

        // Load historical comparison button handler
        document.getElementById('load-historical-btn').addEventListener('click', function() {
            if (lastFetchedLat === null || lastFetchedLon === null) {
                alert('Please select a location on the map first');
                return;
            }

            const btn = this;
            btn.textContent = 'Loading historical data...';
            btn.disabled = true;

            // Fetch historical comparison data
            fetch('/api/historical_comparison', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    lat: lastFetchedLat,
                    lon: lastFetchedLon,
                    years: ['2010', '2016', '2022', 'current']
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }

                // Update chart with historical data
                updateChangeChart(data);

                // Update summary
                updateHistoricalSummary(data);

                btn.textContent = 'Refresh Historical Data';
                btn.disabled = false;
            })
            .catch(err => {
                console.error(err);
                btn.textContent = 'Load Historical Comparison';
                btn.disabled = false;
                alert('Error loading historical data: ' + err.message);
            });
        });

        // Update the change detection chart
        function updateChangeChart(data) {
            const ctx = document.getElementById('changeChart').getContext('2d');

            if (changeChart) {
                changeChart.destroy();
            }

            const years = Object.keys(data.statistics).sort();
            const classes = ['Urban', 'Forest', 'Plantation', 'Water'];

            const datasets = classes.map(className => ({
                label: className,
                data: years.map(year => {
                    const yearStats = data.statistics[year];
                    const classStat = yearStats.find(s => s.name === className);
                    return classStat ? classStat.percentage : 0;
                }),
                backgroundColor: classColors[className],
                borderColor: classColors[className].replace('0.8', '1'),
                borderWidth: 2,
                fill: false,
                tension: 0.3
            }));

            changeChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: years.map(y => y === 'current' ? 'Current' : y),
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { font: { size: 10 } }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: { display: true, text: '% Coverage', font: { size: 10 } }
                        },
                        x: {
                            title: { display: true, text: 'Year', font: { size: 10 } }
                        }
                    }
                }
            });
        }

        // Update historical summary
        function updateHistoricalSummary(data) {
            const years = Object.keys(data.statistics).sort();
            const firstYear = years[0];
            const lastYear = years[years.length - 1];

            const firstStats = data.statistics[firstYear];
            const lastStats = data.statistics[lastYear];

            let summaryHtml = `<h4>Change: ${firstYear === 'current' ? 'Current' : firstYear} → ${lastYear === 'current' ? 'Current' : lastYear}</h4>`;

            ['Urban', 'Forest', 'Plantation'].forEach(className => {
                const first = firstStats.find(s => s.name === className);
                const last = lastStats.find(s => s.name === className);

                if (first && last) {
                    const change = (last.percentage - first.percentage).toFixed(1);
                    const changeClass = parseFloat(change) >= 0 ? 'change-positive' : 'change-negative';
                    const arrow = parseFloat(change) >= 0 ? '↑' : '↓';

                    summaryHtml += `
                        <div class="change-indicator">
                            <span>${className}</span>
                            <span class="${changeClass}">${arrow} ${Math.abs(change)}%</span>
                        </div>
                    `;
                }
            });

            document.getElementById('historical-summary').innerHTML = summaryHtml;
        }

        // Fetch on map move end (after pan/zoom)
        map.on('moveend', function() {
            const center = map.getCenter();
            fetchAndClassify(center.lat, center.lng);
        });

        // Fetch on click
        map.on('click', function(e) {
            fetchAndClassify(e.latlng.lat, e.latlng.lng, true);
        });

        // Initial fetch
        setTimeout(() => {
            fetchAndClassify({{ center_lat }}, {{ center_lon }});
        }, 1000);
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Main page"""
    return render_template_string(
        HTML_TEMPLATE,
        center_lat=MAURITIUS_BOUNDS['center_lat'],
        center_lon=MAURITIUS_BOUNDS['center_lon']
    )


@app.route('/api/classify_location', methods=['POST'])
def classify_location():
    """Download and classify satellite imagery for location and year"""
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        year = data.get('year', 'current')

        print(f"\n{'='*60}")
        print(f"Classifying location: {lat:.4f}, {lon:.4f} for year: {year}")

        # Download imagery for the specified year
        image_data = download_imagery_for_year(lat, lon, year)

        # Classify (pass year for correct NDVI thresholds)
        prediction = classify_image(image_data['bands_data'], year=year)

        # Create colored visualization
        classification_colored = create_colored_mask(prediction)

        # Get statistics
        statistics = get_class_statistics(prediction)

        # Convert to base64
        satellite_b64 = array_to_base64(image_data['rgb_image'])
        classification_b64 = array_to_base64(classification_colored)

        print(f"Classification complete for {year}!")
        print(f"{'='*60}\n")

        return jsonify({
            'satellite': satellite_b64,
            'classification': classification_b64,
            'statistics': statistics,
            'year': year
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/historical_comparison', methods=['POST'])
def historical_comparison():
    """Get land cover statistics for multiple years for comparison"""
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        years = data.get('years', ['2010', '2016', '2022', 'current'])

        print(f"\n{'='*60}")
        print(f"Historical comparison for: {lat:.4f}, {lon:.4f}")
        print(f"Years: {years}")

        all_statistics = {}

        for year in years:
            print(f"Processing {year}...")
            try:
                # Download imagery for this year
                image_data = download_imagery_for_year(lat, lon, year)

                # Classify (pass year for correct NDVI thresholds)
                prediction = classify_image(image_data['bands_data'], year=year)

                # Get statistics
                statistics = get_class_statistics(prediction)

                all_statistics[year] = statistics
                print(f"  {year} complete")

            except Exception as e:
                print(f"  Error for {year}: {e}")
                all_statistics[year] = []

        print(f"Historical comparison complete!")
        print(f"{'='*60}\n")

        return jsonify({
            'statistics': all_statistics,
            'location': {'lat': lat, 'lon': lon}
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Directory for cached full-island classification images
CLASSIFICATION_CACHE_DIR = Path('data/classification_cache')
CLASSIFICATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Directory for cached raw satellite tile data (never re-downloaded)
RAW_TILES_CACHE_DIR = CLASSIFICATION_CACHE_DIR / 'raw_tiles'
RAW_TILES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Mauritius bounding box (constant)
MAURITIUS_BOUNDS_FULL = {
    'north': -19.98,
    'south': -20.52,
    'west': 57.30,
    'east': 57.80
}


def get_cached_classification_path(year):
    """Get file paths for cached classification data"""
    return {
        'image': CLASSIFICATION_CACHE_DIR / f'mauritius_{year}.png',
        'stats': CLASSIFICATION_CACHE_DIR / f'mauritius_{year}_stats.json'
    }


def load_cached_classification(year):
    """Load classification from disk if it exists"""
    paths = get_cached_classification_path(year)

    if paths['image'].exists() and paths['stats'].exists():
        print(f"Loading cached classification for {year} from disk")

        # Load image as base64
        with open(paths['image'], 'rb') as f:
            classification_b64 = base64.b64encode(f.read()).decode()

        # Load statistics
        with open(paths['stats'], 'r') as f:
            stats_data = json.load(f)

        return {
            'classification_image': classification_b64,
            'bounds': MAURITIUS_BOUNDS_FULL,
            'statistics': stats_data['statistics'],
            'year': year
        }

    return None


def save_classification_to_cache(year, image_data, statistics):
    """Save classification image and stats to disk"""
    paths = get_cached_classification_path(year)

    # Save image
    image_bytes = base64.b64decode(image_data)
    with open(paths['image'], 'wb') as f:
        f.write(image_bytes)

    # Save statistics
    with open(paths['stats'], 'w') as f:
        json.dump({'statistics': statistics, 'year': year}, f)

    print(f"Saved classification for {year} to {paths['image']}")


@app.route('/api/classify_mauritius', methods=['POST'])
def classify_mauritius():
    """
    Classify the full island of Mauritius using hierarchical mosaic approach.

    The key insight: each 256x256 tile represents exactly 2km x 2km.
    We create a grid where each cell is 2km, download/classify each cell,
    then assemble them in the correct geographic order.
    """
    try:
        data = request.get_json()
        year = data.get('year', 'current')
        force_reclassify = data.get('force_reclassify', False)

        # Check disk cache first (skip if force reclassify requested)
        if not force_reclassify:
            cached = load_cached_classification(year)
            if cached:
                print(f"Using CACHED classification for {year} - skipping re-run")
                return jsonify(cached)

        print(f"\n{'='*60}")
        print(f"Full Mauritius classification for year: {year}")
        print(f"Using mosaic approach (2km tiles)...")

        bounds = MAURITIUS_BOUNDS_FULL

        # Each 256x256 tile = 2km x 2km
        # Convert 2km to degrees at Mauritius latitude (~-20.25)
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(20.25))  # ~104 km/deg

        tile_size_km = 2.0
        tile_deg_lat = tile_size_km / km_per_deg_lat  # ~0.018 deg
        tile_deg_lon = tile_size_km / km_per_deg_lon  # ~0.019 deg

        # Calculate grid dimensions
        lat_range = bounds['north'] - bounds['south']  # ~0.54 deg = ~60km
        lon_range = bounds['east'] - bounds['west']    # ~0.50 deg = ~52km

        # Use a fixed grid size - since we cache the result, processing time is okay
        # 15x13 grid gives good coverage (~195 tiles, will take ~15-20 min first time)
        n_rows = 15  # covers ~60km with ~4km per tile
        n_cols = 13  # covers ~52km with ~4km per tile

        # Calculate step size and tile size to match
        step_lat = lat_range / n_rows
        step_lon = lon_range / n_cols

        # Each tile should cover exactly one grid cell
        # Convert step degrees to km for download
        tile_size_km_lat = step_lat * km_per_deg_lat  # ~4km
        tile_size_km_lon = step_lon * km_per_deg_lon  # ~4km
        tile_size_km = max(tile_size_km_lat, tile_size_km_lon)  # Use larger to ensure coverage

        print(f"  Grid: {n_rows} rows x {n_cols} cols = {n_rows * n_cols} tiles")
        print(f"  Step: {step_lat:.4f} deg lat, {step_lon:.4f} deg lon")
        print(f"  Tile size: {tile_size_km:.1f} km (to cover each grid cell)")

        # Create composite array
        composite_h = 256 * n_rows
        composite_w = 256 * n_cols
        composite = np.zeros((composite_h, composite_w), dtype=np.uint8)

        total_statistics = {}
        valid_tiles = 0

        # Raw tile cache for this year
        year_tile_dir = RAW_TILES_CACHE_DIR / str(year)
        year_tile_dir.mkdir(parents=True, exist_ok=True)

        # Count cached vs download needed
        cached_count = 0
        download_count = 0

        # Process tiles: row 0 = southernmost, col 0 = westernmost
        for row in range(n_rows):
            for col in range(n_cols):
                # Tile center coordinates
                center_lat = bounds['south'] + (row + 0.5) * step_lat
                center_lon = bounds['west'] + (col + 0.5) * step_lon

                tile_num = row * n_cols + col + 1
                raw_tile_path = year_tile_dir / f"tile_{row:02d}_{col:02d}.npy"

                try:
                    # Check if raw satellite data is already cached
                    if raw_tile_path.exists():
                        bands_data = np.load(raw_tile_path)
                        cached_count += 1
                        if tile_num % 20 == 0 or tile_num == 1:
                            print(f"  [{row},{col}] Tile {tile_num}/{n_rows*n_cols} CACHED")
                    else:
                        print(f"  [{row},{col}] Tile {tile_num}/{n_rows*n_cols} downloading...")
                        image_data = download_imagery_for_year(center_lat, center_lon, year, size_km=tile_size_km)
                        bands_data = image_data['bands_data']
                        # Save raw data to cache
                        np.save(raw_tile_path, bands_data)
                        download_count += 1

                    # Classify the tile (always re-run — this is fast)
                    prediction = classify_image(bands_data, year=year)

                    # Place in composite:
                    # - row 0 (south) should be at BOTTOM of image (high y)
                    # - col 0 (west) should be at LEFT of image (low x)
                    y_start = (n_rows - 1 - row) * 256  # row 0 -> bottom
                    x_start = col * 256

                    composite[y_start:y_start+256, x_start:x_start+256] = prediction
                    valid_tiles += 1

                except Exception as e:
                    print(f"    Error: {e}")

        print(f"  Tiles: {cached_count} cached, {download_count} downloaded")

        # Apply ocean mask to the entire composite
        print("  Applying ocean mask to separate inland water from ocean...")
        composite = apply_ocean_mask_composite(composite, bounds)

        # Recalculate statistics after ocean masking — LAND ONLY
        # Exclude Clouds(0), Water(1), Ocean(7) from denominator
        excluded_classes = {0, 1, 7}
        land_pixels = sum(
            np.sum(composite == cid)
            for cid in CLASSES.keys()
            if cid not in excluded_classes
        )
        if land_pixels == 0:
            land_pixels = 1  # Avoid division by zero

        total_statistics = {}
        for class_id, class_info in CLASSES.items():
            if class_id in excluded_classes:
                continue
            count = np.sum(composite == class_id)
            if count > 0:
                total_statistics[class_info['name']] = (count / land_pixels) * 100

        # Create colored RGBA image
        colored = np.zeros((composite_h, composite_w, 4), dtype=np.uint8)
        for class_id, class_info in CLASSES.items():
            mask = composite == class_id
            colored[mask] = class_info['color'] + [255]

        # Convert to PNG
        img = Image.fromarray(colored, 'RGBA')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        classification_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Format statistics
        statistics = [
            {'class': name, 'percentage': round(pct, 2)}
            for name, pct in sorted(total_statistics.items(), key=lambda x: -x[1])
        ]

        # Save to cache
        save_classification_to_cache(year, classification_b64, statistics)

        print(f"Complete! {valid_tiles}/{n_rows*n_cols} tiles processed")
        print(f"{'='*60}\n")

        return jsonify({
            'classification_image': classification_b64,
            'bounds': bounds,
            'statistics': statistics,
            'year': year
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Load model at module level (for gunicorn)
print("=" * 60)
print("Live Interactive Map - Mauritius Land Cover")
print("=" * 60)
load_model()
print(f"\nUsing device: {DEVICE}")
print(f"Google Earth Engine: {'Available' if GEE_AVAILABLE else 'Not available (using fallback)'}")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Starting web server...")
    print("Visit: http://localhost:5003")
    print("=" * 60)
    print("\nHow to use:")
    print("1. Pan around the map to explore different areas")
    print("2. Click on any location to classify")
    print("3. Imagery is fetched automatically as you move")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)
