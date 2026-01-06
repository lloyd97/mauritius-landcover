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
    0: {'name': 'Background', 'color': [245, 243, 240]},      # Warm off-white (Apple Maps background)
    1: {'name': 'Water', 'color': [168, 216, 234]},           # Soft light blue (Apple Maps water)
    2: {'name': 'Forest', 'color': [139, 195, 74]},           # Rich natural green (Apple Maps parks)
    3: {'name': 'Plantation', 'color': [197, 225, 165]},      # Light green (Apple Maps vegetation)
    4: {'name': 'Urban', 'color': [215, 204, 200]},           # Warm gray/tan (Apple Maps buildings)
    5: {'name': 'Roads', 'color': [158, 158, 158]},           # Medium gray (Apple Maps roads)
    6: {'name': 'Bare Land', 'color': [239, 235, 233]}        # Sandy cream (Apple Maps bare areas)
}

# Initialize Earth Engine
GEE_AVAILABLE = False
try:
    # Check for EARTHENGINE_TOKEN environment variable (for cloud deployment)
    ee_token = os.environ.get('EARTHENGINE_TOKEN')
    if ee_token:
        import json
        credentials_dict = json.loads(ee_token)
        credentials = ee.ServiceAccountCredentials(None, key_data=credentials_dict) if 'private_key' in credentials_dict else None
        if credentials:
            ee.Initialize(credentials)
        else:
            # Use refresh token authentication
            credentials = ee.oauth.Credentials(
                token=None,
                refresh_token=credentials_dict.get('refresh_token'),
                token_uri='https://oauth2.googleapis.com/token',
                client_id='517222506229-vsmmajv00ul0bs7p89v5m89ber09i6ko.apps.googleusercontent.com',
                client_secret='RUP0RCmq1Zd5vqYGn5SgO7zh',
                scopes=credentials_dict.get('scopes', [])
            )
            ee.Initialize(credentials, project=credentials_dict.get('project'))
        print("SUCCESS: Google Earth Engine initialized from EARTHENGINE_TOKEN")
    else:
        ee.Initialize()
        print("SUCCESS: Google Earth Engine initialized from local credentials")
    GEE_AVAILABLE = True
except Exception as e:
    print(f"WARNING: GEE initialization failed: {e}")


def load_model():
    """Load trained U-Net model"""
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

    # Load the enhanced model (trained with proper water detection)
    checkpoint_path = Path('checkpoints/enhanced_model.pth')
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


def classify_image(bands_data):
    """Run model inference"""
    # Debug: print data statistics
    print(f"Input bands_data shape: {bands_data.shape}")
    print(f"Input data range: min={bands_data.min():.4f}, max={bands_data.max():.4f}, mean={bands_data.mean():.4f}")

    input_tensor = torch.from_numpy(bands_data).float().unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        output = MODEL(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze(0)
        prediction = prediction.cpu().numpy()

    # Debug: print prediction statistics
    unique, counts = np.unique(prediction, return_counts=True)
    print(f"Prediction classes: {dict(zip(unique, counts))}")

    return prediction


def create_colored_mask(prediction):
    """Convert class indices to RGB"""
    h, w = prediction.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, info in CLASSES.items():
        mask = prediction == class_id
        colored[mask] = info['color']

    return colored


def get_class_statistics(prediction):
    """Calculate class distribution"""
    unique, counts = np.unique(prediction, return_counts=True)
    total_pixels = prediction.size

    stats = []
    for class_id, count in zip(unique, counts):
        if class_id in CLASSES:
            percentage = (count / total_pixels) * 100
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
    <title>Live Interactive Map - Mauritius Land Cover</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; overflow: hidden; }

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
            height: calc(100vh - 120px);
            gap: 0;
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
        }

        .panel:last-child {
            border-right: none;
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

        .image-container {
            text-align: center;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .image-container img {
            max-width: 100%;
            max-height: 100%;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
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

        .legend-percentage {
            font-weight: 700;
            font-size: 16px;
            color: #333;
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
            <div class="panel-header">🛰️ Sentinel-2 Satellite Image</div>
            <div class="panel-content">
                <div class="image-container" id="satellite-container">
                    <p style="color: #999;">Pan the map to load imagery</p>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">🗺️ Land Cover Classification</div>
            <div class="panel-content">
                <div class="image-container" id="classification-container">
                    <p style="color: #999;">Classification will appear here</p>
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
            'Forest': 'rgba(139, 195, 74, 0.8)',
            'Plantation': 'rgba(197, 225, 165, 0.8)',
            'Urban': 'rgba(215, 204, 200, 0.8)',
            'Roads': 'rgba(158, 158, 158, 0.8)',
            'Bare Land': 'rgba(239, 235, 233, 0.8)',
            'Background': 'rgba(245, 243, 240, 0.8)'
        };

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

                // Update statistics
                let legendHtml = '';
                data.statistics.forEach(stat => {
                    legendHtml += `
                        <div class="legend-item">
                            <div class="legend-color" style="background: rgb(${stat.color.join(',')})"></div>
                            <div class="legend-text">${stat.name}</div>
                            <div class="legend-percentage">${stat.percentage}%</div>
                        </div>
                    `;
                });
                document.getElementById('legend').innerHTML = legendHtml;

                // Store statistics for historical comparison
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
            if (lastFetchedLat !== null && lastFetchedLon !== null) {
                fetchAndClassify(lastFetchedLat, lastFetchedLon, true);
            }
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

        # Classify
        prediction = classify_image(image_data['bands_data'])

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

                # Classify
                prediction = classify_image(image_data['bands_data'])

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
