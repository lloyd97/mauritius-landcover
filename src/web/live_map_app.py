"""
Live Sentinel-2 Map Viewer for Mauritius Land Cover Classification
===================================================================

Interactive web app that shows:
- LEFT: Real Sentinel-2 satellite imagery
- RIGHT: Classified land cover with color coding

Features:
- Download live Sentinel-2 data from Google Earth Engine
- Real-time land cover classification
- Side-by-side comparison view
- Interactive map controls
"""

import os
import sys
import io
import base64
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import cv2

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import ee
    import geemap
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    print("WARNING: Google Earth Engine not available. Using demo mode.")

import segmentation_models_pytorch as smp
import rasterio
from rasterio.transform import from_bounds

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Configuration
MAURITIUS_BOUNDS = {
    'min_lon': 57.30,
    'max_lon': 57.82,
    'min_lat': -20.53,
    'max_lat': -19.98
}

LAND_COVER_CLASSES = {
    0: {'name': 'Background', 'color': [0, 0, 0]},
    1: {'name': 'Roads', 'color': [128, 128, 128]},
    2: {'name': 'Water', 'color': [0, 100, 255]},
    3: {'name': 'Forest', 'color': [0, 100, 0]},
    4: {'name': 'Plantation', 'color': [50, 205, 50]},
    5: {'name': 'Buildings', 'color': [139, 69, 19]},
    6: {'name': 'Bare Land', 'color': [210, 180, 140]},
}

# Global variables
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = Path('data/live_imagery')
DATA_DIR.mkdir(parents=True, exist_ok=True)


# Initialize Google Earth Engine
if GEE_AVAILABLE:
    try:
        ee.Initialize()
        print("SUCCESS: Google Earth Engine initialized")
    except Exception as e:
        print(f"WARNING: Google Earth Engine initialization failed: {e}")
        print("Run: earthengine authenticate")
        GEE_AVAILABLE = False


def load_model():
    """Load the trained U-Net model"""
    global MODEL

    # Create model with same architecture as training script
    MODEL = smp.Unet(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,  # Start with 3 for ImageNet weights
        classes=7,
        activation=None
    )

    # Modify first conv layer for 9 channels (same as training script)
    import torch.nn as nn
    first_conv = MODEL.encoder.conv1
    new_conv = nn.Conv2d(
        9, first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None
    )

    # Initialize: copy RGB weights, initialize extra channels
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = first_conv.weight[:, :3, :, :]
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
        new_conv.weight[:, 3:, :, :] *= 0.01

    MODEL.encoder.conv1 = new_conv

    # Try to load checkpoint
    checkpoint_paths = [
        'checkpoints/best_model.pth',
        'checkpoints/best.pt',
        'checkpoints/latest.pt',
        '../checkpoints/best_model.pth',
        '../checkpoints/best.pt'
    ]

    for checkpoint_path in checkpoint_paths:
        if Path(checkpoint_path).exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                MODEL.load_state_dict(checkpoint['model_state_dict'])
                print(f"SUCCESS: Loaded model from {checkpoint_path}")
                break
            except Exception as e:
                print(f"WARNING: Failed to load {checkpoint_path}: {e}")
    else:
        print("WARNING: No trained model found - using untrained model for demo")

    MODEL.to(DEVICE)
    MODEL.eval()


def download_sentinel2_image(center_lon, center_lat, zoom_level=12, use_recent=True):
    """
    Download Sentinel-2 image for a specific location in Mauritius

    Args:
        center_lon: Longitude of center point
        center_lat: Latitude of center point
        zoom_level: Map zoom level (higher = smaller area, more detail)
        use_recent: If True, use most recent imagery; else use baseline

    Returns:
        dict with 'rgb_image' and 'bands_image' (numpy arrays)
    """
    if not GEE_AVAILABLE:
        # Return demo/synthetic data
        return create_demo_satellite_image(512, 512, center_lat, center_lon)

    try:
        # Calculate bounds based on zoom level
        # Zoom 12 ≈ 5km x 5km, Zoom 14 ≈ 1.25km x 1.25km
        degrees_per_pixel = 156543.03392 * np.cos(np.radians(center_lat)) / (2 ** zoom_level) / 111320
        size_degrees = degrees_per_pixel * 512  # 512 pixels

        bounds = ee.Geometry.Rectangle([
            center_lon - size_degrees/2,
            center_lat - size_degrees/2,
            center_lon + size_degrees/2,
            center_lat + size_degrees/2
        ])

        # Date range
        if use_recent:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
        else:
            start_date = datetime(2015, 6, 1)
            end_date = datetime(2016, 5, 31)

        # Get Sentinel-2 collection
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(bounds)
              .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

        # Get median composite
        composite = s2.median().clip(bounds)

        # Select bands
        bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        composite_bands = composite.select(bands)

        # Compute indices
        ndvi = composite_bands.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndwi = composite_bands.normalizedDifference(['B3', 'B8']).rename('NDWI')
        ndbi = composite_bands.normalizedDifference(['B11', 'B8']).rename('NDBI')

        # Combine all
        full_composite = composite_bands.addBands([ndvi, ndwi, ndbi])

        # Download RGB for display
        rgb_params = {
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000,
            'dimensions': 512
        }

        rgb_url = full_composite.getThumbURL(rgb_params)

        # Download full bands for classification
        full_params = {
            'bands': bands + ['NDVI', 'NDWI', 'NDBI'],
            'dimensions': 512,
            'format': 'GEO_TIFF'
        }

        # Use geemap to download
        output_path = DATA_DIR / f'sentinel2_{center_lat}_{center_lon}.tif'
        geemap.ee_export_image(
            full_composite,
            filename=str(output_path),
            scale=10,
            region=bounds,
            file_per_band=False
        )

        # Load the data
        with rasterio.open(output_path) as src:
            bands_data = src.read()  # Shape: (9, H, W)

        # Download RGB image separately
        import requests
        rgb_response = requests.get(rgb_url)
        rgb_image = np.array(Image.open(io.BytesIO(rgb_response.content)))

        return {
            'rgb_image': rgb_image,
            'bands_data': bands_data,
            'bounds': [center_lon - size_degrees/2, center_lat - size_degrees/2,
                      center_lon + size_degrees/2, center_lat + size_degrees/2]
        }

    except Exception as e:
        print(f"Error downloading Sentinel-2 image: {e}")
        import traceback
        traceback.print_exc()
        return create_demo_satellite_image(512, 512, center_lat, center_lon)


def create_demo_satellite_image(height=512, width=512, lat=-20.25, lon=57.55):
    """Create synthetic satellite-like image for demo"""
    # Use coordinates to vary the output
    seed = int(abs(lat * 1000 + lon * 1000)) % 10000
    np.random.seed(seed)

    # Create synthetic RGB image
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))

    # Use lat/lon to vary the patterns
    offset_x = (lon - 57.55) * 10
    offset_y = (lat + 20.25) * 10

    # Simulate terrain features with variation based on location
    vegetation = (np.sin((x + offset_x) * 10) * np.cos((y + offset_y) * 8) * 0.3 + 0.5) * 200
    water = np.exp(-(((x + offset_x * 0.1) - 0.3)**2 + ((y + offset_y * 0.1) - 0.7)**2) * 20) * 150
    urban = ((np.sin((x + offset_x * 0.05) * 30) > 0.8).astype(float) *
             (np.sin((y + offset_y * 0.05) * 30) > 0.8).astype(float)) * 180

    # RGB channels
    r = (vegetation * 0.4 + urban * 0.6 + 30).astype(np.uint8)
    g = (vegetation * 0.7 + water * 0.5 + 40).astype(np.uint8)
    b = (water * 0.8 + 20).astype(np.uint8)

    rgb_image = np.stack([r, g, b], axis=-1)

    # Create 9-band data (6 bands + 3 indices)
    bands_data = np.random.rand(9, height, width).astype(np.float32)

    # Simulate realistic bands
    bands_data[0] = b / 255.0  # B2 - Blue
    bands_data[1] = g / 255.0  # B3 - Green
    bands_data[2] = r / 255.0  # B4 - Red
    bands_data[3] = vegetation / 255.0  # B8 - NIR
    bands_data[4] = urban / 255.0  # B11 - SWIR1
    bands_data[5] = urban / 255.0  # B12 - SWIR2

    # Indices
    bands_data[6] = (bands_data[3] - bands_data[2]) / (bands_data[3] + bands_data[2] + 1e-6)  # NDVI
    bands_data[7] = (bands_data[1] - bands_data[3]) / (bands_data[1] + bands_data[3] + 1e-6)  # NDWI
    bands_data[8] = (bands_data[4] - bands_data[3]) / (bands_data[4] + bands_data[3] + 1e-6)  # NDBI

    return {
        'rgb_image': rgb_image,
        'bands_data': bands_data,
        'bounds': [MAURITIUS_BOUNDS['min_lon'], MAURITIUS_BOUNDS['min_lat'],
                  MAURITIUS_BOUNDS['max_lon'], MAURITIUS_BOUNDS['max_lat']]
    }


def classify_image(bands_data):
    """
    Classify satellite image using trained U-Net model

    Args:
        bands_data: numpy array of shape (9, H, W)

    Returns:
        classification: numpy array of shape (H, W) with class indices
    """
    # Prepare input tensor
    input_tensor = torch.from_numpy(bands_data).float().unsqueeze(0)  # (1, 9, H, W)
    input_tensor = input_tensor.to(DEVICE)

    # Run inference
    with torch.no_grad():
        output = MODEL(input_tensor)  # (1, 7, H, W)
        prediction = torch.argmax(output, dim=1).squeeze(0)  # (H, W)
        prediction = prediction.cpu().numpy()

    return prediction


def create_colored_classification(classification):
    """
    Convert class indices to RGB colored image

    Args:
        classification: numpy array of shape (H, W) with class indices

    Returns:
        colored_image: numpy array of shape (H, W, 3) with RGB colors
    """
    h, w = classification.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, info in LAND_COVER_CLASSES.items():
        mask = classification == class_id
        colored[mask] = info['color']

    return colored


def array_to_base64(img_array):
    """Convert numpy array to base64 string for web display"""
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


# ==================== ROUTES ====================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Mauritius Live Satellite Viewer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; }

        #header {
            background: #2d5016;
            color: white;
            padding: 15px 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }

        #header h1 { font-size: 24px; margin-bottom: 5px; }
        #header p { font-size: 14px; opacity: 0.9; }

        #controls {
            background: #f5f5f5;
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
        }

        #controls button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
        }

        #controls button:hover { background: #45a049; }
        #controls button:disabled { background: #ccc; cursor: not-allowed; }

        #loading {
            display: none;
            color: #666;
            margin-left: 10px;
        }

        #container {
            display: flex;
            height: calc(100vh - 140px);
        }

        .panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #ddd;
        }

        .panel:last-child { border-right: none; }

        .panel-header {
            background: #fff;
            padding: 10px 15px;
            border-bottom: 2px solid #4CAF50;
            font-weight: bold;
            font-size: 16px;
        }

        .panel-content {
            flex: 1;
            overflow: hidden;
            position: relative;
            background: #000;
        }

        #map {
            width: 100%;
            height: 100%;
        }

        .image-view {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .image-view img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }

        #legend {
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            z-index: 1000;
        }

        #legend h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
            font-weight: bold;
        }

        .legend-item {
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 12px;
        }

        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 3px;
            margin-right: 8px;
            border: 1px solid #999;
        }

        .placeholder {
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div id="header">
        <h1>🇲🇺 Mauritius Land Cover - Live Satellite Viewer</h1>
        <p>Click on the map to view Sentinel-2 imagery and land cover classification</p>
    </div>

    <div id="controls">
        <button onclick="useCurrentView()">Classify Current View</button>
        <button onclick="toggleTimeframe()" id="timeBtn">Switch to 2015 Baseline</button>
        <span id="loading">⏳ Loading...</span>
    </div>

    <div id="container">
        <!-- Map Panel -->
        <div class="panel">
            <div class="panel-header">📍 Interactive Map - Click to Select Area</div>
            <div class="panel-content">
                <div id="map"></div>
            </div>
        </div>

        <!-- Satellite Image Panel -->
        <div class="panel">
            <div class="panel-header">🛰️ Sentinel-2 Satellite Image</div>
            <div class="panel-content">
                <div class="image-view" id="satellite-view">
                    <div class="placeholder">Click on the map to load satellite imagery</div>
                </div>
            </div>
        </div>

        <!-- Classification Panel -->
        <div class="panel">
            <div class="panel-header">🎨 Land Cover Classification</div>
            <div class="panel-content">
                <div class="image-view" id="classification-view">
                    <div class="placeholder">Classification will appear here</div>
                </div>
                <div id="legend">
                    <h3>Land Cover Classes</h3>
                    <div class="legend-item"><div class="legend-color" style="background: rgb(128,128,128)"></div>Roads</div>
                    <div class="legend-item"><div class="legend-color" style="background: rgb(0,100,255)"></div>Water</div>
                    <div class="legend-item"><div class="legend-color" style="background: rgb(0,100,0)"></div>Forest</div>
                    <div class="legend-item"><div class="legend-color" style="background: rgb(50,205,50)"></div>Plantation</div>
                    <div class="legend-item"><div class="legend-color" style="background: rgb(139,69,19)"></div>Buildings</div>
                    <div class="legend-item"><div class="legend-color" style="background: rgb(210,180,140)"></div>Bare Land</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize map centered on Mauritius
        const map = L.map('map').setView([-20.25, 57.55], 10);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        let currentMarker = null;
        let useRecent = true;

        // Handle map clicks
        map.on('click', function(e) {
            const lat = e.latlng.lat;
            const lon = e.latlng.lng;

            // Add/update marker
            if (currentMarker) {
                map.removeLayer(currentMarker);
            }
            currentMarker = L.marker([lat, lon]).addTo(map);

            // Classify this location
            classifyLocation(lat, lon);
        });

        function useCurrentView() {
            const center = map.getCenter();
            classifyLocation(center.lat, center.lng);
        }

        function toggleTimeframe() {
            useRecent = !useRecent;
            document.getElementById('timeBtn').textContent =
                useRecent ? 'Switch to 2015 Baseline' : 'Switch to Current (2024)';

            if (currentMarker) {
                const pos = currentMarker.getLatLng();
                classifyLocation(pos.lat, pos.lng);
            }
        }

        async function classifyLocation(lat, lon) {
            const loadingEl = document.getElementById('loading');
            loadingEl.style.display = 'inline';

            try {
                const response = await fetch('/api/classify_location', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        lat: lat,
                        lon: lon,
                        zoom: map.getZoom(),
                        use_recent: useRecent
                    })
                });

                const data = await response.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Display satellite image
                document.getElementById('satellite-view').innerHTML =
                    `<img src="data:image/png;base64,${data.satellite_image}" alt="Satellite">`;

                // Display classification
                document.getElementById('classification-view').innerHTML =
                    `<img src="data:image/png;base64,${data.classification_image}" alt="Classification">`;

            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                loadingEl.style.display = 'none';
            }
        }

        // Load center of Mauritius on startup
        window.addEventListener('load', function() {
            setTimeout(() => useCurrentView(), 500);
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Main page"""
    return HTML_TEMPLATE


@app.route('/api/classify_location', methods=['POST'])
def classify_location():
    """
    Download satellite image for location and classify it
    """
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        zoom = data.get('zoom', 12)
        use_recent = data.get('use_recent', True)

        print(f"Classifying location: {lat}, {lon} (zoom: {zoom}, recent: {use_recent})")

        # Download satellite image
        image_data = download_sentinel2_image(lon, lat, zoom, use_recent)

        # Classify using model
        classification = classify_image(image_data['bands_data'])

        # Create colored visualization
        classification_colored = create_colored_classification(classification)

        # Convert to base64
        satellite_b64 = array_to_base64(image_data['rgb_image'])
        classification_b64 = array_to_base64(classification_colored)

        return jsonify({
            'satellite_image': satellite_b64,
            'classification_image': classification_b64,
            'bounds': image_data['bounds']
        })

    except Exception as e:
        print(f"Error in classify_location: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Mauritius Live Satellite Viewer")
    print("=" * 60)

    # Load model
    print("\nLoading classification model...")
    load_model()

    # Check GEE status
    if GEE_AVAILABLE:
        print("SUCCESS: Google Earth Engine Available")
        print("  -> Will use live Sentinel-2 imagery")
    else:
        print("WARNING: Google Earth Engine Not Available")
        print("  -> Using demo/synthetic imagery")
        print("  -> To enable: pip install earthengine-api geemap")
        print("  -> Then run: earthengine authenticate")

    print(f"\nDevice: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")

    print("\n" + "=" * 60)
    print("Starting web server...")
    print("Visit: http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
