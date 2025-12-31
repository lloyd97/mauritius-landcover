"""
Live Interactive Map Viewer for Mauritius Land Cover Classification
Fetches and classifies Sentinel-2 imagery in real-time as you pan the map
"""

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

app = Flask(__name__)

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
    ee.Initialize()
    GEE_AVAILABLE = True
    print("SUCCESS: Google Earth Engine initialized")
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
        }

        #coordinates {
            font-size: 14px;
            font-weight: 500;
            color: #333;
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
            grid-template-columns: 400px 1fr 1fr 350px;
            height: calc(100vh - 120px);
            gap: 0;
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

        // Function to fetch and classify
        function fetchAndClassify(lat, lon) {
            // Don't fetch if already loading
            if (isLoading) return;

            // Don't fetch if coordinates haven't changed much (< 0.01 degrees)
            if (lastFetchedLat !== null && lastFetchedLon !== null) {
                const latDiff = Math.abs(lat - lastFetchedLat);
                const lonDiff = Math.abs(lon - lastFetchedLon);
                if (latDiff < 0.01 && lonDiff < 0.01) {
                    return;
                }
            }

            isLoading = true;
            lastFetchedLat = lat;
            lastFetchedLon = lon;

            // Update UI
            document.getElementById('coordinates').textContent =
                `Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`;
            document.getElementById('status').textContent = 'Fetching...';
            document.getElementById('status').className = 'loading';
            document.getElementById('loading-overlay').style.display = 'flex';

            // Update marker
            marker.setLatLng([lat, lon]);

            // Fetch from backend
            fetch('/api/classify_location', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lat, lon })
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
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

        // Fetch on map move end (after pan/zoom)
        map.on('moveend', function() {
            const center = map.getCenter();
            fetchAndClassify(center.lat, center.lng);
        });

        // Fetch on click
        map.on('click', function(e) {
            fetchAndClassify(e.latlng.lat, e.latlng.lng);
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
    """Download and classify Sentinel-2 imagery for location"""
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')

        print(f"\n{'='*60}")
        print(f"Classifying location: {lat:.4f}, {lon:.4f}")

        # Download Sentinel-2 imagery
        image_data = download_sentinel2_for_location(lat, lon)

        # Classify
        prediction = classify_image(image_data['bands_data'])

        # Create colored visualization
        classification_colored = create_colored_mask(prediction)

        # Get statistics
        statistics = get_class_statistics(prediction)

        # Convert to base64
        satellite_b64 = array_to_base64(image_data['rgb_image'])
        classification_b64 = array_to_base64(classification_colored)

        print(f"Classification complete!")
        print(f"{'='*60}\n")

        return jsonify({
            'satellite': satellite_b64,
            'classification': classification_b64,
            'statistics': statistics
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Live Interactive Map - Mauritius Land Cover")
    print("=" * 60)

    # Load model
    load_model()

    print(f"\nUsing device: {DEVICE}")
    print(f"Google Earth Engine: {'Available' if GEE_AVAILABLE else 'Not available (using fallback)'}")
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
