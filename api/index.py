"""
Vercel Serverless Function Entry Point for Mauritius Land Cover Classification
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
import segmentation_models_pytorch as smp
from PIL import Image
import io
import base64
import json

app = Flask(__name__)

# Configuration
DEVICE = torch.device('cpu')  # Vercel runs on CPU
MODEL = None

# Historical time periods configuration
TIME_PERIODS = {
    'current': {'start': None, 'end': None, 'satellite': 'sentinel', 'label': 'Current (Last 60 days)'},
    '2022': {'start': '2022-01-01', 'end': '2022-12-31', 'satellite': 'sentinel', 'label': '2022'},
    '2019': {'start': '2019-01-01', 'end': '2019-12-31', 'satellite': 'sentinel', 'label': '2019'},
    '2016': {'start': '2016-01-01', 'end': '2016-12-31', 'satellite': 'sentinel', 'label': '2016'},
    '2013': {'start': '2013-01-01', 'end': '2013-12-31', 'satellite': 'landsat', 'label': '2013'},
    '2010': {'start': '2010-01-01', 'end': '2010-12-31', 'satellite': 'landsat', 'label': '2010'},
}

# Mauritius bounds
MAURITIUS_BOUNDS = {
    'min_lat': -20.53,
    'max_lat': -19.98,
    'min_lon': 57.30,
    'max_lon': 57.82,
    'center_lat': -20.25,
    'center_lon': 57.55
}

# Land cover classes with Apple Maps-inspired colors
CLASS_COLORS = {
    0: {'name': 'Background', 'color': [245, 243, 240]},
    1: {'name': 'Water', 'color': [168, 216, 234]},
    2: {'name': 'Forest', 'color': [139, 195, 74]},
    3: {'name': 'Plantation', 'color': [197, 225, 165]},
    4: {'name': 'Urban', 'color': [215, 204, 200]},
    5: {'name': 'Roads', 'color': [158, 158, 158]},
    6: {'name': 'Bare Land', 'color': [239, 235, 233]}
}

# Try to initialize Google Earth Engine
GEE_AVAILABLE = False
try:
    import ee
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        GEE_AVAILABLE = True
        print("Google Earth Engine initialized successfully")
    except Exception as e:
        print(f"GEE initialization failed: {e}")
        # Try with default credentials
        try:
            ee.Initialize()
            GEE_AVAILABLE = True
            print("GEE initialized with default credentials")
        except:
            print("GEE not available - will use demo mode")
except ImportError:
    print("earthengine-api not installed")


def load_model():
    """Load the trained model"""
    global MODEL

    if MODEL is not None:
        return MODEL

    print("Loading model...")

    # Create model architecture
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=9,
        classes=7
    )

    # Try to load checkpoint
    checkpoint_paths = [
        Path('checkpoints/best_model.pth'),
        Path('../checkpoints/best_model.pth'),
        Path('/var/task/checkpoints/best_model.pth'),
    ]

    checkpoint_path = None
    for p in checkpoint_paths:
        if p.exists():
            checkpoint_path = p
            break

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    else:
        print("WARNING: No checkpoint found - model will produce random predictions")

    model.to(DEVICE)
    model.eval()
    MODEL = model
    return model


def create_colored_mask(prediction):
    """Convert class predictions to colored RGB image"""
    h, w = prediction.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, class_info in CLASS_COLORS.items():
        mask = prediction == class_idx
        colored[mask] = class_info['color']

    return colored


def get_class_statistics(prediction):
    """Calculate class distribution statistics"""
    total_pixels = prediction.size
    stats = []

    for class_idx, class_info in CLASS_COLORS.items():
        count = np.sum(prediction == class_idx)
        percentage = (count / total_pixels) * 100

        if percentage > 0.1:  # Only include classes with >0.1%
            stats.append({
                'name': class_info['name'],
                'color': class_info['color'],
                'percentage': round(percentage, 1),
                'pixels': int(count)
            })

    # Sort by percentage descending
    stats.sort(key=lambda x: x['percentage'], reverse=True)
    return stats


def array_to_base64(arr):
    """Convert numpy array to base64 PNG"""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_demo_response(lat, lon, year='current'):
    """Create a demo response when GEE is not available"""
    # Create synthetic demo images
    np.random.seed(int(abs(lat * 1000) + abs(lon * 1000)))

    # Create a simple pattern based on location
    size = 256
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))

    # Create class distribution based on typical Mauritius patterns
    prediction = np.zeros((size, size), dtype=np.int32)

    # Water around edges (coast simulation)
    water_mask = (x < 0.1) | (x > 0.9) | (y < 0.1) | (y > 0.9)
    prediction[water_mask] = 1  # Water

    # Forest in highlands
    forest_mask = ((x - 0.5)**2 + (y - 0.5)**2 < 0.15) & ~water_mask
    prediction[forest_mask] = 2  # Forest

    # Plantation around forest
    plantation_mask = ((x - 0.5)**2 + (y - 0.5)**2 < 0.25) & ~forest_mask & ~water_mask
    prediction[plantation_mask] = 3  # Plantation

    # Urban clusters
    urban_centers = [(0.3, 0.3), (0.7, 0.7), (0.3, 0.7)]
    for cx, cy in urban_centers:
        urban_mask = ((x - cx)**2 + (y - cy)**2 < 0.02) & ~water_mask
        prediction[urban_mask] = 4  # Urban

    # Roads connecting urban areas
    road_mask = (np.abs(x - y) < 0.02) | (np.abs(x + y - 1) < 0.02)
    road_mask = road_mask & ~water_mask & (prediction != 4)
    prediction[road_mask] = 5  # Roads

    # Create colored mask
    classification_colored = create_colored_mask(prediction)

    # Create synthetic RGB satellite-like image
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[prediction == 0] = [180, 170, 160]  # Background - tan
    rgb[prediction == 1] = [50, 100, 150]    # Water - blue
    rgb[prediction == 2] = [30, 80, 30]      # Forest - dark green
    rgb[prediction == 3] = [100, 150, 80]    # Plantation - light green
    rgb[prediction == 4] = [150, 140, 135]   # Urban - gray
    rgb[prediction == 5] = [100, 100, 100]   # Roads - dark gray
    rgb[prediction == 6] = [200, 190, 180]   # Bare land - light tan

    # Add some noise for realism
    noise = np.random.randint(-20, 20, rgb.shape).astype(np.int16)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    statistics = get_class_statistics(prediction)

    return {
        'satellite': array_to_base64(rgb),
        'classification': array_to_base64(classification_colored),
        'statistics': statistics,
        'year': year,
        'demo_mode': True,
        'message': 'Demo mode - GEE not available on Vercel free tier'
    }


# HTML Template for the interactive map
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Mauritius Land Cover Classification</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }

        #header {
            background: linear-gradient(135deg, #2d5016 0%, #4a7c24 100%);
            color: white;
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        #header h1 { font-size: 1.5rem; }
        #header p { font-size: 0.85rem; opacity: 0.9; }

        #info-bar {
            background: #f8f9fa;
            padding: 10px 30px;
            border-bottom: 2px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }

        #year-selector {
            padding: 8px 15px;
            font-size: 14px;
            border: 2px solid #4a7c24;
            border-radius: 8px;
            background: white;
            cursor: pointer;
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
            grid-template-columns: 1fr 1fr 1fr 300px;
            height: calc(100vh - 110px);
        }

        @media (max-width: 1200px) {
            #container {
                grid-template-columns: 1fr 1fr;
                grid-template-rows: 1fr 1fr;
            }
        }

        .panel {
            border-right: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
        }

        .panel-header {
            background: #f0f7e6;
            padding: 12px 15px;
            font-weight: 600;
            color: #2d5016;
            border-bottom: 1px solid #e0e0e0;
        }

        .panel-content {
            flex: 1;
            overflow: auto;
            padding: 10px;
        }

        .panel-content img {
            width: 100%;
            height: auto;
            display: block;
        }

        #map { height: 100%; width: 100%; }

        #legend { padding: 5px; }

        .legend-item {
            display: flex;
            align-items: center;
            padding: 8px 10px;
            border-bottom: 1px solid #eee;
        }

        .legend-color {
            width: 24px;
            height: 24px;
            border-radius: 4px;
            margin-right: 12px;
            border: 1px solid #ddd;
        }

        .legend-text { flex: 1; font-size: 13px; }
        .legend-percentage { font-weight: 600; color: #2d5016; }

        .demo-banner {
            background: #fff3cd;
            color: #856404;
            padding: 8px 15px;
            text-align: center;
            font-size: 13px;
        }

        #loading-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(255,255,255,0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #e0e0e0;
            border-top-color: #4a7c24;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div id="header">
        <div>
            <h1>🇲🇺 Mauritius Land Cover Classification</h1>
            <p>Interactive map with AI-powered land cover analysis</p>
        </div>
    </div>

    <div class="demo-banner" id="demo-banner" style="display: none;">
        ⚠️ Demo Mode: Using synthetic data. For full functionality, run locally with Google Earth Engine.
    </div>

    <div id="info-bar">
        <div id="coordinates">Click on map to classify</div>
        <div>
            <label for="year-selector">Year: </label>
            <select id="year-selector">
                <option value="current">Current</option>
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
        <div class="spinner"></div>
        <p style="margin-top: 20px;">Analyzing imagery...</p>
    </div>

    <div id="container">
        <div class="panel">
            <div class="panel-header">🗺️ Map</div>
            <div class="panel-content" style="padding: 0;">
                <div id="map"></div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">🛰️ Satellite Image</div>
            <div class="panel-content" id="satellite-container">
                <p style="color: #999; text-align: center; padding: 20px;">
                    Click on the map to load imagery
                </p>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">🎨 Classification</div>
            <div class="panel-content" id="classification-container">
                <p style="color: #999; text-align: center; padding: 20px;">
                    Classification will appear here
                </p>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">📊 Statistics</div>
            <div class="panel-content">
                <div id="legend">
                    <p style="color: #999; text-align: center;">
                        Statistics will appear here
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const map = L.map('map').setView([-20.25, 57.55], 11);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap'
        }).addTo(map);

        let marker = L.marker([-20.25, 57.55]).addTo(map);
        let isLoading = false;

        function fetchAndClassify(lat, lon) {
            if (isLoading) return;

            const year = document.getElementById('year-selector').value;

            isLoading = true;
            document.getElementById('coordinates').textContent = `Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`;
            document.getElementById('status').textContent = 'Loading...';
            document.getElementById('status').className = 'loading';
            document.getElementById('loading-overlay').style.display = 'flex';

            marker.setLatLng([lat, lon]);

            fetch('/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lat, lon, year })
            })
            .then(r => r.json())
            .then(data => {
                if (data.demo_mode) {
                    document.getElementById('demo-banner').style.display = 'block';
                }

                document.getElementById('satellite-container').innerHTML =
                    '<img src="data:image/png;base64,' + data.satellite + '">';

                document.getElementById('classification-container').innerHTML =
                    '<img src="data:image/png;base64,' + data.classification + '">';

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

                document.getElementById('status').textContent = 'Ready';
                document.getElementById('status').className = '';
                document.getElementById('loading-overlay').style.display = 'none';
                isLoading = false;
            })
            .catch(err => {
                console.error(err);
                document.getElementById('status').textContent = 'Error';
                document.getElementById('loading-overlay').style.display = 'none';
                isLoading = false;
            });
        }

        map.on('click', e => fetchAndClassify(e.latlng.lat, e.latlng.lng));

        document.getElementById('year-selector').addEventListener('change', () => {
            const center = marker.getLatLng();
            fetchAndClassify(center.lat, center.lng);
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Serve the main page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify a location"""
    try:
        data = request.get_json()
        lat = data.get('lat', -20.25)
        lon = data.get('lon', 57.55)
        year = data.get('year', 'current')

        # For Vercel deployment, use demo mode
        # GEE requires authentication that's complex on serverless
        return jsonify(create_demo_response(lat, lon, year))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'gee_available': GEE_AVAILABLE,
        'model_loaded': MODEL is not None
    })


# For Vercel
if __name__ == '__main__':
    app.run(debug=True)
