"""
Vercel Serverless Function Entry Point for Mauritius Land Cover Classification
Simplified version for Vercel deployment - uses demo mode with synthetic data
"""

import numpy as np
from flask import Flask, render_template_string, jsonify, request
from PIL import Image
import io
import base64

app = Flask(__name__)

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


def array_to_base64(arr):
    """Convert numpy array to base64 PNG"""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_demo_response(lat, lon, year='current'):
    """Create a demo response with synthetic visualization"""
    # Create deterministic pattern based on location
    np.random.seed(int(abs(lat * 1000) + abs(lon * 1000)) % 2147483647)

    size = 256
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))

    # Create class distribution based on typical Mauritius patterns
    prediction = np.zeros((size, size), dtype=np.int32)

    # Adjust based on location relative to Mauritius center
    lat_offset = (lat - MAURITIUS_BOUNDS['center_lat']) / 0.3
    lon_offset = (lon - MAURITIUS_BOUNDS['center_lon']) / 0.3

    # Water around edges (coast simulation) - more water on west/east edges
    water_threshold = 0.12 + 0.05 * abs(lon_offset)
    water_mask = (x < water_threshold) | (x > 1-water_threshold) | (y < 0.08) | (y > 0.92)
    prediction[water_mask] = 1  # Water

    # Forest in central highlands
    forest_cx, forest_cy = 0.5 - lon_offset * 0.1, 0.5 - lat_offset * 0.1
    forest_mask = ((x - forest_cx)**2 + (y - forest_cy)**2 < 0.12) & ~water_mask
    prediction[forest_mask] = 2  # Forest

    # Plantation around forest
    plantation_mask = ((x - forest_cx)**2 + (y - forest_cy)**2 < 0.22) & ~forest_mask & ~water_mask
    prediction[plantation_mask] = 3  # Plantation

    # Urban clusters - vary by year to show urbanization
    year_factor = 1.0
    if year == '2022':
        year_factor = 1.3
    elif year == '2019':
        year_factor = 1.2
    elif year == '2016':
        year_factor = 1.1
    elif year == '2013':
        year_factor = 0.9
    elif year == '2010':
        year_factor = 0.8

    urban_centers = [
        (0.25 + lon_offset * 0.05, 0.3 + lat_offset * 0.05),
        (0.75 - lon_offset * 0.05, 0.7 - lat_offset * 0.05),
        (0.3, 0.7)
    ]
    for cx, cy in urban_centers:
        urban_radius = 0.025 * year_factor
        urban_mask = ((x - cx)**2 + (y - cy)**2 < urban_radius) & ~water_mask
        prediction[urban_mask] = 4  # Urban

    # Roads connecting areas
    road_mask = (np.abs(x - y) < 0.015) | (np.abs(x + y - 1) < 0.015)
    road_mask = road_mask & ~water_mask & (prediction != 4)
    prediction[road_mask] = 5  # Roads

    # Some bare land
    bare_mask = (np.random.random((size, size)) < 0.02) & (prediction == 0)
    prediction[bare_mask] = 6  # Bare Land

    # Create colored classification mask
    colored = np.zeros((size, size, 3), dtype=np.uint8)
    for class_idx, class_info in CLASS_COLORS.items():
        mask = prediction == class_idx
        colored[mask] = class_info['color']

    # Create synthetic satellite-like RGB image
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[prediction == 0] = [180, 170, 160]  # Background - tan
    rgb[prediction == 1] = [50, 100, 150]   # Water - blue
    rgb[prediction == 2] = [30, 80, 30]     # Forest - dark green
    rgb[prediction == 3] = [100, 150, 80]   # Plantation - light green
    rgb[prediction == 4] = [150, 140, 135]  # Urban - gray
    rgb[prediction == 5] = [100, 100, 100]  # Roads - dark gray
    rgb[prediction == 6] = [200, 190, 180]  # Bare land - light tan

    # Add noise for realism
    noise = np.random.randint(-15, 15, rgb.shape).astype(np.int16)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Calculate statistics
    total_pixels = prediction.size
    stats = []
    for class_idx, class_info in CLASS_COLORS.items():
        count = np.sum(prediction == class_idx)
        percentage = (count / total_pixels) * 100
        if percentage > 0.5:
            stats.append({
                'name': class_info['name'],
                'color': class_info['color'],
                'percentage': round(percentage, 1),
                'pixels': int(count)
            })
    stats.sort(key=lambda x: x['percentage'], reverse=True)

    return {
        'satellite': array_to_base64(rgb),
        'classification': array_to_base64(colored),
        'statistics': stats,
        'year': year,
        'demo_mode': True
    }


# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mauritius Land Cover Classification</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }

        #header {
            background: linear-gradient(135deg, #2d5016 0%, #4a7c24 100%);
            color: white;
            padding: 12px 20px;
        }
        #header h1 { font-size: 1.3rem; margin-bottom: 4px; }
        #header p { font-size: 0.8rem; opacity: 0.9; }

        .demo-banner {
            background: #fff3cd;
            color: #856404;
            padding: 8px 20px;
            text-align: center;
            font-size: 12px;
            border-bottom: 1px solid #ffc107;
        }

        #info-bar {
            background: #f8f9fa;
            padding: 8px 20px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
            font-size: 13px;
        }

        #year-selector {
            padding: 6px 12px;
            border: 2px solid #4a7c24;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
        }

        #status {
            padding: 4px 12px;
            border-radius: 12px;
            background: #e8f5e9;
            color: #2e7d32;
            font-size: 12px;
        }
        #status.loading { background: #fff3e0; color: #f57c00; }

        #container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            height: calc(100vh - 140px);
        }

        @media (min-width: 1200px) {
            #container {
                grid-template-columns: 1fr 1fr 1fr 280px;
                grid-template-rows: 1fr;
            }
        }

        .panel {
            border: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
            min-height: 200px;
        }

        .panel-header {
            background: #f0f7e6;
            padding: 10px 12px;
            font-weight: 600;
            color: #2d5016;
            font-size: 13px;
            border-bottom: 1px solid #e0e0e0;
        }

        .panel-content {
            flex: 1;
            overflow: auto;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .panel-content img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }

        #map { height: 100%; width: 100%; min-height: 250px; }

        #legend { padding: 10px; width: 100%; }

        .legend-item {
            display: flex;
            align-items: center;
            padding: 6px 8px;
            border-bottom: 1px solid #eee;
            font-size: 12px;
        }

        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 10px;
            border: 1px solid #ddd;
            flex-shrink: 0;
        }

        .legend-text { flex: 1; }
        .legend-percentage { font-weight: 600; color: #2d5016; }

        .placeholder {
            color: #999;
            text-align: center;
            padding: 20px;
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
            width: 40px;
            height: 40px;
            border: 3px solid #e0e0e0;
            border-top-color: #4a7c24;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div id="header">
        <h1>Mauritius Land Cover Classification</h1>
        <p>Interactive map with AI-powered land cover analysis</p>
    </div>

    <div class="demo-banner">
        Demo Mode: Using synthetic visualization. For real satellite imagery, run locally with Google Earth Engine.
        <a href="https://github.com/lloyd97/mauritius-landcover" target="_blank" style="color: #0056b3; margin-left: 10px;">View on GitHub</a>
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
                <option value="2013">2013</option>
                <option value="2010">2010</option>
            </select>
        </div>
        <div id="status">Ready</div>
    </div>

    <div id="loading-overlay">
        <div class="spinner"></div>
        <p style="margin-top: 15px; color: #666;">Generating visualization...</p>
    </div>

    <div id="container">
        <div class="panel">
            <div class="panel-header">Map</div>
            <div class="panel-content" style="padding: 0;">
                <div id="map"></div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">Satellite Image</div>
            <div class="panel-content" id="satellite-container">
                <p class="placeholder">Click on the map to load imagery</p>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">Classification</div>
            <div class="panel-content" id="classification-container">
                <p class="placeholder">Classification will appear here</p>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">Statistics</div>
            <div class="panel-content" style="align-items: flex-start;">
                <div id="legend">
                    <p class="placeholder">Statistics will appear here</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const map = L.map('map').setView([-20.25, 57.55], 10);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap'
        }).addTo(map);

        let marker = L.marker([-20.25, 57.55]).addTo(map);
        let isLoading = false;

        function fetchAndClassify(lat, lon) {
            if (isLoading) return;

            const year = document.getElementById('year-selector').value;

            isLoading = true;
            document.getElementById('coordinates').textContent = 'Lat: ' + lat.toFixed(4) + ', Lon: ' + lon.toFixed(4);
            document.getElementById('status').textContent = 'Loading...';
            document.getElementById('status').className = 'loading';
            document.getElementById('loading-overlay').style.display = 'flex';

            marker.setLatLng([lat, lon]);

            fetch('/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lat: lat, lon: lon, year: year })
            })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                document.getElementById('satellite-container').innerHTML =
                    '<img src="data:image/png;base64,' + data.satellite + '" alt="Satellite">';

                document.getElementById('classification-container').innerHTML =
                    '<img src="data:image/png;base64,' + data.classification + '" alt="Classification">';

                var legendHtml = '';
                data.statistics.forEach(function(stat) {
                    legendHtml += '<div class="legend-item">' +
                        '<div class="legend-color" style="background: rgb(' + stat.color.join(',') + ')"></div>' +
                        '<div class="legend-text">' + stat.name + '</div>' +
                        '<div class="legend-percentage">' + stat.percentage + '%</div>' +
                        '</div>';
                });
                document.getElementById('legend').innerHTML = legendHtml;

                document.getElementById('status').textContent = 'Ready';
                document.getElementById('status').className = '';
                document.getElementById('loading-overlay').style.display = 'none';
                isLoading = false;
            })
            .catch(function(err) {
                console.error(err);
                document.getElementById('status').textContent = 'Error';
                document.getElementById('loading-overlay').style.display = 'none';
                isLoading = false;
            });
        }

        map.on('click', function(e) {
            fetchAndClassify(e.latlng.lat, e.latlng.lng);
        });

        document.getElementById('year-selector').addEventListener('change', function() {
            var latlng = marker.getLatLng();
            fetchAndClassify(latlng.lat, latlng.lng);
        });

        // Initial load
        setTimeout(function() {
            fetchAndClassify(-20.25, 57.55);
        }, 500);
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
    """Classify a location - returns demo visualization"""
    try:
        data = request.get_json()
        lat = float(data.get('lat', -20.25))
        lon = float(data.get('lon', 57.55))
        year = data.get('year', 'current')

        return jsonify(create_demo_response(lat, lon, year))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'mode': 'demo'})


# Vercel handler
if __name__ == '__main__':
    app.run(debug=True)
