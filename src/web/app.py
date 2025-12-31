"""
Web Interface for Mauritius Land Cover Analysis
================================================

Interactive web application for:
    - Viewing land cover classification results
    - Comparing different time periods
    - Analyzing change statistics
    - Exporting results

Usage:
    python app.py
    
    Then visit: http://localhost:5000
"""

import os
import io
import base64
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
import torch

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.unet import create_model
from utils.visualization import create_colormap, overlay_mask_on_image


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class definitions
CLASSES = {
    0: {'name': 'Background', 'color': '#000000'},
    1: {'name': 'Roads', 'color': '#808080'},
    2: {'name': 'Water', 'color': '#0064FF'},
    3: {'name': 'Forest', 'color': '#006400'},
    4: {'name': 'Plantation', 'color': '#32CD32'},
    5: {'name': 'Buildings', 'color': '#8B4513'},
    6: {'name': 'Bare Land', 'color': '#D2B48C'},
}


def load_model(checkpoint_path: str = None):
    """Load trained model."""
    global MODEL
    
    config = {
        'model': {
            'architecture': 'unet',
            'encoder': 'resnet34',
            'encoder_weights': None,
            'in_channels': 9,
            'num_classes': 7
        }
    }
    
    MODEL = create_model(config)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    
    MODEL.to(DEVICE)
    MODEL.eval()


def create_sample_prediction():
    """Create a sample prediction for demo purposes."""
    # Generate synthetic land cover map
    np.random.seed(42)
    h, w = 512, 512
    
    # Create realistic-looking land cover pattern
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    
    prediction = np.zeros((h, w), dtype=np.uint8)
    
    # Water (class 2) - coastal and rivers
    water_mask = (x < 0.1) | (np.abs(y - 0.5) < 0.02) | (
        ((x - 0.3)**2 + (y - 0.7)**2) < 0.01
    )
    prediction[water_mask] = 2
    
    # Forest (class 3) - mountain/natural areas
    forest_mask = (y > 0.6) & (x > 0.3) & ~water_mask
    prediction[forest_mask] = 3
    
    # Plantation (class 4) - agricultural areas
    plantation_mask = (
        (np.sin(x * 20) > 0.3) & 
        (np.sin(y * 20) > 0.3) & 
        (y < 0.6) & 
        ~water_mask
    )
    prediction[plantation_mask] = 4
    
    # Buildings (class 5) - urban areas
    building_mask = (
        (x > 0.4) & (x < 0.7) & 
        (y > 0.2) & (y < 0.5) & 
        ~water_mask
    )
    prediction[building_mask] = 5
    
    # Roads (class 1)
    road_mask = (
        (np.abs(x - 0.5) < 0.01) | 
        (np.abs(y - 0.35) < 0.01)
    ) & ~water_mask
    prediction[road_mask] = 1
    
    # Bare land (class 6) - remaining
    bare_mask = (
        (prediction == 0) & 
        ~water_mask & 
        (np.random.rand(h, w) > 0.7)
    )
    prediction[bare_mask] = 6
    
    return prediction


def prediction_to_colored_image(prediction: np.ndarray) -> np.ndarray:
    """Convert class prediction to RGB image."""
    h, w = prediction.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    color_map = {
        0: (0, 0, 0),
        1: (128, 128, 128),
        2: (0, 100, 255),
        3: (0, 100, 0),
        4: (50, 205, 50),
        5: (139, 69, 19),
        6: (210, 180, 140),
    }
    
    for class_id, color in color_map.items():
        mask = prediction == class_id
        colored[mask] = color
    
    return colored


def compute_statistics(prediction: np.ndarray) -> Dict:
    """Compute area statistics from prediction."""
    total_pixels = prediction.size
    stats = {}
    
    for class_id, info in CLASSES.items():
        count = np.sum(prediction == class_id)
        percentage = (count / total_pixels) * 100
        # Assuming 10m resolution -> 100 sq m per pixel
        area_hectares = count * 100 / 10000
        
        stats[info['name']] = {
            'pixels': int(count),
            'percentage': round(percentage, 2),
            'area_hectares': round(area_hectares, 2)
        }
    
    return stats


def compute_change_statistics(pred1: np.ndarray, pred2: np.ndarray) -> Dict:
    """Compute change statistics between two predictions."""
    changes = {}
    
    for class_id, info in CLASSES.items():
        if class_id == 0:
            continue
            
        count1 = np.sum(pred1 == class_id)
        count2 = np.sum(pred2 == class_id)
        
        change = count2 - count1
        change_hectares = change * 100 / 10000
        
        changes[info['name']] = {
            'before_hectares': round(count1 * 100 / 10000, 2),
            'after_hectares': round(count2 * 100 / 10000, 2),
            'change_hectares': round(change_hectares, 2),
            'change_percent': round((change / max(count1, 1)) * 100, 2)
        }
    
    return changes


def array_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


# ============== Routes ==============

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html', classes=CLASSES)


@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    try:
        # Load image
        img = Image.open(file)
        img_array = np.array(img)
        
        # For demo, use sample prediction
        # In production, run through model
        prediction = create_sample_prediction()
        
        # Convert to colored image
        colored = prediction_to_colored_image(prediction)
        
        # Compute statistics
        stats = compute_statistics(prediction)
        
        # Convert to base64
        pred_base64 = array_to_base64(colored)
        
        return jsonify({
            'prediction': pred_base64,
            'statistics': stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo')
def demo():
    """Generate demo prediction."""
    # Create sample prediction
    prediction = create_sample_prediction()
    
    # Convert to colored image
    colored = prediction_to_colored_image(prediction)
    
    # Compute statistics
    stats = compute_statistics(prediction)
    
    # Convert to base64
    pred_base64 = array_to_base64(colored)
    
    return jsonify({
        'prediction': pred_base64,
        'statistics': stats,
        'classes': CLASSES
    })


@app.route('/api/change_detection', methods=['POST'])
def change_detection():
    """Perform change detection between two images."""
    try:
        data = request.get_json()
        year1 = data.get('year1', 2015)
        year2 = data.get('year2', 2024)
        
        # Generate sample predictions for both years
        np.random.seed(year1)
        pred1 = create_sample_prediction()
        
        np.random.seed(year2)
        pred2 = create_sample_prediction()
        
        # Simulate some changes
        # More buildings in 2024
        if year2 > year1:
            expansion = np.random.rand(*pred2.shape) > 0.9
            pred2[expansion & (pred2 == 4)] = 5  # Plantation -> Building
        
        # Compute change map
        change_map = np.zeros_like(pred1)
        changed_pixels = pred1 != pred2
        change_map[changed_pixels] = 1
        
        # Convert to images
        pred1_colored = prediction_to_colored_image(pred1)
        pred2_colored = prediction_to_colored_image(pred2)
        
        # Change map colored (red for change)
        change_colored = np.zeros((*change_map.shape, 3), dtype=np.uint8)
        change_colored[changed_pixels] = [255, 0, 0]
        
        # Compute statistics
        changes = compute_change_statistics(pred1, pred2)
        
        return jsonify({
            'year1': {
                'year': year1,
                'image': array_to_base64(pred1_colored),
                'statistics': compute_statistics(pred1)
            },
            'year2': {
                'year': year2,
                'image': array_to_base64(pred2_colored),
                'statistics': compute_statistics(pred2)
            },
            'change_map': array_to_base64(change_colored),
            'change_statistics': changes
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/time_series')
def time_series():
    """Get time series data for area changes."""
    years = [2015, 2017, 2019, 2021, 2023, 2025]
    
    # Generate synthetic time series data
    data = {class_info['name']: [] for class_info in CLASSES.values()}
    
    base_values = {
        'Background': 5,
        'Roads': 8,
        'Water': 15,
        'Forest': 25,
        'Plantation': 30,
        'Buildings': 12,
        'Bare Land': 5
    }
    
    trends = {
        'Background': -0.1,
        'Roads': 0.3,
        'Water': 0,
        'Forest': -0.5,
        'Plantation': -0.3,
        'Buildings': 0.8,
        'Bare Land': 0.1
    }
    
    for i, year in enumerate(years):
        for class_name, base in base_values.items():
            value = base + trends[class_name] * i + np.random.randn() * 0.5
            data[class_name].append({
                'year': year,
                'area_hectares': max(0, round(value * 1000, 2))
            })
    
    return jsonify({
        'years': years,
        'data': data
    })


@app.route('/api/export/<format_type>')
def export_results(format_type: str):
    """Export results in various formats."""
    # Generate prediction
    prediction = create_sample_prediction()
    stats = compute_statistics(prediction)
    
    if format_type == 'geotiff':
        # Create GeoTIFF (simplified)
        buffer = io.BytesIO()
        img = Image.fromarray(prediction.astype(np.uint8))
        img.save(buffer, format='TIFF')
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype='image/tiff',
            as_attachment=True,
            download_name='mauritius_landcover.tif'
        )
    
    elif format_type == 'json':
        return jsonify(stats)
    
    elif format_type == 'csv':
        csv_lines = ['Class,Pixels,Percentage,Area (ha)']
        for class_name, values in stats.items():
            csv_lines.append(
                f"{class_name},{values['pixels']},{values['percentage']},{values['area_hectares']}"
            )
        return Response(
            '\n'.join(csv_lines),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment;filename=statistics.csv'}
        )
    
    else:
        return jsonify({'error': 'Unknown format'}), 400


# ============== HTML Templates ==============

@app.route('/templates/index.html')
def get_template():
    """Serve the main template."""
    return render_template('index.html')


# Create templates directory and HTML file
TEMPLATES_DIR = Path(__file__).parent / 'templates'
TEMPLATES_DIR.mkdir(exist_ok=True)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mauritius Land Cover Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .legend-color {
            width: 20px;
            height: 20px;
            display: inline-block;
            margin-right: 8px;
            border-radius: 4px;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <nav class="bg-green-700 text-white p-4">
        <div class="container mx-auto">
            <h1 class="text-2xl font-bold">🇲🇺 Mauritius Land Cover Analysis</h1>
            <p class="text-green-200">PhD Research - Sentinel-2 Satellite Imagery Analysis</p>
        </div>
    </nav>

    <main class="container mx-auto p-6">
        <!-- Tabs -->
        <div class="flex space-x-4 mb-6">
            <button onclick="showTab('classification')" class="tab-btn bg-green-600 text-white px-4 py-2 rounded">
                Classification
            </button>
            <button onclick="showTab('change')" class="tab-btn bg-gray-300 px-4 py-2 rounded">
                Change Detection
            </button>
            <button onclick="showTab('timeseries')" class="tab-btn bg-gray-300 px-4 py-2 rounded">
                Time Series
            </button>
        </div>

        <!-- Classification Tab -->
        <div id="classification-tab" class="tab-content">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Input Panel -->
                <div class="bg-white rounded-lg shadow p-6">
                    <h2 class="text-xl font-semibold mb-4">Input</h2>
                    <button onclick="loadDemo()" class="bg-green-600 text-white px-4 py-2 rounded mb-4">
                        Load Demo Data
                    </button>
                    <div id="input-preview" class="border rounded p-4 min-h-64 flex items-center justify-center">
                        <p class="text-gray-400">Click "Load Demo Data" to start</p>
                    </div>
                </div>

                <!-- Output Panel -->
                <div class="bg-white rounded-lg shadow p-6">
                    <h2 class="text-xl font-semibold mb-4">Classification Result</h2>
                    <div id="output-preview" class="border rounded p-4 min-h-64 flex items-center justify-center">
                        <p class="text-gray-400">Results will appear here</p>
                    </div>
                </div>
            </div>

            <!-- Legend -->
            <div class="bg-white rounded-lg shadow p-6 mt-6">
                <h2 class="text-xl font-semibold mb-4">Legend</h2>
                <div class="flex flex-wrap gap-4">
                    <div><span class="legend-color" style="background: #808080"></span>Roads</div>
                    <div><span class="legend-color" style="background: #0064FF"></span>Water</div>
                    <div><span class="legend-color" style="background: #006400"></span>Forest</div>
                    <div><span class="legend-color" style="background: #32CD32"></span>Plantation</div>
                    <div><span class="legend-color" style="background: #8B4513"></span>Buildings</div>
                    <div><span class="legend-color" style="background: #D2B48C"></span>Bare Land</div>
                </div>
            </div>

            <!-- Statistics -->
            <div class="bg-white rounded-lg shadow p-6 mt-6">
                <h2 class="text-xl font-semibold mb-4">Statistics</h2>
                <div id="statistics-table"></div>
            </div>
        </div>

        <!-- Change Detection Tab -->
        <div id="change-tab" class="tab-content hidden">
            <div class="bg-white rounded-lg shadow p-6">
                <h2 class="text-xl font-semibold mb-4">Change Detection</h2>
                <div class="flex space-x-4 mb-4">
                    <select id="year1" class="border rounded px-3 py-2">
                        <option value="2015">2015</option>
                        <option value="2017">2017</option>
                        <option value="2019">2019</option>
                    </select>
                    <span class="self-center">to</span>
                    <select id="year2" class="border rounded px-3 py-2">
                        <option value="2021">2021</option>
                        <option value="2023">2023</option>
                        <option value="2024" selected>2024</option>
                    </select>
                    <button onclick="runChangeDetection()" class="bg-green-600 text-white px-4 py-2 rounded">
                        Analyze Changes
                    </button>
                </div>
                <div class="grid grid-cols-3 gap-4" id="change-results">
                    <div class="text-center">
                        <p class="font-semibold mb-2">Before</p>
                        <div id="before-image" class="border rounded p-2 min-h-48"></div>
                    </div>
                    <div class="text-center">
                        <p class="font-semibold mb-2">After</p>
                        <div id="after-image" class="border rounded p-2 min-h-48"></div>
                    </div>
                    <div class="text-center">
                        <p class="font-semibold mb-2">Changes (Red)</p>
                        <div id="change-image" class="border rounded p-2 min-h-48"></div>
                    </div>
                </div>
                <div id="change-stats" class="mt-4"></div>
            </div>
        </div>

        <!-- Time Series Tab -->
        <div id="timeseries-tab" class="tab-content hidden">
            <div class="bg-white rounded-lg shadow p-6">
                <h2 class="text-xl font-semibold mb-4">Land Cover Change Over Time</h2>
                <canvas id="timeSeriesChart" height="100"></canvas>
            </div>
        </div>
    </main>

    <script>
        // Tab switching
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tab-btn').forEach(el => {
                el.classList.remove('bg-green-600', 'text-white');
                el.classList.add('bg-gray-300');
            });
            
            document.getElementById(tabName + '-tab').classList.remove('hidden');
            event.target.classList.add('bg-green-600', 'text-white');
            event.target.classList.remove('bg-gray-300');
            
            if (tabName === 'timeseries') {
                loadTimeSeries();
            }
        }

        // Load demo data
        async function loadDemo() {
            const response = await fetch('/api/demo');
            const data = await response.json();
            
            document.getElementById('output-preview').innerHTML = 
                `<img src="data:image/png;base64,${data.prediction}" class="max-w-full" alt="Classification">`;
            
            // Display statistics
            let tableHtml = '<table class="w-full"><thead><tr class="bg-gray-100">' +
                '<th class="p-2 text-left">Class</th><th class="p-2">Area (ha)</th><th class="p-2">%</th></tr></thead><tbody>';
            
            for (const [className, stats] of Object.entries(data.statistics)) {
                if (className !== 'Background') {
                    tableHtml += `<tr><td class="p-2">${className}</td>` +
                        `<td class="p-2 text-center">${stats.area_hectares}</td>` +
                        `<td class="p-2 text-center">${stats.percentage}%</td></tr>`;
                }
            }
            tableHtml += '</tbody></table>';
            document.getElementById('statistics-table').innerHTML = tableHtml;
        }

        // Change detection
        async function runChangeDetection() {
            const year1 = document.getElementById('year1').value;
            const year2 = document.getElementById('year2').value;
            
            const response = await fetch('/api/change_detection', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({year1: parseInt(year1), year2: parseInt(year2)})
            });
            const data = await response.json();
            
            document.getElementById('before-image').innerHTML = 
                `<img src="data:image/png;base64,${data.year1.image}" class="max-w-full">`;
            document.getElementById('after-image').innerHTML = 
                `<img src="data:image/png;base64,${data.year2.image}" class="max-w-full">`;
            document.getElementById('change-image').innerHTML = 
                `<img src="data:image/png;base64,${data.change_map}" class="max-w-full">`;
            
            // Display change statistics
            let statsHtml = '<h3 class="font-semibold mb-2">Change Statistics</h3><table class="w-full">' +
                '<thead><tr class="bg-gray-100"><th class="p-2">Class</th><th class="p-2">Before</th>' +
                '<th class="p-2">After</th><th class="p-2">Change</th></tr></thead><tbody>';
            
            for (const [className, stats] of Object.entries(data.change_statistics)) {
                const changeColor = stats.change_hectares > 0 ? 'text-green-600' : 'text-red-600';
                statsHtml += `<tr><td class="p-2">${className}</td>` +
                    `<td class="p-2 text-center">${stats.before_hectares} ha</td>` +
                    `<td class="p-2 text-center">${stats.after_hectares} ha</td>` +
                    `<td class="p-2 text-center ${changeColor}">${stats.change_hectares > 0 ? '+' : ''}${stats.change_hectares} ha</td></tr>`;
            }
            statsHtml += '</tbody></table>';
            document.getElementById('change-stats').innerHTML = statsHtml;
        }

        // Load time series
        async function loadTimeSeries() {
            const response = await fetch('/api/time_series');
            const data = await response.json();
            
            const ctx = document.getElementById('timeSeriesChart').getContext('2d');
            
            const colors = {
                'Roads': '#808080',
                'Water': '#0064FF',
                'Forest': '#006400',
                'Plantation': '#32CD32',
                'Buildings': '#8B4513',
                'Bare Land': '#D2B48C'
            };
            
            const datasets = Object.entries(data.data)
                .filter(([name]) => name !== 'Background')
                .map(([name, values]) => ({
                    label: name,
                    data: values.map(v => v.area_hectares),
                    borderColor: colors[name],
                    backgroundColor: colors[name] + '40',
                    fill: false,
                    tension: 0.1
                }));
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.years,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Land Cover Area Over Time (hectares)'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {display: true, text: 'Area (hectares)'}
                        },
                        x: {
                            title: {display: true, text: 'Year'}
                        }
                    }
                }
            });
        }
    </script>
</body>
</html>
'''

# Write template
(TEMPLATES_DIR / 'index.html').write_text(HTML_TEMPLATE, encoding='utf-8')


if __name__ == '__main__':
    # Load model (optional)
    checkpoint_path = Path('checkpoints/best.pt')
    if checkpoint_path.exists():
        load_model(str(checkpoint_path))
    
    # Run app
    print("Starting web server...")
    print("Visit: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
