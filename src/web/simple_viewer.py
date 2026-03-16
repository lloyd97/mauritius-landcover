"""
Simple Land Cover Classification Viewer
Uses pre-downloaded training tiles to demonstrate the trained model
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from flask import Flask, render_template_string, jsonify
import segmentation_models_pytorch as smp
from PIL import Image
import io
import base64

app = Flask(__name__)

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None
TILES_DIR = Path('data/training/tiles')

# Land cover classes with proper colors
CLASSES = {
    0: {'name': 'Background', 'color': [0, 0, 0]},
    1: {'name': 'Water', 'color': [0, 100, 255]},          # Blue
    2: {'name': 'Forest', 'color': [0, 100, 0]},           # Dark Green
    3: {'name': 'Plantation', 'color': [50, 205, 50]},     # Light Green
    4: {'name': 'Urban', 'color': [139, 69, 19]},          # Brown
    5: {'name': 'Roads', 'color': [128, 128, 128]},        # Grey
    6: {'name': 'Bare Land', 'color': [210, 180, 140]}     # Tan
}


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

    # Load checkpoint
    checkpoint_path = Path('checkpoints/best_model.pth')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        print(f"SUCCESS: Loaded model from {checkpoint_path}")
    else:
        print("WARNING: No checkpoint found, using untrained model")

    MODEL.to(DEVICE)
    MODEL.eval()


def get_all_tiles():
    """Get list of all available tiles"""
    all_npys = list(TILES_DIR.glob('*.npy'))
    tiles = [f for f in all_npys if '_mask' not in f.name and '_tile_' in f.name]
    return sorted(tiles)


def create_rgb_from_tile(tile_data):
    """Create RGB visualization from 9-channel tile data"""
    # tile_data shape: (9, H, W)
    # Channels: B2(Blue), B3(Green), B4(Red), B8(NIR), B11(SWIR1), B12(SWIR2), NDVI, NDWI, NDBI

    blue = tile_data[0, :, :]
    green = tile_data[1, :, :]
    red = tile_data[2, :, :]

    # Normalize to 0-255
    def normalize(band):
        band = band.copy()
        vmin, vmax = np.percentile(band, [2, 98])
        band = np.clip((band - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255)
        return band.astype(np.uint8)

    rgb = np.stack([
        normalize(red),
        normalize(green),
        normalize(blue)
    ], axis=-1)

    return rgb


def classify_tile(tile_data):
    """Run model inference on tile"""
    # tile_data: (9, H, W)
    input_tensor = torch.from_numpy(tile_data).float().unsqueeze(0)  # (1, 9, H, W)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        output = MODEL(input_tensor)  # (1, 7, H, W)
        prediction = torch.argmax(output, dim=1).squeeze(0)  # (H, W)
        prediction = prediction.cpu().numpy()

    return prediction


def create_colored_mask(prediction):
    """Convert class indices to RGB colored image"""
    h, w = prediction.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, info in CLASSES.items():
        mask = prediction == class_id
        colored[mask] = info['color']

    return colored


def array_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


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

    # Sort by percentage descending
    stats.sort(key=lambda x: x['percentage'], reverse=True)
    return stats


# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Mauritius Land Cover - Trained Model Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f2f5;
        }

        #header {
            background: linear-gradient(135deg, #2d5016 0%, #4a7c24 100%);
            color: white;
            padding: 20px 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }

        #header h1 {
            font-size: 28px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
        }

        #header h1::before {
            content: "🇲🇺";
            margin-right: 12px;
            font-size: 32px;
        }

        #header p {
            font-size: 15px;
            opacity: 0.95;
            margin-left: 44px;
        }

        #controls {
            background: white;
            padding: 20px 30px;
            border-bottom: 1px solid #ddd;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        #controls button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 15px;
            font-weight: 500;
            margin-right: 10px;
            transition: all 0.3s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        #controls button:hover {
            background: #45a049;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }

        #controls button:active {
            transform: translateY(0);
        }

        #tile-selector {
            margin-left: 20px;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
            background: white;
        }

        #container {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            padding: 30px;
            max-width: 1800px;
            margin: 0 auto;
        }

        .panel {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s;
        }

        .panel:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }

        .panel-header {
            padding: 15px 20px;
            font-weight: 600;
            font-size: 16px;
            border-bottom: 2px solid #f0f2f5;
            display: flex;
            align-items: center;
        }

        .panel-header::before {
            font-size: 20px;
            margin-right: 10px;
        }

        .panel:nth-child(1) .panel-header { background: #e3f2fd; }
        .panel:nth-child(2) .panel-header { background: #e8f5e9; }
        .panel:nth-child(3) .panel-header { background: #fff3e0; }

        .panel:nth-child(1) .panel-header::before { content: "🛰️"; }
        .panel:nth-child(2) .panel-header::before { content: "🗺️"; }
        .panel:nth-child(3) .panel-header::before { content: "📊"; }

        .panel-content {
            padding: 20px;
            min-height: 400px;
        }

        .image-container {
            text-align: center;
        }

        .image-container img {
            max-width: 100%;
            height: auto;
            border: 2px solid #eee;
            border-radius: 8px;
        }

        .legend {
            margin-top: 20px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            margin: 10px 0;
            padding: 8px 12px;
            background: #f8f9fa;
            border-radius: 6px;
        }

        .legend-color {
            width: 30px;
            height: 30px;
            border-radius: 4px;
            margin-right: 12px;
            border: 2px solid #ddd;
        }

        .legend-text {
            flex: 1;
            font-size: 14px;
        }

        .legend-percentage {
            font-weight: 600;
            font-size: 15px;
            color: #333;
        }

        #loading {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 255, 255, 0.95);
            padding: 30px 50px;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
            display: none;
            z-index: 1000;
            text-align: center;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div id="header">
        <h1>Mauritius Land Cover Classification</h1>
        <p>Trained U-Net Model with ResNet50 Encoder | Validation Loss: 0.698</p>
    </div>

    <div id="controls">
        <button onclick="loadRandomTile()">🎲 Load Random Tile</button>
        <select id="tile-selector" onchange="loadSpecificTile()">
            <option value="">Select a specific tile...</option>
        </select>
    </div>

    <div id="loading">
        <div class="spinner"></div>
        <div>Classifying...</div>
    </div>

    <div id="container">
        <div class="panel">
            <div class="panel-header">Sentinel-2 Satellite Image</div>
            <div class="panel-content">
                <div class="image-container" id="satellite-container">
                    <p style="color: #999;">Click "Load Random Tile" to begin</p>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">Land Cover Classification</div>
            <div class="panel-content">
                <div class="image-container" id="classification-container">
                    <p style="color: #999;">Classification will appear here</p>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">Class Distribution</div>
            <div class="panel-content">
                <div class="legend" id="legend">
                    <p style="color: #999;">Statistics will appear here</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let allTiles = [];

        // Load available tiles list
        fetch('/api/tiles')
            .then(r => r.json())
            .then(data => {
                allTiles = data.tiles;
                const select = document.getElementById('tile-selector');
                data.tiles.forEach((tile, idx) => {
                    const option = document.createElement('option');
                    option.value = idx;
                    option.textContent = tile;
                    select.appendChild(option);
                });
            });

        function loadRandomTile() {
            if (allTiles.length === 0) return;
            const randomIdx = Math.floor(Math.random() * allTiles.length);
            loadTile(randomIdx);
        }

        function loadSpecificTile() {
            const select = document.getElementById('tile-selector');
            const idx = select.value;
            if (idx !== '') {
                loadTile(parseInt(idx));
            }
        }

        function loadTile(idx) {
            document.getElementById('loading').style.display = 'block';

            fetch('/api/classify/' + idx)
                .then(r => r.json())
                .then(data => {
                    // Update satellite image
                    document.getElementById('satellite-container').innerHTML =
                        '<img src="data:image/png;base64,' + data.satellite + '" alt="Satellite">';

                    // Update classification
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

                    document.getElementById('loading').style.display = 'none';
                })
                .catch(err => {
                    console.error(err);
                    document.getElementById('loading').style.display = 'none';
                    alert('Error loading tile: ' + err);
                });
        }

        // Load first tile on page load
        setTimeout(() => {
            if (allTiles.length > 0) {
                loadRandomTile();
            }
        }, 500);
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/tiles')
def get_tiles():
    """Get list of available tiles"""
    tiles = get_all_tiles()
    tile_names = [t.name for t in tiles]
    return jsonify({'tiles': tile_names})


@app.route('/api/classify/<int:tile_idx>')
def classify_tile_route(tile_idx):
    """Classify a specific tile"""
    try:
        tiles = get_all_tiles()
        if tile_idx < 0 or tile_idx >= len(tiles):
            return jsonify({'error': 'Invalid tile index'}), 400

        # Load tile data
        tile_path = tiles[tile_idx]
        tile_data = np.load(tile_path).astype(np.float32)  # (9, 256, 256)

        print(f"Classifying tile: {tile_path.name}")

        # Create RGB visualization
        rgb_image = create_rgb_from_tile(tile_data)

        # Classify
        prediction = classify_tile(tile_data)

        # Create colored classification
        classification_colored = create_colored_mask(prediction)

        # Get statistics
        statistics = get_class_statistics(prediction)

        # Convert to base64
        satellite_b64 = array_to_base64(rgb_image)
        classification_b64 = array_to_base64(classification_colored)

        return jsonify({
            'satellite': satellite_b64,
            'classification': classification_b64,
            'statistics': statistics,
            'tile_name': tile_path.name
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Mauritius Land Cover - Simple Viewer")
    print("=" * 60)
    print("\nLoading model...")
    load_model()

    print(f"\nUsing device: {DEVICE}")
    print(f"Tiles directory: {TILES_DIR}")

    tiles = get_all_tiles()
    print(f"Found {len(tiles)} training tiles\n")

    print("=" * 60)
    print("Starting web server...")
    print("Visit: http://localhost:5001")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5001, debug=True)
