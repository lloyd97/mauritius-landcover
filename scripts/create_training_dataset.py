"""
Create Training Dataset for Mauritius Land Cover Classification
================================================================

This script:
1. Downloads Sentinel-2 imagery from different areas of Mauritius
2. Creates tiles for training
3. Allows manual labeling or semi-automatic annotation
4. Builds a dataset for training the U-Net model

Usage:
    python create_training_dataset.py --download
    python create_training_dataset.py --create-samples
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import cv2
from datetime import datetime
import json

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import ee
    import geemap
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    print("WARNING: Google Earth Engine not available")

import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt


# Mauritius regions to sample (diverse areas)
SAMPLING_LOCATIONS = [
    {"name": "Port_Louis", "lat": -20.1609, "lon": 57.5012, "type": "urban"},
    {"name": "Black_River", "lat": -20.3717, "lon": 57.3717, "type": "coast"},
    {"name": "Grand_Bassin", "lat": -20.4167, "lon": 57.4833, "type": "forest"},
    {"name": "Pamplemousses", "lat": -20.1083, "lon": 57.5667, "type": "agricultural"},
    {"name": "Curepipe", "lat": -20.3167, "lon": 57.5167, "type": "urban"},
    {"name": "Riviere_du_Rempart", "lat": -20.0583, "lon": 57.6583, "type": "agricultural"},
    {"name": "Mahebourg", "lat": -20.4067, "lon": 57.7000, "type": "coastal"},
    {"name": "Central_Plateau", "lat": -20.2333, "lon": 57.4833, "type": "mixed"},
]

# Land cover classes
CLASSES = {
    0: {"name": "background", "color": [0, 0, 0]},
    1: {"name": "roads", "color": [128, 128, 128]},
    2: {"name": "water", "color": [0, 100, 255]},
    3: {"name": "forest", "color": [0, 100, 0]},
    4: {"name": "plantation", "color": [50, 205, 50]},
    5: {"name": "buildings", "color": [139, 69, 19]},
    6: {"name": "bare_land", "color": [210, 180, 140]},
}


def download_sentinel2_tile(lat, lon, name, output_dir, size=512):
    """Download a Sentinel-2 tile for a specific location"""

    if not GEE_AVAILABLE:
        print(f"Skipping {name} - GEE not available")
        return None

    try:
        # Initialize with the configured project
        ee.Initialize(project='ee-lloydflorens12111997')
    except Exception as init_error:
        print(f"GEE initialization failed for {name}: {init_error}")
        print("Try running: earthengine authenticate")
        return None

    print(f"\nDownloading tile: {name} ({lat}, {lon})")

    # Create bounds (approx 5km x 5km)
    buffer = 0.025  # degrees
    bounds = ee.Geometry.Rectangle([
        lon - buffer, lat - buffer,
        lon + buffer, lat + buffer
    ])

    # Get recent Sentinel-2 composite
    end_date = datetime.now()
    start_date = datetime(end_date.year, end_date.month - 2, 1)

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(bounds)
          .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

    composite = s2.median().clip(bounds)

    # Select bands
    bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    composite_bands = composite.select(bands)

    # Compute indices
    ndvi = composite_bands.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = composite_bands.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = composite_bands.normalizedDifference(['B11', 'B8']).rename('NDBI')

    full_composite = composite_bands.addBands([ndvi, ndwi, ndbi])

    # Download
    output_path = output_dir / f'{name}_sentinel2.tif'

    geemap.ee_export_image(
        full_composite,
        filename=str(output_path),
        scale=10,
        region=bounds,
        file_per_band=False
    )

    print(f"Saved to: {output_path}")
    return output_path


def create_rgb_preview(tif_path, output_dir):
    """Create RGB preview of the tile"""
    with rasterio.open(tif_path) as src:
        # Read RGB bands (B4, B3, B2)
        r = src.read(3)  # B4 - Red
        g = src.read(2)  # B3 - Green
        b = src.read(1)  # B2 - Blue

        # Normalize to 0-255
        def normalize(band):
            band = np.clip(band, 0, 3000)
            return ((band / 3000) * 255).astype(np.uint8)

        rgb = np.dstack([normalize(r), normalize(g), normalize(b)])

        # Save preview
        preview_path = output_dir / f'{tif_path.stem}_preview.png'
        cv2.imwrite(str(preview_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        return preview_path, rgb


def create_tiles(image_path, output_dir, tile_size=256, overlap=32):
    """Split large image into training tiles"""

    tiles_dir = output_dir / 'tiles'
    tiles_dir.mkdir(exist_ok=True)

    with rasterio.open(image_path) as src:
        data = src.read()  # (9, H, W)
        height, width = data.shape[1], data.shape[2]

    tile_count = 0
    stride = tile_size - overlap

    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile_data = data[:, y:y+tile_size, x:x+tile_size]

            # Save tile
            tile_path = tiles_dir / f'{image_path.stem}_tile_{tile_count:04d}.npy'
            np.save(tile_path, tile_data)

            # Create RGB preview
            rgb_tile = np.dstack([
                tile_data[2],  # R
                tile_data[1],  # G
                tile_data[0]   # B
            ])
            rgb_tile = ((rgb_tile - rgb_tile.min()) / (rgb_tile.max() - rgb_tile.min()) * 255).astype(np.uint8)

            preview_path = tiles_dir / f'{image_path.stem}_tile_{tile_count:04d}.png'
            cv2.imwrite(str(preview_path), cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2BGR))

            tile_count += 1

    print(f"Created {tile_count} tiles from {image_path.name}")
    return tile_count


def create_labeling_tool_html(tiles_dir):
    """Create a simple HTML tool for labeling tiles"""

    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Mauritius Land Cover Labeling Tool</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        #canvas { border: 2px solid #000; cursor: crosshair; }
        .controls { margin: 20px 0; }
        button { margin: 5px; padding: 10px; font-size: 14px; }
        .class-btn { min-width: 120px; }
        .selected { background: #4CAF50; color: white; font-weight: bold; }
        #preview { display: flex; gap: 20px; margin: 20px 0; }
        .legend { background: #f0f0f0; padding: 15px; border-radius: 5px; }
        .legend-item { margin: 5px 0; display: flex; align-items: center; }
        .legend-color { width: 30px; height: 20px; margin-right: 10px; border: 1px solid #000; }
    </style>
</head>
<body>
    <h1>Mauritius Land Cover Labeling Tool</h1>

    <div class="legend">
        <h3>Classes (Click to select brush)</h3>
        <div class="legend-item"><div class="legend-color" style="background: rgb(128,128,128)"></div>
            <button class="class-btn" onclick="selectClass(1, this)">1 - Roads (Grey)</button></div>
        <div class="legend-item"><div class="legend-color" style="background: rgb(0,100,255)"></div>
            <button class="class-btn" onclick="selectClass(2, this)">2 - Water (Blue)</button></div>
        <div class="legend-item"><div class="legend-color" style="background: rgb(0,100,0)"></div>
            <button class="class-btn" onclick="selectClass(3, this)">3 - Forest (Dark Green)</button></div>
        <div class="legend-item"><div class="legend-color" style="background: rgb(50,205,50)"></div>
            <button class="class-btn" onclick="selectClass(4, this)">4 - Plantation (Light Green)</button></div>
        <div class="legend-item"><div class="legend-color" style="background: rgb(139,69,19)"></div>
            <button class="class-btn" onclick="selectClass(5, this)">5 - Buildings (Brown)</button></div>
        <div class="legend-item"><div class="legend-color" style="background: rgb(210,180,140)"></div>
            <button class="class-btn" onclick="selectClass(6, this)">6 - Bare Land (Tan)</button></div>
    </div>

    <div class="controls">
        <label>Brush Size: <input type="range" id="brushSize" min="1" max="20" value="5"> <span id="brushValue">5</span></label>
    </div>

    <div id="preview">
        <div>
            <h3>Satellite Image</h3>
            <canvas id="imageCanvas" width="512" height="512"></canvas>
        </div>
        <div>
            <h3>Label Mask</h3>
            <canvas id="canvas" width="512" height="512"></canvas>
        </div>
    </div>

    <div class="controls">
        <button onclick="clearMask()">Clear Mask</button>
        <button onclick="saveMask()">Save Mask</button>
        <button onclick="nextImage()">Next Image</button>
        <input type="file" id="imageInput" accept="image/*" onchange="loadImage(event)">
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const imageCanvas = document.getElementById('imageCanvas');
        const imageCtx = imageCanvas.getContext('2d');

        let currentClass = 1;
        let brushSize = 5;
        let isDrawing = false;
        let maskData = new Uint8Array(512 * 512).fill(0);

        const classColors = {
            0: [0, 0, 0],
            1: [128, 128, 128],
            2: [0, 100, 255],
            3: [0, 100, 0],
            4: [50, 205, 50],
            5: [139, 69, 19],
            6: [210, 180, 140]
        };

        document.getElementById('brushSize').oninput = function() {
            brushSize = this.value;
            document.getElementById('brushValue').textContent = brushSize;
        };

        function selectClass(classId, btn) {
            currentClass = classId;
            document.querySelectorAll('.class-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
        }

        canvas.onmousedown = () => isDrawing = true;
        canvas.onmouseup = () => isDrawing = false;
        canvas.onmouseleave = () => isDrawing = false;

        canvas.onmousemove = function(e) {
            if (!isDrawing) return;

            const rect = canvas.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) * 512 / rect.width);
            const y = Math.floor((e.clientY - rect.top) * 512 / rect.height);

            // Draw on mask
            for (let dy = -brushSize; dy <= brushSize; dy++) {
                for (let dx = -brushSize; dx <= brushSize; dx++) {
                    if (dx*dx + dy*dy <= brushSize*brushSize) {
                        const px = x + dx;
                        const py = y + dy;
                        if (px >= 0 && px < 512 && py >= 0 && py < 512) {
                            maskData[py * 512 + px] = currentClass;
                        }
                    }
                }
            }

            renderMask();
        };

        function renderMask() {
            const imageData = ctx.createImageData(512, 512);
            for (let i = 0; i < maskData.length; i++) {
                const classId = maskData[i];
                const color = classColors[classId];
                imageData.data[i*4] = color[0];
                imageData.data[i*4+1] = color[1];
                imageData.data[i*4+2] = color[2];
                imageData.data[i*4+3] = classId === 0 ? 100 : 200;
            }
            ctx.putImageData(imageData, 0, 0);
        }

        function clearMask() {
            maskData.fill(0);
            renderMask();
        }

        function saveMask() {
            // Download mask as JSON
            const blob = new Blob([JSON.stringify(Array.from(maskData))], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'mask.json';
            a.click();
        }

        function loadImage(event) {
            const file = event.target.files[0];
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = new Image();
                img.onload = function() {
                    imageCtx.drawImage(img, 0, 0, 512, 512);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        function nextImage() {
            saveMask();
            clearMask();
            alert('Load next image using the file input');
        }

        // Initialize
        clearMask();
    </script>
</body>
</html>
"""

    html_path = tiles_dir.parent / 'labeling_tool.html'
    html_path.write_text(html_content, encoding='utf-8')
    print(f"\nLabeling tool created: {html_path}")
    print(f"Open this file in your browser to label tiles!")

    return html_path


def main():
    parser = argparse.ArgumentParser(description='Create training dataset for Mauritius land cover')
    parser.add_argument('--download', action='store_true', help='Download Sentinel-2 tiles')
    parser.add_argument('--create-tiles', action='store_true', help='Create training tiles')
    parser.add_argument('--create-labeling-tool', action='store_true', help='Create labeling tool')
    parser.add_argument('--output-dir', type=str, default='data/training', help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        print("=" * 60)
        print("Downloading Sentinel-2 tiles from Mauritius")
        print("=" * 60)

        for location in SAMPLING_LOCATIONS:
            download_sentinel2_tile(
                location['lat'],
                location['lon'],
                location['name'],
                output_dir
            )

    if args.create_tiles:
        print("\n" + "=" * 60)
        print("Creating training tiles")
        print("=" * 60)

        tif_files = list(output_dir.glob('*.tif'))
        for tif_file in tif_files:
            create_tiles(tif_file, output_dir)

    if args.create_labeling_tool:
        create_labeling_tool_html(output_dir)

    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Open the labeling tool: {output_dir}/labeling_tool.html")
    print(f"2. Label the tiles in: {output_dir}/tiles/")
    print(f"3. Train the model with the labeled data")


if __name__ == '__main__':
    main()
