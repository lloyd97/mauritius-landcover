"""
Google Earth Engine Data Acquisition for Mauritius
===================================================

This script downloads Sentinel-2 imagery for Mauritius from Google Earth Engine.
Supports multiple time periods for change detection analysis.

Usage:
    python gee_download.py --config configs/config.yaml
    python gee_download.py --start 2015-01-01 --end 2025-01-01

Requirements:
    pip install earthengine-api geemap
    
Setup:
    1. Create GEE account: https://earthengine.google.com/
    2. Authenticate: earthengine authenticate
"""

import ee
import os
import yaml
import argparse
from datetime import datetime
from pathlib import Path
import geemap
import numpy as np

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()


class MauritiusDataDownloader:
    """Download Sentinel-2 imagery for Mauritius from Google Earth Engine."""
    
    # Mauritius bounding box
    MAURITIUS_BOUNDS = {
        'min_lon': 57.30,
        'max_lon': 57.82,
        'min_lat': -20.53,
        'max_lat': -19.98
    }
    
    # Sentinel-2 band specifications
    S2_BANDS = {
        'B2': {'name': 'Blue', 'wavelength': '490nm', 'resolution': 10},
        'B3': {'name': 'Green', 'wavelength': '560nm', 'resolution': 10},
        'B4': {'name': 'Red', 'wavelength': '665nm', 'resolution': 10},
        'B8': {'name': 'NIR', 'wavelength': '842nm', 'resolution': 10},
        'B11': {'name': 'SWIR1', 'wavelength': '1610nm', 'resolution': 20},
        'B12': {'name': 'SWIR2', 'wavelength': '2190nm', 'resolution': 20},
    }
    
    def __init__(self, config_path: str = None):
        """Initialize downloader with configuration."""
        self.config = self._load_config(config_path) if config_path else {}
        self.aoi = self._create_aoi()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_aoi(self) -> ee.Geometry:
        """Create Area of Interest geometry for Mauritius."""
        bounds = self.config.get('aoi', {}).get('bounds', self.MAURITIUS_BOUNDS)
        return ee.Geometry.Rectangle([
            bounds['min_lon'], bounds['min_lat'],
            bounds['max_lon'], bounds['max_lat']
        ])
    
    def _mask_clouds_s2(self, image: ee.Image) -> ee.Image:
        """
        Mask clouds and shadows in Sentinel-2 imagery using SCL band.
        
        SCL Classes:
            0: No data
            1: Saturated/Defective
            2: Dark Area Pixels (shadows)
            3: Cloud Shadows
            4: Vegetation
            5: Bare Soils
            6: Water
            7: Unclassified
            8: Cloud Medium Probability
            9: Cloud High Probability
            10: Thin Cirrus
            11: Snow/Ice
        """
        scl = image.select('SCL')
        
        # Mask out clouds, shadows, and problematic pixels
        mask = (scl.neq(0)   # No data
                .And(scl.neq(1))   # Saturated
                .And(scl.neq(2))   # Dark area (shadows)
                .And(scl.neq(3))   # Cloud shadows
                .And(scl.neq(8))   # Cloud medium
                .And(scl.neq(9))   # Cloud high
                .And(scl.neq(10))) # Cirrus
        
        return image.updateMask(mask)
    
    def _add_indices(self, image: ee.Image) -> ee.Image:
        """Add spectral indices to the image."""
        # NDVI - Normalized Difference Vegetation Index
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # NDWI - Normalized Difference Water Index (McFeeters)
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # NDBI - Normalized Difference Built-up Index
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
        
        # MNDWI - Modified NDWI (better for water detection)
        mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        
        # BSI - Bare Soil Index
        bsi = image.expression(
            '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
            {
                'SWIR': image.select('B11'),
                'RED': image.select('B4'),
                'NIR': image.select('B8'),
                'BLUE': image.select('B2')
            }
        ).rename('BSI')
        
        return image.addBands([ndvi, ndwi, ndbi, mndwi, bsi])
    
    def get_sentinel2_collection(
        self,
        start_date: str,
        end_date: str,
        cloud_cover_max: int = 20
    ) -> ee.ImageCollection:
        """
        Get Sentinel-2 L2A image collection for the specified period.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cloud_cover_max: Maximum cloud cover percentage
            
        Returns:
            Filtered and processed ImageCollection
        """
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(self.aoi)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
                      .map(self._mask_clouds_s2)
                      .map(self._add_indices))
        
        print(f"Found {collection.size().getInfo()} images for {start_date} to {end_date}")
        return collection
    
    def create_composite(
        self,
        start_date: str,
        end_date: str,
        method: str = 'median'
    ) -> ee.Image:
        """
        Create cloud-free composite for the specified period.
        
        Args:
            start_date: Start date
            end_date: End date
            method: Compositing method ('median', 'mean', 'mosaic')
            
        Returns:
            Composite image
        """
        collection = self.get_sentinel2_collection(start_date, end_date)
        
        if method == 'median':
            composite = collection.median()
        elif method == 'mean':
            composite = collection.mean()
        elif method == 'mosaic':
            composite = collection.mosaic()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Select bands and clip to AOI
        bands = list(self.S2_BANDS.keys()) + ['NDVI', 'NDWI', 'NDBI', 'MNDWI', 'BSI']
        composite = composite.select(bands).clip(self.aoi)
        
        return composite
    
    def download_composite(
        self,
        composite: ee.Image,
        output_path: str,
        scale: int = 10,
        crs: str = 'EPSG:32740'
    ):
        """
        Download composite image to local file.
        
        Args:
            composite: Earth Engine image
            output_path: Output file path
            scale: Resolution in meters
            crs: Coordinate reference system
        """
        # Using geemap for download
        geemap.ee_export_image(
            composite,
            filename=output_path,
            scale=scale,
            region=self.aoi,
            crs=crs,
            file_per_band=False
        )
        print(f"Downloaded: {output_path}")
    
    def download_time_series(
        self,
        time_periods: list,
        output_dir: str,
        scale: int = 10
    ):
        """
        Download composites for multiple time periods.
        
        Args:
            time_periods: List of {'name': str, 'start': str, 'end': str}
            output_dir: Output directory
            scale: Resolution in meters
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for period in time_periods:
            print(f"\nProcessing period: {period['name']}")
            
            composite = self.create_composite(period['start'], period['end'])
            output_path = output_dir / f"mauritius_{period['name']}.tif"
            
            self.download_composite(composite, str(output_path), scale=scale)
    
    def visualize_in_map(
        self,
        start_date: str,
        end_date: str,
        output_html: str = None
    ):
        """
        Create interactive map visualization.
        
        Args:
            start_date: Start date
            end_date: End date
            output_html: Optional HTML file to save
        """
        # Create map centered on Mauritius
        Map = geemap.Map(center=[-20.25, 57.55], zoom=10)
        
        # Get composite
        composite = self.create_composite(start_date, end_date)
        
        # Add layers
        vis_params_rgb = {
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000,
            'gamma': 1.2
        }
        
        vis_params_ndvi = {
            'min': -0.2,
            'max': 0.8,
            'palette': ['red', 'yellow', 'green', 'darkgreen']
        }
        
        vis_params_ndwi = {
            'min': -0.5,
            'max': 0.5,
            'palette': ['brown', 'white', 'blue']
        }
        
        Map.addLayer(composite, vis_params_rgb, 'RGB')
        Map.addLayer(composite.select('NDVI'), vis_params_ndvi, 'NDVI')
        Map.addLayer(composite.select('NDWI'), vis_params_ndwi, 'NDWI')
        
        # Add legend
        Map.add_legend(
            title='NDVI',
            legend_dict={
                'Low vegetation': 'red',
                'Moderate': 'yellow',
                'High': 'green',
                'Very high': 'darkgreen'
            }
        )
        
        if output_html:
            Map.to_html(output_html)
            print(f"Map saved to: {output_html}")
        
        return Map


def get_sample_data_for_demo():
    """
    Generate sample data for demonstration when GEE is not available.
    Creates synthetic Sentinel-2-like data for testing.
    """
    import numpy as np
    from PIL import Image
    
    print("Generating sample demonstration data...")
    
    # Create synthetic multi-band image (256x256, 9 bands)
    np.random.seed(42)
    
    # Simulate different land cover patterns
    h, w = 512, 512
    
    # Create base patterns
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    
    # Vegetation pattern (forests/plantations)
    vegetation = np.sin(x * 10) * np.cos(y * 8) * 0.3 + 0.5
    
    # Water bodies (based on distance from center)
    water = np.exp(-((x - 0.3)**2 + (y - 0.7)**2) * 20) * 0.8
    
    # Urban areas (grid pattern)
    urban = (np.sin(x * 30) > 0.8).astype(float) * (np.sin(y * 30) > 0.8).astype(float)
    
    # Create synthetic bands
    bands = {
        'B2': (0.1 + vegetation * 0.1 + water * 0.2) * 3000,  # Blue
        'B3': (0.15 + vegetation * 0.15 + water * 0.15) * 3000,  # Green
        'B4': (0.1 + vegetation * 0.05 + urban * 0.2) * 3000,  # Red
        'B8': (0.3 + vegetation * 0.4 - water * 0.2) * 3000,  # NIR
        'B11': (0.2 + urban * 0.3 - vegetation * 0.1) * 3000,  # SWIR1
        'B12': (0.15 + urban * 0.25 - vegetation * 0.1) * 3000,  # SWIR2
    }
    
    # Add noise
    for band in bands:
        bands[band] += np.random.randn(h, w) * 100
        bands[band] = np.clip(bands[band], 0, 10000)
    
    # Calculate indices
    bands['NDVI'] = (bands['B8'] - bands['B4']) / (bands['B8'] + bands['B4'] + 1e-6)
    bands['NDWI'] = (bands['B3'] - bands['B8']) / (bands['B3'] + bands['B8'] + 1e-6)
    bands['NDBI'] = (bands['B11'] - bands['B8']) / (bands['B11'] + bands['B8'] + 1e-6)
    
    return bands


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Download Sentinel-2 data for Mauritius')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2020-12-31', help='End date')
    parser.add_argument('--output', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--scale', type=int, default=10, help='Resolution in meters')
    parser.add_argument('--demo', action='store_true', help='Generate demo data without GEE')
    
    args = parser.parse_args()
    
    if args.demo:
        # Generate sample data for demonstration
        bands = get_sample_data_for_demo()
        print(f"Generated sample data with {len(bands)} bands")
        print(f"Shape: {bands['B4'].shape}")
        return bands
    
    # Initialize downloader
    downloader = MauritiusDataDownloader(args.config)
    
    if args.config:
        # Use time periods from config
        config = downloader.config
        time_periods = config.get('data', {}).get('time_periods', [])
        downloader.download_time_series(time_periods, args.output, args.scale)
    else:
        # Single download
        composite = downloader.create_composite(args.start, args.end)
        output_path = f"{args.output}/mauritius_{args.start[:4]}.tif"
        downloader.download_composite(composite, output_path, args.scale)


if __name__ == '__main__':
    main()
