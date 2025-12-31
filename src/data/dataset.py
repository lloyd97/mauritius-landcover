"""
PyTorch Dataset Classes for Satellite Imagery
==============================================

Provides dataset classes for loading and augmenting satellite imagery
for land cover classification.

Classes:
    - LandCoverDataset: Single-image dataset
    - TemporalLandCoverDataset: Multi-temporal dataset for LSTM
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import rasterio
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Optional, Dict
import yaml


class LandCoverDataset(Dataset):
    """
    Dataset for single-image land cover classification.
    
    Loads satellite imagery tiles and corresponding labels for
    semantic segmentation training.
    """
    
    # Class color mapping (RGB)
    CLASS_COLORS = {
        0: (0, 0, 0),         # Background
        1: (128, 128, 128),   # Roads
        2: (0, 100, 255),     # Water
        3: (0, 100, 0),       # Forest
        4: (50, 205, 50),     # Plantation
        5: (139, 69, 19),     # Buildings
        6: (210, 180, 140),   # Bare land
    }
    
    CLASS_NAMES = {
        0: 'background',
        1: 'roads',
        2: 'water',
        3: 'forest',
        4: 'plantation',
        5: 'buildings',
        6: 'bare_land',
    }
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str = None,
        transform: A.Compose = None,
        tile_size: int = 256,
        bands: List[str] = None,
        normalize: bool = True,
        return_metadata: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            image_dir: Directory containing satellite images
            mask_dir: Directory containing label masks (None for inference)
            transform: Albumentations transform pipeline
            tile_size: Size of image tiles
            bands: List of band names to use
            normalize: Whether to normalize pixel values
            return_metadata: Whether to return image metadata
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.tile_size = tile_size
        self.normalize = normalize
        self.return_metadata = return_metadata
        
        # Default bands
        self.bands = bands or ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI']
        
        # Get list of image files
        self.image_files = sorted(list(self.image_dir.glob('*.tif')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No .tif files found in {image_dir}")
        
        # Setup transforms
        self.transform = transform or self._get_default_transform()
        
        # Band statistics for normalization (Sentinel-2 typical values)
        self.band_stats = {
            'B2': {'mean': 0.1, 'std': 0.05},
            'B3': {'mean': 0.1, 'std': 0.05},
            'B4': {'mean': 0.1, 'std': 0.05},
            'B8': {'mean': 0.3, 'std': 0.1},
            'B11': {'mean': 0.2, 'std': 0.08},
            'B12': {'mean': 0.15, 'std': 0.07},
            'NDVI': {'mean': 0.3, 'std': 0.25},
            'NDWI': {'mean': -0.1, 'std': 0.3},
            'NDBI': {'mean': -0.1, 'std': 0.2},
        }
        
        print(f"Loaded {len(self.image_files)} images")
        
    def _get_default_transform(self, training: bool = True) -> A.Compose:
        """Get default augmentation pipeline."""
        if training:
            return A.Compose([
                A.RandomCrop(self.tile_size, self.tile_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.CenterCrop(self.tile_size, self.tile_size),
                ToTensorV2()
            ])
    
    def _normalize_bands(self, image: np.ndarray) -> np.ndarray:
        """Normalize image bands."""
        normalized = np.zeros_like(image, dtype=np.float32)
        
        for i, band in enumerate(self.bands):
            if band in self.band_stats:
                mean = self.band_stats[band]['mean']
                std = self.band_stats[band]['std']
                # Scale to [0, 1] range first if needed
                if image[:, :, i].max() > 1:
                    normalized[:, :, i] = image[:, :, i] / 10000.0  # Sentinel-2 scale
                else:
                    normalized[:, :, i] = image[:, :, i]
                # Z-score normalization
                normalized[:, :, i] = (normalized[:, :, i] - mean) / (std + 1e-6)
            else:
                normalized[:, :, i] = image[:, :, i]
        
        return normalized
    
    def _load_image(self, path: Path) -> Tuple[np.ndarray, dict]:
        """Load multi-band satellite image."""
        with rasterio.open(path) as src:
            # Read all bands
            image = src.read()  # Shape: (bands, height, width)
            image = np.transpose(image, (1, 2, 0))  # Shape: (height, width, bands)
            
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'width': src.width,
                'height': src.height,
            }
        
        return image.astype(np.float32), metadata
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load segmentation mask."""
        with rasterio.open(path) as src:
            mask = src.read(1)  # Single band
        return mask.astype(np.int64)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary with 'image' tensor and optionally 'mask' tensor
        """
        # Load image
        image_path = self.image_files[idx]
        image, metadata = self._load_image(image_path)
        
        # Load mask if available
        mask = None
        if self.mask_dir:
            mask_path = self.mask_dir / image_path.name.replace('.tif', '_mask.tif')
            if mask_path.exists():
                mask = self._load_mask(mask_path)
        
        # Normalize
        if self.normalize:
            image = self._normalize_bands(image)
        
        # Apply transforms
        if self.transform:
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        
        # Prepare output
        sample = {'image': image.float()}
        
        if mask is not None:
            sample['mask'] = torch.tensor(mask, dtype=torch.long)
        
        if self.return_metadata:
            sample['metadata'] = metadata
            sample['filename'] = image_path.name
        
        return sample
    
    @staticmethod
    def get_train_val_transforms(tile_size: int = 256):
        """Get separate transforms for training and validation."""
        train_transform = A.Compose([
            A.RandomCrop(tile_size, tile_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.RandomGamma(gamma_limit=(80, 120)),
            ], p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            ToTensorV2()
        ])
        
        val_transform = A.Compose([
            A.CenterCrop(tile_size, tile_size),
            ToTensorV2()
        ])
        
        return train_transform, val_transform


class TemporalLandCoverDataset(Dataset):
    """
    Dataset for multi-temporal land cover classification.
    
    Loads sequences of satellite imagery for LSTM-based models.
    Supports change detection and temporal analysis.
    """
    
    def __init__(
        self,
        image_dirs: List[str],
        mask_dir: str = None,
        transform: A.Compose = None,
        tile_size: int = 256,
        sequence_length: int = 6,
        bands: List[str] = None,
        normalize: bool = True
    ):
        """
        Initialize temporal dataset.
        
        Args:
            image_dirs: List of directories for each time step
            mask_dir: Directory containing label masks
            transform: Albumentations transform pipeline
            tile_size: Size of image tiles
            sequence_length: Number of temporal images per sample
            bands: List of band names to use
            normalize: Whether to normalize pixel values
        """
        self.image_dirs = [Path(d) for d in image_dirs]
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.tile_size = tile_size
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        self.bands = bands or ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI']
        
        # Get common files across all time steps
        self._find_common_files()
        
        self.transform = transform
        
        print(f"Loaded {len(self.common_files)} temporal sequences")
    
    def _find_common_files(self):
        """Find files that exist in all time step directories."""
        file_sets = []
        for dir_path in self.image_dirs:
            files = {f.name for f in dir_path.glob('*.tif')}
            file_sets.append(files)
        
        # Find intersection
        self.common_files = sorted(list(set.intersection(*file_sets)))
        
        if len(self.common_files) == 0:
            raise ValueError("No common files found across time steps")
    
    def _load_sequence(self, filename: str) -> np.ndarray:
        """Load image sequence for all time steps."""
        sequence = []
        
        for dir_path in self.image_dirs:
            path = dir_path / filename
            with rasterio.open(path) as src:
                image = src.read()
                image = np.transpose(image, (1, 2, 0))
            sequence.append(image)
        
        # Stack: (time, height, width, channels)
        return np.stack(sequence, axis=0).astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.common_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a temporal sample.
        
        Returns:
            Dictionary with 'image' tensor (T, C, H, W) and optionally 'mask'
        """
        filename = self.common_files[idx]
        
        # Load sequence
        sequence = self._load_sequence(filename)  # (T, H, W, C)
        
        # Load mask (from latest time step usually)
        mask = None
        if self.mask_dir:
            mask_path = self.mask_dir / filename.replace('.tif', '_mask.tif')
            if mask_path.exists():
                with rasterio.open(mask_path) as src:
                    mask = src.read(1).astype(np.int64)
        
        # Apply same spatial transform to all time steps
        if self.transform:
            # Get random crop coordinates
            h, w = sequence.shape[1:3]
            if h > self.tile_size and w > self.tile_size:
                top = np.random.randint(0, h - self.tile_size)
                left = np.random.randint(0, w - self.tile_size)
                
                sequence = sequence[:, top:top+self.tile_size, left:left+self.tile_size, :]
                if mask is not None:
                    mask = mask[top:top+self.tile_size, left:left+self.tile_size]
        
        # Convert to tensor: (T, C, H, W)
        sequence = torch.from_numpy(sequence).permute(0, 3, 1, 2).float()
        
        sample = {'image': sequence}
        
        if mask is not None:
            sample['mask'] = torch.tensor(mask, dtype=torch.long)
        
        return sample


class TileGenerator:
    """
    Generate tiles from large satellite images.
    
    Useful for creating training data and for inference on large images.
    """
    
    def __init__(
        self,
        tile_size: int = 256,
        overlap: int = 32,
        min_valid_ratio: float = 0.5
    ):
        """
        Initialize tile generator.
        
        Args:
            tile_size: Size of output tiles
            overlap: Overlap between adjacent tiles
            min_valid_ratio: Minimum ratio of valid (non-zero) pixels
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_valid_ratio = min_valid_ratio
        self.stride = tile_size - overlap
    
    def generate_tiles(
        self,
        image_path: str,
        output_dir: str,
        mask_path: str = None
    ) -> List[str]:
        """
        Generate tiles from a large image.
        
        Args:
            image_path: Path to source image
            output_dir: Directory to save tiles
            mask_path: Optional path to mask image
            
        Returns:
            List of generated tile paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_tiles = []
        
        with rasterio.open(image_path) as src:
            height, width = src.height, src.width
            profile = src.profile.copy()
            
            # Calculate number of tiles
            n_rows = (height - self.overlap) // self.stride + 1
            n_cols = (width - self.overlap) // self.stride + 1
            
            print(f"Generating {n_rows * n_cols} tiles...")
            
            for i in range(n_rows):
                for j in range(n_cols):
                    # Calculate window
                    row_start = i * self.stride
                    col_start = j * self.stride
                    
                    # Handle edge cases
                    row_end = min(row_start + self.tile_size, height)
                    col_end = min(col_start + self.tile_size, width)
                    
                    window = Window(col_start, row_start, 
                                    col_end - col_start, row_end - row_start)
                    
                    # Read tile
                    tile = src.read(window=window)
                    
                    # Check valid ratio
                    valid_ratio = np.sum(tile != 0) / tile.size
                    if valid_ratio < self.min_valid_ratio:
                        continue
                    
                    # Pad if necessary
                    if tile.shape[1] < self.tile_size or tile.shape[2] < self.tile_size:
                        padded = np.zeros((tile.shape[0], self.tile_size, self.tile_size))
                        padded[:, :tile.shape[1], :tile.shape[2]] = tile
                        tile = padded
                    
                    # Save tile
                    tile_name = f"tile_{i:04d}_{j:04d}.tif"
                    tile_path = output_dir / tile_name
                    
                    tile_profile = profile.copy()
                    tile_profile.update({
                        'height': self.tile_size,
                        'width': self.tile_size,
                        'transform': rasterio.windows.transform(window, src.transform)
                    })
                    
                    with rasterio.open(tile_path, 'w', **tile_profile) as dst:
                        dst.write(tile)
                    
                    generated_tiles.append(str(tile_path))
        
        print(f"Generated {len(generated_tiles)} valid tiles")
        return generated_tiles


def create_dataloaders(
    config: dict,
    train_dir: str,
    val_dir: str,
    mask_dir: str = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        train_dir: Training images directory
        val_dir: Validation images directory
        mask_dir: Masks directory
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    tile_size = config.get('data', {}).get('tile_size', 256)
    batch_size = config.get('training', {}).get('batch_size', 16)
    
    train_transform, val_transform = LandCoverDataset.get_train_val_transforms(tile_size)
    
    train_dataset = LandCoverDataset(
        image_dir=train_dir,
        mask_dir=mask_dir,
        transform=train_transform,
        tile_size=tile_size
    )
    
    val_dataset = LandCoverDataset(
        image_dir=val_dir,
        mask_dir=mask_dir,
        transform=val_transform,
        tile_size=tile_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset
    print("Testing LandCoverDataset...")
    
    # Create sample data for testing
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy image
        dummy_data = np.random.rand(9, 512, 512).astype(np.float32)
        
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'width': 512,
            'height': 512,
            'count': 9,
            'crs': 'EPSG:32740',
        }
        
        img_path = Path(tmpdir) / 'test_image.tif'
        with rasterio.open(img_path, 'w', **profile) as dst:
            dst.write(dummy_data)
        
        # Test dataset
        dataset = LandCoverDataset(tmpdir, tile_size=256)
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
