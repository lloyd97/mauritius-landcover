"""
Training Pipeline for Land Cover Classification
================================================

Comprehensive training script with:
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving
    - TensorBoard logging
    - Weights & Biases integration (optional)

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --resume checkpoints/last.pt
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import LandCoverDataset, TemporalLandCoverDataset, create_dataloaders
from models.unet import create_model
from models.lstm_unet import create_temporal_model
from utils.metrics import SegmentationMetrics, compute_class_weights
from utils.visualization import visualize_predictions


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Dice Loss for semantic segmentation."""
    
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predictions (B, C, H, W)
            target: Ground truth (B, H, W)
        """
        num_classes = pred.shape[1]
        
        # Softmax predictions
        pred = torch.softmax(pred, dim=1)
        
        # Create one-hot encoding
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Mask ignore index
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index).unsqueeze(1).float()
            pred = pred * mask
            target_one_hot = target_one_hot * mask
        
        # Compute Dice
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy and Dice Loss."""
    
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ):
        super().__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


class Trainer:
    """Training manager for land cover classification."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Computation device
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Training config
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 100)
        self.start_epoch = 0
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config.get('learning_rate', 1e-4),
            weight_decay=train_config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        scheduler_config = train_config.get('scheduler', {})
        if scheduler_config.get('type') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        
        # Loss function
        loss_config = train_config.get('loss', {})
        class_weights = self._compute_class_weights() if loss_config.get('class_weights') == 'auto' else None
        
        self.criterion = CombinedLoss(
            ce_weight=loss_config.get('ce_weight', 0.5),
            dice_weight=loss_config.get('dice_weight', 0.5),
            class_weights=class_weights
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        self.use_amp = True
        
        # Early stopping
        es_config = train_config.get('early_stopping', {})
        self.early_stopping_patience = es_config.get('patience', 15)
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        
        # Metrics
        num_classes = config.get('model', {}).get('num_classes', 7)
        self.metrics = SegmentationMetrics(num_classes)
        
        # Logging
        self.log_dir = Path('runs') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Checkpoints
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _compute_class_weights(self) -> torch.Tensor:
        """Compute class weights from training data."""
        logger.info("Computing class weights...")
        
        class_counts = torch.zeros(self.config.get('model', {}).get('num_classes', 7))
        
        for batch in tqdm(self.train_loader, desc="Counting classes"):
            mask = batch['mask']
            for c in range(len(class_counts)):
                class_counts[c] += (mask == c).sum()
        
        # Inverse frequency weighting
        weights = 1.0 / (class_counts + 1)
        weights = weights / weights.sum()
        
        logger.info(f"Class weights: {weights}")
        return weights.to(self.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        self.metrics.reset()
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Get data
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            self.metrics.update(preds, masks)
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        metrics = self.metrics.compute()
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0
        self.metrics.reset()
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            self.metrics.update(preds, masks)
        
        metrics = self.metrics.compute()
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save last checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'last.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            logger.info(f"Saved best model at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self):
        """Full training loop."""
        logger.info(f"Starting training for {self.epochs} epochs")
        
        for epoch in range(self.start_epoch, self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"mIoU: {train_metrics['miou']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"mIoU: {val_metrics['miou']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}")
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # Log to TensorBoard
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")
        
        self.writer.close()
        logger.info("Training completed!")
        
        return self.best_val_loss
    
    def _log_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log metrics to TensorBoard."""
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
        
        # Log learning rate
        self.writer.add_scalar(
            'train/lr',
            self.optimizer.param_groups[0]['lr'],
            epoch
        )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train land cover classification model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Path to data directory')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders
    data_config = config.get('data', {})
    train_loader, val_loader = create_dataloaders(
        config,
        train_dir=f"{args.data_dir}/train",
        val_dir=f"{args.data_dir}/val",
        mask_dir=f"{args.data_dir}/masks"
    )
    
    # Create model
    model_config = config.get('model', {})
    if model_config.get('lstm', {}).get('enabled', False):
        model = create_temporal_model(config)
    else:
        model = create_model(config)
    
    logger.info(f"Created model: {model_config.get('architecture', 'unet')}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
