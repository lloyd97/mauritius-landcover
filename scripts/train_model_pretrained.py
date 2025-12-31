"""
Train Land Cover Model with Pre-trained Weights
===============================================

Uses EuroSAT or ImageNet pre-trained encoder with U-Net/DeepLabV3
Fine-tunes on Mauritius-specific land cover data

Usage:
    python train_model_pretrained.py --config configs/config.yaml
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import numpy as np
from tqdm import tqdm
import yaml
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MauritiusLandCoverDataset(Dataset):
    """Dataset for Mauritius land cover tiles"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        # Only load tiles, not masks
        all_npys = list(self.data_dir.glob('*.npy'))
        self.tiles = [f for f in all_npys if '_mask' not in f.name and '_tile_' in f.name]
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        # Load tile (9, H, W)
        tile = np.load(self.tiles[idx]).astype(np.float32)  # Ensure float32

        # Load corresponding mask
        # Change from _tile_XXXX.npy to _tile_XXXX_mask.npy
        mask_path = str(self.tiles[idx]).replace('.npy', '_mask.npy')
        if Path(mask_path).exists():
            mask = np.load(mask_path).astype(np.int64)
        else:
            # If no mask, return zeros
            mask = np.zeros((tile.shape[1], tile.shape[2]), dtype=np.int64)

        # Transpose to (H, W, C) for albumentations
        tile = tile.transpose(1, 2, 0)  # (H, W, 9)

        if self.transform:
            transformed = self.transform(image=tile, mask=mask)
            tile = transformed['image']
            mask = transformed['mask']

        return tile.float(), mask.long()  # Ensure correct dtypes


def get_training_augmentation():
    """Training data augmentation - simplified for multi-channel data"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ])


def get_validation_transform():
    """Validation transform"""
    return A.Compose([
        ToTensorV2(),
    ])


def create_model_with_pretrained(architecture='unet', encoder='resnet50', in_channels=9, num_classes=7):
    """
    Create segmentation model with pre-trained encoder

    Args:
        architecture: 'unet', 'unetplusplus', 'deeplabv3', 'deeplabv3plus'
        encoder: 'resnet50', 'resnet101', 'efficientnet-b0', etc.
        in_channels: Number of input channels
        num_classes: Number of output classes
    """

    print(f"Creating {architecture} with {encoder} encoder...")

    # Create model based on architecture
    if architecture == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet',  # Use ImageNet pre-trained weights
            in_channels=in_channels,
            classes=num_classes,
            activation=None  # We'll use softmax during training
        )

    elif architecture == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )

    elif architecture == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )

    elif architecture == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Modify first conv layer for 9 channels
    if in_channels != 3:
        # Get the first conv layer
        first_conv = model.encoder.conv1 if hasattr(model.encoder, 'conv1') else model.encoder.layer0.conv

        # Create new conv with 9 input channels
        new_conv = nn.Conv2d(
            in_channels, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        # Initialize new channels
        with torch.no_grad():
            # Copy RGB weights to first 3 channels
            new_conv.weight[:, :3, :, :] = first_conv.weight[:, :3, :, :]
            # Initialize extra channels (4-9) with small random values
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            new_conv.weight[:, 3:, :, :] *= 0.01  # Scale down extra channels

        # Replace first conv
        if hasattr(model.encoder, 'conv1'):
            model.encoder.conv1 = new_conv
        else:
            model.encoder.layer0.conv = new_conv

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    return model


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""

    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/training/tiles', help='Training data directory')
    parser.add_argument('--architecture', type=str, default='unet', choices=['unet', 'unetplusplus', 'deeplabv3', 'deeplabv3plus'])
    parser.add_argument('--encoder', type=str, default='resnet50', help='Encoder backbone')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--output', type=str, default='checkpoints/trained_model.pt', help='Output path')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model with pre-trained weights
    model = create_model_with_pretrained(
        architecture=args.architecture,
        encoder=args.encoder,
        in_channels=9,
        num_classes=7
    )
    model = model.to(device)

    # Create datasets
    train_dataset = MauritiusLandCoverDataset(
        args.data_dir,
        transform=get_training_augmentation()
    )

    val_dataset = MauritiusLandCoverDataset(
        args.data_dir,
        transform=get_validation_transform()
    )

    # Split dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = DiceLoss()  # Can also use Dice loss

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'architecture': args.architecture,
                'encoder': args.encoder
            }, output_path)

            print(f"Saved best model to {output_path}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
