"""
Enhanced training script for Mauritius land cover classification
Inspired by best practices from:
- pavlo-seimskyi/semantic-segmentation-satellite-imagery
- souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch

Improvements:
1. Discriminative learning rates (smaller for encoder, larger for decoder)
2. Learning rate scheduling with ReduceLROnPlateau
3. Per-class IoU and F1 metrics
4. Better logging and checkpointing
5. Mixed precision training support
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
from pathlib import Path
import argparse
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from datetime import datetime

# Land cover classes
CLASSES = {
    'background': 0,
    'water': 1,
    'forest': 2,
    'plantation': 3,
    'urban': 4,
    'roads': 5,
    'bare_land': 6
}

CLASS_NAMES = list(CLASSES.keys())
NUM_CLASSES = len(CLASSES)


class MauritiusDataset(Dataset):
    """Enhanced dataset with better error handling"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        all_npys = list(self.data_dir.glob('*.npy'))
        self.tiles = sorted([f for f in all_npys if '_mask' not in f.name and '_tile_' in f.name])
        self.transform = transform

        # Validate that all tiles have masks
        missing_masks = []
        for tile in self.tiles:
            mask_path = str(tile).replace('.npy', '_mask.npy')
            if not Path(mask_path).exists():
                missing_masks.append(tile.name)

        if missing_masks:
            raise ValueError(f"Missing masks for tiles: {missing_masks}")

        print(f"Loaded {len(self.tiles)} tiles from {data_dir}")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        # Load tile and ensure float32
        tile = np.load(self.tiles[idx]).astype(np.float32)

        # Load mask
        mask_path = str(self.tiles[idx]).replace('.npy', '_mask.npy')
        mask = np.load(mask_path).astype(np.int64)

        # Transpose to (H, W, C) for albumentations
        tile = tile.transpose(1, 2, 0)  # (9, 256, 256) -> (256, 256, 9)

        if self.transform:
            transformed = self.transform(image=tile, mask=mask)
            tile = transformed['image']
            mask = transformed['mask']

        return tile.float(), mask.long()


def get_training_augmentation():
    """Training augmentation - geometric only for multi-channel data"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        # Could add ShiftScaleRotate for more variation
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
        ToTensorV2(),
    ])


def get_validation_transform():
    """Validation transform"""
    return A.Compose([
        ToTensorV2(),
    ])


def create_model(architecture='unet', encoder='resnet50', in_channels=9, num_classes=7):
    """Create model with pre-trained encoder"""

    print(f"Creating {architecture} with {encoder} encoder...")

    # Create model
    if architecture == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=3,  # Start with 3 for ImageNet weights
            classes=num_classes,
            activation=None
        )
    elif architecture == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    elif architecture == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    elif architecture == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
            activation=None
        )

    # Modify first conv layer for 9 channels
    if in_channels != 3:
        first_conv = model.encoder.conv1 if hasattr(model.encoder, 'conv1') else model.encoder.layer0.conv

        new_conv = nn.Conv2d(
            in_channels, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        # Initialize: copy RGB weights, initialize extra channels
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = first_conv.weight[:, :3, :, :]
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            new_conv.weight[:, 3:, :, :] *= 0.01

        # Replace
        if hasattr(model.encoder, 'conv1'):
            model.encoder.conv1 = new_conv
        else:
            model.encoder.layer0.conv = new_conv

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {total_params:,} parameters ({trainable_params:,} trainable)")

    return model


def get_discriminative_params(model, encoder_lr, decoder_lr):
    """
    Create parameter groups with discriminative learning rates
    Inspired by pavlo-seimskyi's approach
    """
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    return [
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': decoder_params, 'lr': decoder_lr}
    ]


def calculate_class_weights(data_dir, num_classes=7):
    """
    Calculate class weights from training data to handle imbalance
    Returns tensor of weights (inverse frequency)
    """
    print("Calculating class weights from training data...")
    tiles_dir = Path(data_dir)
    mask_files = list(tiles_dir.glob('*_mask.npy'))

    if not mask_files:
        print("WARNING: No mask files found, using equal weights")
        return torch.ones(num_classes)

    # Count pixels per class
    class_counts = np.zeros(num_classes)

    for mask_file in mask_files:
        mask = np.load(mask_file)
        for class_id in range(num_classes):
            class_counts[class_id] += np.sum(mask == class_id)

    # Calculate weights (inverse frequency with smoothing)
    total_pixels = class_counts.sum()

    # Handle classes with zero samples
    class_counts = np.maximum(class_counts, 1)  # Avoid division by zero

    # Inverse frequency weighting
    class_weights = total_pixels / (num_classes * class_counts)

    # Cap weights to prevent extreme values
    class_weights = np.clip(class_weights, 0.5, 10.0)

    # Normalize weights so they sum to num_classes
    class_weights = class_weights / class_weights.sum() * num_classes

    print(f"Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        pct = 100 * class_counts[i] / total_pixels if total_pixels > 0 else 0
        print(f"  {name:12s}: {pct:5.2f}% (weight: {class_weights[i]:.3f})")

    return torch.FloatTensor(class_weights)


def calculate_metrics(pred, target, num_classes=7):
    """
    Calculate per-class IoU and F1 scores
    Inspired by souvikmajumder26's evaluation approach
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    iou_scores = []
    f1_scores = []

    for class_id in range(num_classes):
        pred_class = (pred == class_id)
        target_class = (target == class_id)

        intersection = np.logical_and(pred_class, target_class).sum()
        union = np.logical_or(pred_class, target_class).sum()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        # F1 score
        true_positive = intersection
        false_positive = pred_class.sum() - intersection
        false_negative = target_class.sum() - intersection

        precision = true_positive / (true_positive + false_positive + 1e-10)
        recall = true_positive / (true_positive + false_negative + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        iou_scores.append(iou)
        f1_scores.append(f1)

    return iou_scores, f1_scores


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Training epoch with optional mixed precision"""
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.2f}'})

    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device, num_classes=7):
    """Validation with detailed metrics"""
    model.eval()
    total_loss = 0
    all_ious = [[] for _ in range(num_classes)]
    all_f1s = [[] for _ in range(num_classes)]

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # Calculate metrics
            preds = outputs.argmax(dim=1)
            ious, f1s = calculate_metrics(preds, masks, num_classes)

            for i in range(num_classes):
                all_ious[i].append(ious[i])
                all_f1s[i].append(f1s[i])

    avg_loss = total_loss / len(loader)

    # Average metrics per class
    mean_ious = [np.mean(ious) for ious in all_ious]
    mean_f1s = [np.mean(f1s) for f1s in all_f1s]

    return avg_loss, mean_ious, mean_f1s


def main():
    parser = argparse.ArgumentParser(description='Enhanced training for land cover classification')
    parser.add_argument('--data-dir', type=str, default='data/training/tiles')
    parser.add_argument('--architecture', choices=['unet', 'unetplusplus', 'deeplabv3', 'deeplabv3plus'],
                       default='unet')
    parser.add_argument('--encoder', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--encoder-lr', type=float, default=0.00001,
                       help='Learning rate for encoder (smaller than decoder)')
    parser.add_argument('--decoder-lr', type=float, default=0.0001,
                       help='Learning rate for decoder')
    parser.add_argument('--output', type=str, default='checkpoints/enhanced_model.pth')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training (requires CUDA)')
    parser.add_argument('--use-class-weights', action='store_true',
                       help='Use class weights to handle imbalanced data')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    full_dataset = MauritiusDataset(args.data_dir, transform=None)

    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    # Apply transforms
    train_dataset.dataset.transform = get_training_augmentation()
    val_dataset.dataset.transform = get_validation_transform()

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = create_model(args.architecture, args.encoder, in_channels=9, num_classes=NUM_CLASSES)
    model = model.to(device)

    # Loss with class weights
    if args.use_class_weights:
        class_weights = calculate_class_weights(args.data_dir, NUM_CLASSES)
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class-weighted loss")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"Using standard CrossEntropyLoss")

    # Optimizer with discriminative learning rates
    param_groups = get_discriminative_params(model, args.encoder_lr, args.decoder_lr)
    optimizer = torch.optim.Adam(param_groups)

    print(f"Discriminative LR: Encoder={args.encoder_lr}, Decoder={args.decoder_lr}")

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None

    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_f1': []
    }

    print(f"\n{'='*60}")
    print(f"Starting enhanced training - {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # Validate
        val_loss, val_ious, val_f1s = validate_epoch(model, val_loader, criterion, device, NUM_CLASSES)

        # Scheduler step
        scheduler.step(val_loss)

        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_ious)
        history['val_f1'].append(val_f1s)

        # Print summary
        mean_iou = np.mean(val_ious)
        mean_f1 = np.mean(val_f1s)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}, Mean F1: {mean_f1:.4f}")

        # Per-class metrics
        print("\nPer-class IoU:")
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"  {class_name:12s}: IoU={val_ious[i]:.4f}, F1={val_f1s[i]:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_ious,
                'val_f1': val_f1s,
                'args': vars(args)
            }, args.output)
            print(f"Saved best model to {args.output}")

        print()

    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'val_iou': [list(x) for x in history['val_iou']],
            'val_f1': [list(x) for x in history['val_f1']]
        }
        json.dump(history_json, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.output}")
    print(f"History saved to: {history_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
