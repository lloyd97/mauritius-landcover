"""
Retrain 9-class model V2 - Improved training pipeline
======================================================

Improvements over V1:
1. Transfer learning from working 7-class model (not ImageNet)
2. Focal Loss + Dice Loss (better for class imbalance + segmentation)
3. Enhanced augmentation (spectral + geometric)
4. Cosine annealing with warm restarts
5. Gradient accumulation for effective batch size 16
6. Save best model by mIoU (not just val_loss)

Classes: 0=Clouds, 1=Water, 2=Forest, 3=Plantation, 4=Urban, 5=Roads, 6=BareLand, 7=Ocean, 8=Wasteland
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import segmentation_models_pytorch as smp
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
import sys

# 9-class scheme
CLASSES = {
    0: 'Clouds',
    1: 'Water',
    2: 'Forest',
    3: 'Plantation',
    4: 'Urban',
    5: 'Roads',
    6: 'Bare Land',
    7: 'Ocean',
    8: 'Wasteland'
}
NUM_CLASSES = 9

# Old class IDs in training masks
OLD_BG = 0
OLD_ROADS = 1
OLD_WATER = 2
OLD_FOREST = 3
OLD_PLANTATION = 4
OLD_BUILDINGS = 5
OLD_BARE_LAND = 6


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss - reduces easy-sample dominance, focuses on hard pixels."""

    def __init__(self, gamma=2.0, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight,
                                  ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss - better for segmentation boundary quality."""

    def __init__(self, num_classes=9, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, input, target):
        probs = F.softmax(input, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)  # sum over batch, height, width
        intersection = (probs * target_onehot).sum(dims)
        cardinality = (probs + target_onehot).sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1 - dice).mean()


class CombinedLoss(nn.Module):
    """Combined Focal + Dice loss."""

    def __init__(self, num_classes=9, gamma=2.0, weight=None, focal_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, weight=weight)
        self.dice = DiceLoss(num_classes=num_classes)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, input, target):
        return self.focal_weight * self.focal(input, target) + \
               self.dice_weight * self.dice(input, target)


# ============================================================
# DATA
# ============================================================

def relabel_mask(old_mask, bands_data):
    """Convert 7-class mask to 9-class mask using actual spectral properties."""
    new_mask = np.zeros_like(old_mask)

    ndvi = bands_data[6].copy()
    ndwi = bands_data[7].copy()

    if np.abs(ndvi).max() > 1.5:
        ndvi = np.clip(ndvi / 10000.0, -1, 1)
    if np.abs(ndwi).max() > 1.5:
        ndwi = np.clip(ndwi / 10000.0, -1, 1)

    blue, green, red = bands_data[0].copy(), bands_data[1].copy(), bands_data[2].copy()
    if blue.max() > 100:
        blue, green, red = blue / 10000.0, green / 10000.0, red / 10000.0
    brightness = (blue + green + red) / 3.0

    # Old "Water"(2) is actually dense forest (NDVI=0.72) → Forest(2)
    new_mask[old_mask == OLD_WATER] = 2

    # Old "Roads"(1) is actually water (NDWI=0.50) → Water(1)
    new_mask[old_mask == OLD_ROADS] = 1

    # Old "Forest"(3) is moderate vegetation → split Forest/Wasteland
    forest_px = old_mask == OLD_FOREST
    new_mask[forest_px & (ndvi >= 0.55)] = 2   # Forest
    new_mask[forest_px & (ndvi < 0.55)] = 8    # Wasteland

    # Old "Plantation"(4) → Plantation(3)
    new_mask[old_mask == OLD_PLANTATION] = 3

    # Old "Buildings"(5) → Urban(4)
    new_mask[old_mask == OLD_BUILDINGS] = 4

    # Old "Bare Land"(6) → Bare Land(6)
    new_mask[old_mask == OLD_BARE_LAND] = 6

    # Background(0) → classify by spectral properties
    bg_px = old_mask == OLD_BG
    new_mask[bg_px & (ndwi > 0.3)] = 7   # Ocean
    new_mask[bg_px & (ndwi > 0.0) & (ndwi <= 0.3)] = 1  # Shallow water
    new_mask[bg_px & (ndwi <= 0.0) & (brightness > 0.25)] = 0  # Clouds
    bg_land = bg_px & (ndwi <= 0.0) & (brightness <= 0.25)
    new_mask[bg_land & (ndvi > 0.5)] = 2   # Forest
    new_mask[bg_land & (ndvi >= 0.2) & (ndvi <= 0.5)] = 8  # Wasteland
    new_mask[bg_land & (ndvi < 0.2)] = 6   # Bare Land

    return new_mask


class MauritiusDataset9Class(Dataset):
    """Dataset that relabels masks from 7 to 9 classes on-the-fly."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        all_npys = list(self.data_dir.glob('*.npy'))
        self.tiles = sorted([f for f in all_npys if '_mask' not in f.name and '_tile_' in f.name])
        self.transform = transform

        # Validate masks exist
        valid_tiles = []
        for tile in self.tiles:
            mask_path = str(tile).replace('.npy', '_mask.npy')
            if Path(mask_path).exists():
                valid_tiles.append(tile)
            else:
                print(f"  WARNING: Missing mask for {tile.name}, skipping")
        self.tiles = valid_tiles

        print(f"Loaded {len(self.tiles)} tiles from {data_dir}")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = np.load(self.tiles[idx]).astype(np.float32)
        mask_path = str(self.tiles[idx]).replace('.npy', '_mask.npy')
        old_mask = np.load(mask_path).astype(np.int64)
        mask = relabel_mask(old_mask, tile)

        # Transpose to (H, W, C) for albumentations
        tile = tile.transpose(1, 2, 0)  # (256, 256, 9)

        if self.transform:
            transformed = self.transform(image=tile, mask=mask)
            tile = transformed['image']
            mask = transformed['mask']

        return tile.float(), mask.long()


def get_training_augmentation():
    """Enhanced augmentation: geometric + spectral."""
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=20, p=0.4),

        # Spectral/intensity augmentations
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.GaussNoise(p=0.2),

        # Spatial augmentations
        A.ElasticTransform(alpha=50, sigma=5, p=0.2),
        A.CoarseDropout(p=0.2),

        ToTensorV2(),
    ])


def get_validation_transform():
    return A.Compose([ToTensorV2()])


# ============================================================
# MODEL
# ============================================================

def create_model_from_7class(checkpoint_path, num_classes=9, in_channels=9):
    """
    Create 9-class model by transferring weights from the working 7-class model.
    Much better starting point than ImageNet - already knows Mauritius spectral patterns.
    """
    print(f"\nTransfer learning from 7-class model: {checkpoint_path}")

    # First create the 7-class model to load weights
    model_7 = smp.Unet(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=7,
        activation=None
    )

    # Modify first conv for 9 channels
    first_conv = model_7.encoder.conv1
    new_conv = nn.Conv2d(
        in_channels, first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None
    )
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = first_conv.weight[:, :3, :, :]
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
        new_conv.weight[:, 3:, :, :] *= 0.01
    model_7.encoder.conv1 = new_conv

    # Load 7-class checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_7.load_state_dict(state_dict)
    print(f"  Loaded 7-class weights (epoch {checkpoint.get('epoch', '?')})")

    # Now create 9-class model with same architecture
    model_9 = smp.Unet(
        encoder_name='resnet50',
        encoder_weights=None,  # Don't load ImageNet - we have better weights
        in_channels=3,
        classes=num_classes,
        activation=None
    )

    # Copy the modified first conv
    model_9.encoder.conv1 = nn.Conv2d(
        in_channels, new_conv.out_channels,
        kernel_size=new_conv.kernel_size,
        stride=new_conv.stride,
        padding=new_conv.padding,
        bias=new_conv.bias is not None
    )

    # Transfer all weights except the final segmentation head
    state_9 = model_9.state_dict()
    transferred = 0
    skipped = 0

    for name, param in model_7.state_dict().items():
        if name in state_9:
            if state_9[name].shape == param.shape:
                state_9[name] = param
                transferred += 1
            else:
                # Shape mismatch (segmentation head: 7 → 9 classes)
                print(f"  Shape mismatch: {name} {param.shape} -> {state_9[name].shape}")
                if len(param.shape) >= 1 and param.shape[0] == 7:
                    # Copy first 7 class weights, init remaining 2 randomly
                    state_9[name][:7] = param
                    if len(param.shape) > 1:
                        nn.init.kaiming_normal_(state_9[name][7:], mode='fan_out', nonlinearity='relu')
                    else:
                        state_9[name][7:] = param.mean()
                    transferred += 1
                else:
                    skipped += 1
        else:
            skipped += 1

    model_9.load_state_dict(state_9)
    print(f"  Transferred {transferred} parameter tensors, skipped {skipped}")

    total_params = sum(p.numel() for p in model_9.parameters())
    print(f"  Model: U-Net + resnet50, {total_params:,} params, {num_classes} classes")

    return model_9


def create_model_from_scratch(num_classes=9, in_channels=9):
    """Fallback: create model from ImageNet weights."""
    print("\nCreating model from ImageNet weights (no 7-class transfer)")
    model = smp.Unet(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=num_classes,
        activation=None
    )

    first_conv = model.encoder.conv1
    new_conv = nn.Conv2d(
        in_channels, first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None
    )
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = first_conv.weight[:, :3, :, :]
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
        new_conv.weight[:, 3:, :, :] *= 0.01
    model.encoder.conv1 = new_conv

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: U-Net + resnet50, {total_params:,} params, {num_classes} classes")
    return model


# ============================================================
# METRICS
# ============================================================

def calculate_iou(pred, target, num_classes=9):
    """Per-class IoU."""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    ious = []
    for c in range(num_classes):
        pc, tc = (pred == c), (target == c)
        inter = np.logical_and(pc, tc).sum()
        union = np.logical_or(pc, tc).sum()
        ious.append(inter / union if union > 0 else (1.0 if inter == 0 else 0.0))
    return ious


def calculate_class_weights(dataset, num_classes=9):
    """Calculate inverse-frequency class weights from dataset."""
    print("\nCalculating class weights...")
    class_counts = np.zeros(num_classes)

    for i in tqdm(range(len(dataset)), desc="Scanning classes"):
        _, mask = dataset[i]
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        for c in range(num_classes):
            class_counts[c] += np.sum(mask_np == c)

    total = class_counts.sum()
    class_counts_safe = np.maximum(class_counts, 1)
    weights = total / (num_classes * class_counts_safe)
    weights = np.clip(weights, 0.5, 10.0)
    weights = weights / weights.sum() * num_classes

    print("\nClass distribution after relabeling:")
    for i in range(num_classes):
        pct = 100 * class_counts[i] / total if total > 0 else 0
        print(f"  {CLASSES[i]:12s}: {class_counts[i]:>10.0f} px ({pct:5.2f}%) weight={weights[i]:.3f}")

    return torch.FloatTensor(weights)


# ============================================================
# COSINE ANNEALING WITH WARM RESTARTS
# ============================================================

class CosineAnnealingWarmRestartsWithWarmup:
    """Cosine annealing with warm restarts and initial warmup."""

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=1e-7, warmup_epochs=5):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            factor = self.current_epoch / max(1, self.warmup_epochs)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * factor
        else:
            # Cosine annealing with warm restarts
            epoch_since_warmup = self.current_epoch - self.warmup_epochs
            T_cur = epoch_since_warmup % self.T_0
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.eta_min + (base_lr - self.eta_min) * \
                           (1 + math.cos(math.pi * T_cur / self.T_0)) / 2


# ============================================================
# TRAINING
# ============================================================

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_dir = 'data/training/tiles'

    # Create datasets
    train_ds = MauritiusDataset9Class(data_dir, transform=get_training_augmentation())
    val_ds = MauritiusDataset9Class(data_dir, transform=get_validation_transform())

    # Deterministic 80/20 split
    n = len(train_ds)
    indices = list(range(n))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    train_size = int(0.8 * n)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds, val_indices)

    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")

    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=0)

    # Model - transfer from 7-class
    checkpoint_7class = Path('checkpoints/enhanced_model_7class_working_backup.pth')
    if not checkpoint_7class.exists():
        checkpoint_7class = Path('checkpoints/enhanced_model_old7class.pth')
    if not checkpoint_7class.exists():
        checkpoint_7class = Path('checkpoints/enhanced_model_backup_before_wasteland.pth')

    if checkpoint_7class.exists():
        model = create_model_from_7class(str(checkpoint_7class), NUM_CLASSES).to(device)
    else:
        print("WARNING: No 7-class checkpoint found, starting from ImageNet")
        model = create_model_from_scratch(NUM_CLASSES).to(device)

    # Class weights
    count_ds = MauritiusDataset9Class(data_dir, transform=get_validation_transform())
    class_weights = calculate_class_weights(count_ds, NUM_CLASSES).to(device)

    # Combined Focal + Dice loss
    criterion = CombinedLoss(
        num_classes=NUM_CLASSES,
        gamma=2.0,
        weight=class_weights,
        focal_weight=0.5,
        dice_weight=0.5
    )

    # Discriminative learning rates (lower for encoder since it's pretrained)
    encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n]
    decoder_params = [p for n, p in model.named_parameters() if 'encoder' not in n]
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 1e-5},
        {'params': decoder_params, 'lr': 1e-4}
    ], weight_decay=1e-4)

    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestartsWithWarmup(
        optimizer, T_0=20, T_mult=1, eta_min=1e-7, warmup_epochs=3
    )

    # Training config
    epochs = 100
    patience_limit = 25
    patience_counter = 0
    best_miou = 0.0
    best_val_loss = float('inf')
    accumulation_steps = 4  # Effective batch size = 4 * 4 = 16

    output_path = Path('checkpoints/enhanced_model_9class_v2.pth')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training 9-class model V2 for {epochs} epochs")
    print(f"  Loss: Focal (gamma=2.0) + Dice")
    print(f"  Optimizer: AdamW (encoder LR=1e-5, decoder LR=1e-4)")
    print(f"  Scheduler: Cosine annealing (T_0=20, warmup=3)")
    print(f"  Gradient accumulation: {accumulation_steps} steps (effective batch=16)")
    print(f"  Early stopping patience: {patience_limit}")
    print(f"  Save criterion: best mIoU")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.3f}'})

        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0
        all_ious = [[] for _ in range(NUM_CLASSES)]

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                ious = calculate_iou(preds, masks, NUM_CLASSES)
                for i in range(NUM_CLASSES):
                    all_ious[i].append(ious[i])

        val_loss /= len(val_loader)

        mean_ious = [np.mean(x) if x else 0 for x in all_ious]
        mean_iou = np.mean(mean_ious)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}: Train={train_loss:.4f} Val={val_loss:.4f} mIoU={mean_iou:.4f} LR={lr:.2e}")
        for i in range(NUM_CLASSES):
            print(f"  {CLASSES[i]:12s}: IoU={mean_ious[i]:.4f}")

        # Save best by mIoU
        if mean_iou > best_miou:
            best_miou = mean_iou
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': mean_ious,
                'mean_iou': mean_iou,
                'num_classes': NUM_CLASSES,
                'class_names': CLASSES,
                'version': 'v2',
            }, str(output_path))
            print(f"  >>> Saved best model (mIoU={mean_iou:.4f}, val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\nEarly stopping at epoch {epoch} (no mIoU improvement for {patience_limit} epochs)")
                break

        print()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    train()
