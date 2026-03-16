"""
Visualization Utilities for Land Cover Analysis
================================================

Provides functions for:
    - Creating color-coded land cover maps
    - Overlaying predictions on satellite imagery
    - Generating comparison visualizations
    - Creating publication-ready figures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import Dict, List, Optional, Tuple
import torch
from pathlib import Path


# Default color scheme
DEFAULT_COLORS = {
    0: (0, 0, 0),         # Background - Black
    1: (128, 128, 128),   # Roads - Grey
    2: (0, 100, 255),     # Water - Blue
    3: (0, 100, 0),       # Forest - Dark Green
    4: (50, 205, 50),     # Plantation - Light Green
    5: (139, 69, 19),     # Buildings - Brown
    6: (210, 180, 140),   # Bare Land - Tan
}

DEFAULT_CLASS_NAMES = {
    0: 'Background',
    1: 'Roads',
    2: 'Water',
    3: 'Forest',
    4: 'Plantation/Crops',
    5: 'Buildings',
    6: 'Bare Land',
}


def create_colormap(colors: Dict[int, Tuple] = None) -> ListedColormap:
    """
    Create a matplotlib colormap from class colors.
    
    Args:
        colors: Dictionary mapping class IDs to RGB tuples (0-255)
        
    Returns:
        ListedColormap for matplotlib
    """
    colors = colors or DEFAULT_COLORS
    
    # Normalize colors to 0-1 range
    color_list = [
        tuple(c / 255 for c in colors.get(i, (0, 0, 0)))
        for i in range(max(colors.keys()) + 1)
    ]
    
    return ListedColormap(color_list)


def prediction_to_rgb(
    prediction: np.ndarray,
    colors: Dict[int, Tuple] = None
) -> np.ndarray:
    """
    Convert class prediction to RGB image.
    
    Args:
        prediction: 2D array of class indices
        colors: Color mapping dictionary
        
    Returns:
        RGB image array (H, W, 3)
    """
    colors = colors or DEFAULT_COLORS
    h, w = prediction.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in colors.items():
        mask = prediction == class_id
        rgb[mask] = color
    
    return rgb


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    colors: Dict[int, Tuple] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay segmentation mask on RGB image.
    
    Args:
        image: RGB image (H, W, 3)
        mask: Class indices (H, W)
        colors: Color mapping
        alpha: Overlay transparency
        
    Returns:
        Overlaid image
    """
    mask_rgb = prediction_to_rgb(mask, colors)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Blend
    overlaid = (alpha * mask_rgb + (1 - alpha) * image).astype(np.uint8)
    
    return overlaid


def visualize_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor = None,
    num_samples: int = 4,
    save_path: str = None,
    class_names: Dict[int, str] = None,
    colors: Dict[int, Tuple] = None
):
    """
    Visualize model predictions.
    
    Args:
        images: Input images (B, C, H, W)
        predictions: Predicted masks (B, H, W)
        targets: Ground truth masks (B, H, W)
        num_samples: Number of samples to show
        save_path: Path to save figure
        class_names: Class name mapping
        colors: Color mapping
    """
    colors = colors or DEFAULT_COLORS
    class_names = class_names or DEFAULT_CLASS_NAMES
    
    num_samples = min(num_samples, images.shape[0])
    
    # Determine number of columns
    num_cols = 2 if targets is None else 3
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(4 * num_cols, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Get RGB from first 3 channels
        img = images[i, :3].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        
        pred = predictions[i].cpu().numpy()
        
        # Input image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        # Prediction
        pred_rgb = prediction_to_rgb(pred, colors)
        axes[i, 1].imshow(pred_rgb)
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
        
        # Ground truth
        if targets is not None:
            target = targets[i].cpu().numpy()
            target_rgb = prediction_to_rgb(target, colors)
            axes[i, 2].imshow(target_rgb)
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
    
    # Add legend
    patches = [
        mpatches.Patch(
            color=tuple(c / 255 for c in colors[i]),
            label=class_names.get(i, f'Class {i}')
        )
        for i in sorted(colors.keys()) if i > 0
    ]
    fig.legend(handles=patches, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_change_detection(
    image_before: np.ndarray,
    image_after: np.ndarray,
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    change_map: np.ndarray = None,
    save_path: str = None,
    title: str = "Land Cover Change Detection"
):
    """
    Create change detection visualization.
    
    Args:
        image_before: RGB image at time 1
        image_after: RGB image at time 2
        mask_before: Classification at time 1
        mask_after: Classification at time 2
        change_map: Optional pre-computed change map
        save_path: Path to save figure
        title: Figure title
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Before
    axes[0, 0].imshow(image_before)
    axes[0, 0].set_title('Image (Before)')
    axes[0, 0].axis('off')
    
    mask_before_rgb = prediction_to_rgb(mask_before)
    axes[0, 1].imshow(mask_before_rgb)
    axes[0, 1].set_title('Classification (Before)')
    axes[0, 1].axis('off')
    
    # After
    axes[1, 0].imshow(image_after)
    axes[1, 0].set_title('Image (After)')
    axes[1, 0].axis('off')
    
    mask_after_rgb = prediction_to_rgb(mask_after)
    axes[1, 1].imshow(mask_after_rgb)
    axes[1, 1].set_title('Classification (After)')
    axes[1, 1].axis('off')
    
    # Change detection
    if change_map is None:
        change_map = (mask_before != mask_after).astype(np.uint8)
    
    # Highlight changes in red
    change_vis = np.zeros((*change_map.shape, 3), dtype=np.uint8)
    change_vis[change_map > 0] = [255, 0, 0]
    
    axes[0, 2].imshow(change_vis)
    axes[0, 2].set_title('Change Map')
    axes[0, 2].axis('off')
    
    # Overlay changes on after image
    overlay = image_after.copy()
    if overlay.max() <= 1:
        overlay = (overlay * 255).astype(np.uint8)
    overlay[change_map > 0] = [255, 0, 0]
    
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Changes Highlighted')
    axes[1, 2].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_time_series(
    data: Dict[str, List[float]],
    years: List[int],
    save_path: str = None,
    title: str = "Land Cover Area Over Time"
):
    """
    Plot time series of land cover areas.
    
    Args:
        data: Dictionary of {class_name: [values]}
        years: List of years
        save_path: Path to save figure
        title: Figure title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        'Roads': '#808080',
        'Water': '#0064FF',
        'Forest': '#006400',
        'Plantation': '#32CD32',
        'Buildings': '#8B4513',
        'Bare Land': '#D2B48C',
    }
    
    for class_name, values in data.items():
        if class_name != 'Background':
            ax.plot(
                years, values,
                marker='o',
                label=class_name,
                color=colors.get(class_name, None),
                linewidth=2
            )
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Area (hectares)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: str = None,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: 2D array
        class_names: List of class names
        normalize: Whether to normalize
        save_path: Path to save figure
        title: Figure title
    """
    if normalize:
        cm = confusion_matrix.astype(float)
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-6)
    else:
        cm = confusion_matrix
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = f'{cm[i, j]:.2f}' if normalize else f'{cm[i, j]}'
            ax.text(
                j, i, value,
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black'
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_publication_figure(
    results: Dict,
    save_path: str,
    title: str = "Land Cover Classification Results - Mauritius"
):
    """
    Create publication-quality figure.
    
    Args:
        results: Dictionary with 'image', 'prediction', 'ground_truth', 'metrics'
        save_path: Path to save figure
        title: Figure title
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Input image
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(results['image'])
    ax1.set_title('Sentinel-2 RGB Composite', fontsize=12)
    ax1.axis('off')
    
    # Prediction
    ax2 = fig.add_subplot(gs[0, 2:4])
    pred_rgb = prediction_to_rgb(results['prediction'])
    ax2.imshow(pred_rgb)
    ax2.set_title('Land Cover Classification', fontsize=12)
    ax2.axis('off')
    
    # Ground truth (if available)
    if 'ground_truth' in results:
        ax3 = fig.add_subplot(gs[1, 0:2])
        gt_rgb = prediction_to_rgb(results['ground_truth'])
        ax3.imshow(gt_rgb)
        ax3.set_title('Ground Truth', fontsize=12)
        ax3.axis('off')
        
        # Difference
        ax4 = fig.add_subplot(gs[1, 2:4])
        diff = (results['prediction'] != results['ground_truth']).astype(np.uint8)
        diff_rgb = np.zeros((*diff.shape, 3), dtype=np.uint8)
        diff_rgb[diff > 0] = [255, 0, 0]
        ax4.imshow(diff_rgb)
        ax4.set_title('Classification Errors', fontsize=12)
        ax4.axis('off')
    
    # Legend
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    patches = [
        mpatches.Patch(
            color=tuple(c / 255 for c in DEFAULT_COLORS[i]),
            label=DEFAULT_CLASS_NAMES[i]
        )
        for i in range(1, len(DEFAULT_COLORS))
    ]
    ax5.legend(handles=patches, loc='center', fontsize=10)
    ax5.set_title('Legend', fontsize=12)
    
    # Metrics table
    if 'metrics' in results:
        ax6 = fig.add_subplot(gs[2, 1:3])
        ax6.axis('off')
        
        metrics = results['metrics']
        table_data = [
            ['Metric', 'Value'],
            ['Accuracy', f"{metrics.get('accuracy', 0):.4f}"],
            ['Mean IoU', f"{metrics.get('miou', 0):.4f}"],
            ['Mean F1', f"{metrics.get('mean_f1', 0):.4f}"],
        ]
        
        table = ax6.table(
            cellText=table_data,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax6.set_title('Performance Metrics', fontsize=12)
    
    # Area statistics
    if 'statistics' in results:
        ax7 = fig.add_subplot(gs[2, 3])
        stats = results['statistics']
        
        classes = list(stats.keys())
        areas = [stats[c].get('area_hectares', 0) for c in classes]
        colors = [tuple(c / 255 for c in DEFAULT_COLORS.get(i, (0, 0, 0))) 
                  for i in range(len(classes))]
        
        ax7.barh(classes, areas, color=colors)
        ax7.set_xlabel('Area (hectares)')
        ax7.set_title('Area Statistics', fontsize=12)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved publication figure to {save_path}")
    plt.close()


if __name__ == '__main__':
    # Test visualizations
    print("Testing visualization utilities...")
    
    # Create sample data
    h, w = 256, 256
    num_classes = 7
    
    # Random prediction
    prediction = np.random.randint(0, num_classes, (h, w))
    
    # Test colormap
    cmap = create_colormap()
    print(f"Created colormap with {cmap.N} colors")
    
    # Test RGB conversion
    rgb = prediction_to_rgb(prediction)
    print(f"RGB image shape: {rgb.shape}")
    
    # Test visualization
    images = torch.randn(2, 9, 256, 256)
    predictions = torch.randint(0, 7, (2, 256, 256))
    targets = torch.randint(0, 7, (2, 256, 256))
    
    visualize_predictions(
        images, predictions, targets,
        num_samples=2,
        save_path='test_visualization.png'
    )
    print("Created test visualization")
