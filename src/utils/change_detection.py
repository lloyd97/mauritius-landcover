"""
Change Detection Utilities
==========================

Comprehensive tools for detecting and analyzing land cover changes
between different time periods.

Methods:
    - Post-classification comparison
    - Image differencing
    - Change vector analysis
    - Object-based change detection
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from pathlib import Path


class ChangeType(Enum):
    """Types of land cover change."""
    NO_CHANGE = 0
    URBANIZATION = 1        # Natural/Agriculture -> Built-up
    DEFORESTATION = 2       # Forest -> Other
    AGRICULTURAL_EXPANSION = 3  # Other -> Agriculture
    AGRICULTURAL_ABANDONMENT = 4  # Agriculture -> Natural
    WATER_CHANGE = 5        # Water body changes
    ROAD_DEVELOPMENT = 6    # New roads


@dataclass
class ChangeStatistics:
    """Statistics for land cover change analysis."""
    total_area: float
    changed_area: float
    change_percentage: float
    class_transitions: Dict[Tuple[int, int], float]
    per_class_gain: Dict[int, float]
    per_class_loss: Dict[int, float]
    net_change: Dict[int, float]


class PostClassificationComparison:
    """
    Post-classification comparison for change detection.
    
    Compares two classified land cover maps to identify changes.
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        class_names: Dict[int, str] = None,
        pixel_area: float = 100.0  # sq meters per pixel (10m resolution)
    ):
        """
        Initialize change detector.
        
        Args:
            num_classes: Number of land cover classes
            class_names: Mapping of class IDs to names
            pixel_area: Area per pixel in square meters
        """
        self.num_classes = num_classes
        self.pixel_area = pixel_area
        self.class_names = class_names or {
            0: 'Background',
            1: 'Roads',
            2: 'Water',
            3: 'Forest',
            4: 'Plantation',
            5: 'Buildings',
            6: 'Bare Land'
        }
    
    def detect_changes(
        self,
        map_before: np.ndarray,
        map_after: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect changes between two land cover maps.
        
        Args:
            map_before: Classification at time 1
            map_after: Classification at time 2
            
        Returns:
            binary_change: Binary change map (0=no change, 1=change)
            change_type: Map showing type of change
        """
        # Binary change detection
        binary_change = (map_before != map_after).astype(np.uint8)
        
        # Compute change type map
        # Encode as: from_class * num_classes + to_class
        change_type = np.where(
            binary_change,
            map_before * self.num_classes + map_after,
            0
        ).astype(np.int16)
        
        return binary_change, change_type
    
    def compute_transition_matrix(
        self,
        map_before: np.ndarray,
        map_after: np.ndarray
    ) -> np.ndarray:
        """
        Compute transition matrix between classes.
        
        Args:
            map_before: Classification at time 1
            map_after: Classification at time 2
            
        Returns:
            Transition matrix (from_class x to_class)
        """
        matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                matrix[i, j] = np.sum((map_before == i) & (map_after == j))
        
        return matrix
    
    def compute_statistics(
        self,
        map_before: np.ndarray,
        map_after: np.ndarray
    ) -> ChangeStatistics:
        """
        Compute comprehensive change statistics.
        
        Args:
            map_before: Classification at time 1
            map_after: Classification at time 2
            
        Returns:
            ChangeStatistics object
        """
        binary_change, _ = self.detect_changes(map_before, map_after)
        transition_matrix = self.compute_transition_matrix(map_before, map_after)
        
        total_pixels = map_before.size
        changed_pixels = binary_change.sum()
        
        # Convert to area (hectares)
        pixel_to_hectare = self.pixel_area / 10000
        total_area = total_pixels * pixel_to_hectare
        changed_area = changed_pixels * pixel_to_hectare
        
        # Class transitions
        transitions = {}
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and transition_matrix[i, j] > 0:
                    transitions[(i, j)] = transition_matrix[i, j] * pixel_to_hectare
        
        # Per-class gain/loss
        per_class_gain = {}
        per_class_loss = {}
        net_change = {}
        
        for c in range(self.num_classes):
            # Gain = pixels that became this class
            gain = transition_matrix[:, c].sum() - transition_matrix[c, c]
            # Loss = pixels that were this class but changed
            loss = transition_matrix[c, :].sum() - transition_matrix[c, c]
            
            per_class_gain[c] = gain * pixel_to_hectare
            per_class_loss[c] = loss * pixel_to_hectare
            net_change[c] = (gain - loss) * pixel_to_hectare
        
        return ChangeStatistics(
            total_area=total_area,
            changed_area=changed_area,
            change_percentage=(changed_pixels / total_pixels) * 100,
            class_transitions=transitions,
            per_class_gain=per_class_gain,
            per_class_loss=per_class_loss,
            net_change=net_change
        )
    
    def categorize_change(
        self,
        from_class: int,
        to_class: int
    ) -> ChangeType:
        """
        Categorize the type of land cover change.
        
        Args:
            from_class: Original class
            to_class: New class
            
        Returns:
            ChangeType enum
        """
        if from_class == to_class:
            return ChangeType.NO_CHANGE
        
        # Urbanization: natural/agriculture -> buildings
        if to_class == 5 and from_class in [3, 4, 6]:
            return ChangeType.URBANIZATION
        
        # Deforestation: forest -> other
        if from_class == 3 and to_class != 3:
            return ChangeType.DEFORESTATION
        
        # Agricultural expansion: other -> plantation
        if to_class == 4 and from_class in [3, 6]:
            return ChangeType.AGRICULTURAL_EXPANSION
        
        # Agricultural abandonment: plantation -> natural
        if from_class == 4 and to_class in [3, 6]:
            return ChangeType.AGRICULTURAL_ABANDONMENT
        
        # Water changes
        if from_class == 2 or to_class == 2:
            return ChangeType.WATER_CHANGE
        
        # Road development
        if to_class == 1 and from_class != 1:
            return ChangeType.ROAD_DEVELOPMENT
        
        return ChangeType.NO_CHANGE
    
    def generate_report(
        self,
        statistics: ChangeStatistics,
        year_before: int,
        year_after: int
    ) -> str:
        """
        Generate text report of changes.
        
        Args:
            statistics: Computed statistics
            year_before: Start year
            year_after: End year
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append(f"LAND COVER CHANGE REPORT: {year_before} - {year_after}")
        report.append("=" * 60)
        report.append("")
        
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total area analyzed: {statistics.total_area:,.2f} hectares")
        report.append(f"Area changed: {statistics.changed_area:,.2f} hectares")
        report.append(f"Change percentage: {statistics.change_percentage:.2f}%")
        report.append("")
        
        report.append("NET CHANGE BY CLASS (hectares)")
        report.append("-" * 40)
        for class_id, change in statistics.net_change.items():
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            sign = "+" if change > 0 else ""
            report.append(f"  {class_name}: {sign}{change:,.2f}")
        report.append("")
        
        report.append("MAJOR TRANSITIONS (>10 ha)")
        report.append("-" * 40)
        for (from_c, to_c), area in sorted(
            statistics.class_transitions.items(),
            key=lambda x: -x[1]
        ):
            if area > 10:
                from_name = self.class_names.get(from_c, f"Class {from_c}")
                to_name = self.class_names.get(to_c, f"Class {to_c}")
                change_type = self.categorize_change(from_c, to_c)
                report.append(f"  {from_name} → {to_name}: {area:,.2f} ha ({change_type.name})")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


class ChangeVectorAnalysis:
    """
    Change Vector Analysis (CVA) for continuous change detection.
    
    Analyzes spectral differences between time periods.
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Initialize CVA.
        
        Args:
            threshold: Magnitude threshold for change detection
        """
        self.threshold = threshold
    
    def compute_change_vector(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute change vector between two images.
        
        Args:
            image1: First image (H, W, C)
            image2: Second image (H, W, C)
            
        Returns:
            magnitude: Change magnitude map
            direction: Change direction map
        """
        # Compute difference
        diff = image2.astype(np.float32) - image1.astype(np.float32)
        
        # Magnitude (Euclidean distance in spectral space)
        magnitude = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # Direction (angle of dominant change)
        # Simplified: use ratio of first two bands
        direction = np.arctan2(diff[..., 1], diff[..., 0])
        
        return magnitude, direction
    
    def detect_changes(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> np.ndarray:
        """
        Detect changed pixels using CVA.
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Binary change map
        """
        magnitude, _ = self.compute_change_vector(image1, image2)
        
        # Normalize magnitude
        magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-6)
        
        # Apply threshold
        change_map = (magnitude_norm > self.threshold).astype(np.uint8)
        
        return change_map


class TimeSeriesAnalyzer:
    """
    Analyze land cover changes over multiple time periods.
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        class_names: Dict[int, str] = None
    ):
        self.num_classes = num_classes
        self.class_names = class_names or {
            0: 'Background', 1: 'Roads', 2: 'Water',
            3: 'Forest', 4: 'Plantation', 5: 'Buildings', 6: 'Bare Land'
        }
        self.pcc = PostClassificationComparison(num_classes, class_names)
    
    def analyze_series(
        self,
        maps: List[np.ndarray],
        years: List[int]
    ) -> pd.DataFrame:
        """
        Analyze time series of land cover maps.
        
        Args:
            maps: List of classification maps
            years: Corresponding years
            
        Returns:
            DataFrame with area statistics per year
        """
        data = []
        
        for i, (map_data, year) in enumerate(zip(maps, years)):
            row = {'year': year}
            
            for c in range(self.num_classes):
                class_name = self.class_names.get(c, f"class_{c}")
                pixel_count = np.sum(map_data == c)
                area_ha = pixel_count * 100 / 10000  # 10m resolution
                row[f'{class_name}_ha'] = area_ha
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def compute_trends(
        self,
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute linear trends for each class.
        
        Args:
            df: DataFrame from analyze_series
            
        Returns:
            Dictionary of trends (ha/year)
        """
        trends = {}
        years = df['year'].values
        
        for col in df.columns:
            if col != 'year' and col.endswith('_ha'):
                values = df[col].values
                # Linear regression
                slope = np.polyfit(years, values, 1)[0]
                trends[col.replace('_ha', '')] = slope
        
        return trends
    
    def find_change_hotspots(
        self,
        maps: List[np.ndarray],
        threshold: int = 2
    ) -> np.ndarray:
        """
        Find pixels that changed multiple times.
        
        Args:
            maps: List of classification maps
            threshold: Minimum number of changes
            
        Returns:
            Hotspot map (count of changes per pixel)
        """
        change_count = np.zeros_like(maps[0], dtype=np.int16)
        
        for i in range(1, len(maps)):
            change_count += (maps[i] != maps[i-1]).astype(np.int16)
        
        hotspots = (change_count >= threshold).astype(np.uint8)
        
        return hotspots, change_count


def export_change_report(
    statistics: ChangeStatistics,
    output_path: str,
    year_before: int,
    year_after: int,
    class_names: Dict[int, str] = None
):
    """
    Export change statistics to CSV.
    
    Args:
        statistics: Computed statistics
        output_path: Output file path
        year_before: Start year
        year_after: End year
        class_names: Class name mapping
    """
    class_names = class_names or {
        0: 'Background', 1: 'Roads', 2: 'Water',
        3: 'Forest', 4: 'Plantation', 5: 'Buildings', 6: 'Bare Land'
    }
    
    # Summary data
    summary = {
        'Metric': ['Total Area (ha)', 'Changed Area (ha)', 'Change Percentage'],
        'Value': [
            f"{statistics.total_area:,.2f}",
            f"{statistics.changed_area:,.2f}",
            f"{statistics.change_percentage:.2f}%"
        ]
    }
    
    # Net change data
    net_change = {
        'Class': [],
        'Gain (ha)': [],
        'Loss (ha)': [],
        'Net Change (ha)': []
    }
    
    for c in range(len(class_names)):
        net_change['Class'].append(class_names.get(c, f"Class {c}"))
        net_change['Gain (ha)'].append(f"{statistics.per_class_gain.get(c, 0):,.2f}")
        net_change['Loss (ha)'].append(f"{statistics.per_class_loss.get(c, 0):,.2f}")
        net_change['Net Change (ha)'].append(f"{statistics.net_change.get(c, 0):,.2f}")
    
    # Save to CSV
    output_path = Path(output_path)
    
    pd.DataFrame(summary).to_csv(
        output_path.with_suffix('.summary.csv'),
        index=False
    )
    
    pd.DataFrame(net_change).to_csv(
        output_path.with_suffix('.changes.csv'),
        index=False
    )
    
    print(f"Exported change report to {output_path}")


if __name__ == '__main__':
    # Test change detection
    print("Testing change detection utilities...")
    
    # Create sample data
    np.random.seed(42)
    h, w = 256, 256
    
    # Simulate land cover maps
    map_2015 = np.random.randint(0, 7, (h, w))
    map_2024 = map_2015.copy()
    
    # Simulate some changes
    # Urbanization
    map_2024[50:100, 50:100] = 5  # Buildings
    # Deforestation
    map_2024[150:200, 100:150] = 4  # Forest -> Plantation
    
    # Run change detection
    detector = PostClassificationComparison()
    
    binary_change, change_type = detector.detect_changes(map_2015, map_2024)
    print(f"Changed pixels: {binary_change.sum()} ({binary_change.mean()*100:.2f}%)")
    
    # Compute statistics
    stats = detector.compute_statistics(map_2015, map_2024)
    print(f"\nChange Statistics:")
    print(f"  Total area: {stats.total_area:,.2f} ha")
    print(f"  Changed area: {stats.changed_area:,.2f} ha")
    print(f"  Change percentage: {stats.change_percentage:.2f}%")
    
    # Generate report
    report = detector.generate_report(stats, 2015, 2024)
    print("\n" + report)
