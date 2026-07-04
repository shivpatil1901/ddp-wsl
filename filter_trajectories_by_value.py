#!/usr/bin/env python
"""
Utility script for practical applications of the value function:
- Filtering trajectories by quality/safety
- Clustering trajectories
- Creating new datasets based on value function classification
"""

import numpy as np
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

safedice_path = Path(__file__).parent / 'SafeDICE'
sys.path.insert(0, str(safedice_path))

from analyze_value_function import ValueFunctionAnalyzer


class TrajectoryFilter:
    """Utilities for filtering and clustering trajectories based on value function."""
    
    def __init__(self, analyzer: ValueFunctionAnalyzer):
        """
        Initialize filter with a value function analyzer.
        
        Args:
            analyzer: ValueFunctionAnalyzer instance
        """
        self.analyzer = analyzer
    
    def filter_trajectories_by_value(
        self,
        trajectories: List[Dict],
        min_value: float = None,
        max_value: float = None,
        min_percentile: float = None,
        max_percentile: float = None,
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Filter trajectories based on average value function.
        
        Args:
            trajectories: List of trajectory dicts with 'observations'
            min_value: Minimum absolute value threshold
            max_value: Maximum absolute value threshold
            min_percentile: Minimum percentile (0-100)
            max_percentile: Maximum percentile (0-100)
            
        Returns:
            (filtered_trajectories, scores) - trajectories that pass filter and their scores
        """
        scores = []
        
        for traj in trajectories:
            states = traj['observations']
            values = self.analyzer.get_state_values(states)
            avg_value = np.mean(values)
            scores.append(avg_value)
        
        scores = np.array(scores)
        mask = np.ones(len(scores), dtype=bool)
        
        if min_value is not None:
            mask &= scores >= min_value
        
        if max_value is not None:
            mask &= scores <= max_value
        
        if min_percentile is not None:
            threshold = np.percentile(scores, min_percentile)
            mask &= scores >= threshold
        
        if max_percentile is not None:
            threshold = np.percentile(scores, max_percentile)
            mask &= scores <= threshold
        
        filtered_trajs = [t for i, t in enumerate(trajectories) if mask[i]]
        filtered_scores = scores[mask]
        
        return filtered_trajs, filtered_scores
    
    def cluster_trajectories(
        self,
        trajectories: List[Dict],
        num_clusters: int = 3,
    ) -> Dict:
        """
        Cluster trajectories based on value function using percentiles.
        
        Args:
            trajectories: List of trajectory dicts
            num_clusters: Number of clusters
            
        Returns:
            Dictionary mapping cluster_id to list of trajectories and statistics
        """
        scores = []
        
        for traj in trajectories:
            states = traj['observations']
            values = self.analyzer.get_state_values(states)
            avg_value = np.mean(values)
            scores.append(avg_value)
        
        scores = np.array(scores)
        
        # Cluster by percentiles
        percentiles = np.linspace(0, 100, num_clusters + 1)
        clusters = {}
        
        for cluster_id in range(num_clusters):
            lower = percentiles[cluster_id]
            upper = percentiles[cluster_id + 1]
            
            mask = (scores >= np.percentile(scores, lower)) & \
                   (scores <= np.percentile(scores, upper))
            
            cluster_trajs = [t for i, t in enumerate(trajectories) if mask[i]]
            cluster_scores = scores[mask]
            
            clusters[cluster_id] = {
                'trajectories': cluster_trajs,
                'scores': cluster_scores,
                'percentile_range': (lower, upper),
                'num_trajectories': len(cluster_trajs),
                'mean_score': np.mean(cluster_scores) if len(cluster_scores) > 0 else 0,
                'std_score': np.std(cluster_scores) if len(cluster_scores) > 0 else 0,
            }
        
        return clusters
    
    def select_high_quality_trajectories(
        self,
        trajectories: List[Dict],
        top_percent: float = 25.0,
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Select top-quality trajectories by value function.
        
        Args:
            trajectories: List of trajectory dicts
            top_percent: Percentage to select (e.g., 25 = top 25%)
            
        Returns:
            (selected_trajectories, scores)
        """
        percentile = 100 - top_percent
        return self.filter_trajectories_by_value(
            trajectories,
            min_percentile=percentile
        )
    
    def select_diverse_trajectories(
        self,
        trajectories: List[Dict],
        num_select: int,
    ) -> List[Dict]:
        """
        Select diverse trajectories spanning the value distribution.
        
        Uses stratified sampling to get representation across all value levels.
        
        Args:
            trajectories: List of trajectory dicts
            num_select: Number to select
            
        Returns:
            Selected trajectories
        """
        scores = []
        
        for traj in trajectories:
            states = traj['observations']
            values = self.analyzer.get_state_values(states)
            avg_value = np.mean(values)
            scores.append(avg_value)
        
        scores = np.array(scores)
        
        # Stratified sampling
        indices = np.argsort(scores)
        strata_size = len(trajectories) // num_select
        
        selected_indices = []
        for i in range(num_select):
            start = i * strata_size
            end = (i + 1) * strata_size if i < num_select - 1 else len(trajectories)
            
            if start < end:
                selected_idx = indices[np.random.randint(start, end)]
                selected_indices.append(selected_idx)
        
        return [trajectories[i] for i in selected_indices]
    
    def create_filtered_dataset(
        self,
        input_dataset_path: str,
        output_dataset_path: str,
        filter_type: str = 'high_quality',
        **filter_kwargs
    ) -> Dict:
        """
        Create a new dataset file with filtered trajectories.
        
        Args:
            input_dataset_path: Path to input pickle file
            output_dataset_path: Path to save filtered dataset
            filter_type: 'high_quality', 'low_quality', 'cluster', or 'diverse'
            **filter_kwargs: Additional arguments for filter_* methods
            
        Returns:
            Dictionary with filtering statistics
        """
        try:
            import pickle5 as pickle_lib
        except ImportError:
            import pickle as pickle_lib
        
        print(f"Loading dataset from {input_dataset_path}...")
        
        with open(input_dataset_path, 'rb') as f:
            data = pickle_lib.load(f)
        
        # Parse trajectories
        if isinstance(data, dict) and 'observations' in data:
            trajectories = self.analyzer._split_trajectories_from_arrays(data)
        else:
            trajectories = data if isinstance(data, list) else [data]
        
        print(f"Loaded {len(trajectories)} trajectories")
        
        # Apply filter
        if filter_type == 'high_quality':
            filtered_trajs, scores = self.select_high_quality_trajectories(
                trajectories,
                top_percent=filter_kwargs.get('top_percent', 25)
            )
            metadata = {
                'filter_type': 'high_quality',
                'top_percent': filter_kwargs.get('top_percent', 25),
                'num_original': len(trajectories),
                'num_filtered': len(filtered_trajs),
                'score_range': (float(np.min(scores)), float(np.max(scores))),
            }
        
        elif filter_type == 'low_quality':
            bottom_percent = filter_kwargs.get('bottom_percent', 25)
            filtered_trajs, scores = self.filter_trajectories_by_value(
                trajectories,
                max_percentile=100 - bottom_percent
            )
            metadata = {
                'filter_type': 'low_quality',
                'bottom_percent': bottom_percent,
                'num_original': len(trajectories),
                'num_filtered': len(filtered_trajs),
                'score_range': (float(np.min(scores)), float(np.max(scores))),
            }
        
        elif filter_type == 'cluster':
            clusters = self.cluster_trajectories(
                trajectories,
                num_clusters=filter_kwargs.get('num_clusters', 3)
            )
            filtered_trajs = []
            for cluster_id, cluster_data in clusters.items():
                filtered_trajs.extend(cluster_data['trajectories'])
            
            metadata = {
                'filter_type': 'cluster',
                'num_clusters': filter_kwargs.get('num_clusters', 3),
                'num_original': len(trajectories),
                'num_filtered': len(filtered_trajs),
                'cluster_sizes': {k: v['num_trajectories'] for k, v in clusters.items()},
            }
        
        elif filter_type == 'diverse':
            filtered_trajs = self.select_diverse_trajectories(
                trajectories,
                num_select=filter_kwargs.get('num_select', 100)
            )
            metadata = {
                'filter_type': 'diverse',
                'num_select': filter_kwargs.get('num_select', 100),
                'num_original': len(trajectories),
                'num_filtered': len(filtered_trajs),
            }
        
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Save filtered dataset
        # Convert back to array format if needed
        filtered_data = {
            'trajectories': filtered_trajs,
            'metadata': metadata,
        }
        
        print(f"Saving filtered dataset to {output_dataset_path}...")
        with open(output_dataset_path, 'wb') as f:
            pickle_lib.dump(filtered_data, f, protocol=pickle_lib.HIGHEST_PROTOCOL)
        
        print(f"✅ Saved {len(filtered_trajs)} trajectories")
        
        return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Filter and cluster trajectories using value function'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle',
        help='Path to weights file'
    )
    parser.add_argument(
        '--input-dataset',
        type=str,
        default='SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle',
        help='Input dataset path'
    )
    parser.add_argument(
        '--output-dataset',
        type=str,
        help='Output dataset path (required to save filtered data)'
    )
    parser.add_argument(
        '--filter-type',
        type=str,
        choices=['high_quality', 'low_quality', 'cluster', 'diverse', 'analyze'],
        default='analyze',
        help='Type of filtering to apply'
    )
    parser.add_argument(
        '--top-percent',
        type=float,
        default=25,
        help='Top percent for high_quality filter'
    )
    parser.add_argument(
        '--bottom-percent',
        type=float,
        default=25,
        help='Bottom percent for low_quality filter'
    )
    parser.add_argument(
        '--num-clusters',
        type=int,
        default=3,
        help='Number of clusters for cluster filter'
    )
    parser.add_argument(
        '--num-select',
        type=int,
        default=100,
        help='Number to select for diverse filter'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer and filter
    print("=" * 80)
    print("SafeDICE Trajectory Filter Tool")
    print("=" * 80)
    
    analyzer = ValueFunctionAnalyzer(args.weights)
    filter_util = TrajectoryFilter(analyzer)
    
    # Load trajectories
    try:
        import pickle5 as pickle_lib
    except ImportError:
        import pickle as pickle_lib
    
    with open(args.input_dataset, 'rb') as f:
        data = pickle_lib.load(f)
    
    if isinstance(data, dict) and 'observations' in data:
        trajectories = analyzer._split_trajectories_from_arrays(data)
    else:
        trajectories = data if isinstance(data, list) else [data]
    
    print(f"Loaded {len(trajectories)} trajectories\n")
    
    # Apply filter
    if args.filter_type == 'analyze':
        print("Analyzing trajectory quality distribution...\n")
        clusters = filter_util.cluster_trajectories(trajectories, num_clusters=5)
        
        for cluster_id, cluster_data in clusters.items():
            print(f"Cluster {cluster_id}:")
            print(f"  Trajectories: {cluster_data['num_trajectories']}")
            print(f"  Value range: {cluster_data['percentile_range']}")
            print(f"  Mean value: {cluster_data['mean_score']:.4f}")
            print(f"  Std value: {cluster_data['std_score']:.4f}\n")
    
    elif args.output_dataset:
        print(f"Applying {args.filter_type} filter...\n")
        
        filter_kwargs = {
            'top_percent': args.top_percent,
            'bottom_percent': args.bottom_percent,
            'num_clusters': args.num_clusters,
            'num_select': args.num_select,
        }
        
        metadata = filter_util.create_filtered_dataset(
            args.input_dataset,
            args.output_dataset,
            filter_type=args.filter_type,
            **filter_kwargs
        )
        
        print("\nFiltering Statistics:")
        for key, val in metadata.items():
            print(f"  {key}: {val}")
    
    else:
        print("❌ Error: --output-dataset required to save filtered data")
        print("Use --filter-type analyze for distribution analysis without saving")


if __name__ == '__main__':
    main()
