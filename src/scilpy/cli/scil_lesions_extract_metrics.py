#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract statistics from multiple metric maps for each lesion in a label map.

This script takes a labeled lesion image (where each lesion has a unique
integer label) and a folder containing metric maps (e.g., FA, MD, T1, T2, etc.).
For each lesion and each metric map, it computes:
- Mean value
- Standard deviation
- Median value
- Volume in mm³
- Volume in number of voxels

The output is a CSV file where each row represents a lesion and each metric
has columns for its statistics.

Input requirements:
- Lesion label map: NIfTI file with unique integer labels for each lesion
- Metrics folder: Contains one or more NIfTI metric maps
- All images must be co-registered and have the same dimensions

Usage example:
    scil_lesions_extract_metrics.py lesions_labels.nii.gz \\
                                     metrics_folder/ \\
                                     output_stats.csv \\
                                     --metrics FA.nii.gz MD.nii.gz T1.nii.gz

    # Or extract all NIfTI files in the folder:
    scil_lesions_extract_metrics.py lesions_labels.nii.gz \\
                                     metrics_folder/ \\
                                     output_stats.csv

Output CSV columns:
- lesion_id: Unique lesion identifier
- volume_voxels: Number of voxels in the lesion
- volume_mm3: Volume in cubic millimeters
- <metric_name>_mean: Mean value for the metric
- <metric_name>_std: Standard deviation for the metric
- <metric_name>_median: Median value for the metric
- ... (repeated for each metric)

References:
    [1] Filippi, M., et al. "Assessment of lesions on magnetic resonance
    imaging in multiple sclerosis: practical guidelines." Brain 142.7
    (2019): 1858-1875.
"""

import argparse
import logging
import os
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd

from scilpy.image.labels import get_data_as_labels
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_headers_compatible,
                             add_verbose_arg)


def _build_arg_parser():
    """Build argument parser."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    p.add_argument('in_lesions',
                   help='Input lesion label file in NIfTI format.')
    p.add_argument('in_metrics_folder',
                   help='Folder containing metric maps in NIfTI format.')
    p.add_argument('out_csv',
                   help='Output CSV file with lesion statistics.')
    
    p.add_argument('--metrics', nargs='+',
                   help='Specific metric files to process (filenames only). '
                        'If not specified, all .nii.gz files in the folder '
                        'will be processed.')
    p.add_argument('--min_lesion_vol', type=float, default=0.0,
                   help='Minimum lesion volume in mm3 to include in analysis '
                        '[%(default)s].')
    p.add_argument('--ignore_zeros', action='store_true',
                   help='Ignore zero values when computing statistics '
                        '(useful for masked metrics).')
    p.add_argument('--add_sid', type=str, metavar='SID',
                   help='Add a subject ID column to the output CSV with '
                        'the specified value.')
    
    add_verbose_arg(p)
    add_overwrite_arg(p)
    
    return p


def get_metric_files(metrics_folder, specific_metrics=None):
    """
    Get list of metric files to process.
    
    Parameters
    ----------
    metrics_folder : str
        Path to folder containing metric maps.
    specific_metrics : list of str, optional
        Specific metric filenames to process.
    
    Returns
    -------
    metric_files : list of tuple
        List of (filepath, metric_name) tuples.
    """
    metric_files = []
    
    if specific_metrics:
        # Use specific metrics provided by user
        for metric in specific_metrics:
            filepath = os.path.join(metrics_folder, metric)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Metric file not found: {filepath}")
            # Extract metric name from filename (remove .nii.gz or .nii)
            metric_name = os.path.basename(metric)
            metric_name = metric_name.replace('.nii.gz', '').replace('.nii', '')
            metric_files.append((filepath, metric_name))
    else:
        # Find all NIfTI files in folder
        nii_files = glob(os.path.join(metrics_folder, '*.nii.gz'))
        nii_files += glob(os.path.join(metrics_folder, '*.nii'))
        
        if not nii_files:
            raise ValueError(f"No NIfTI files found in {metrics_folder}")
        
        for filepath in sorted(nii_files):
            metric_name = os.path.basename(filepath)
            metric_name = metric_name.replace('.nii.gz', '').replace('.nii', '')
            metric_files.append((filepath, metric_name))
    
    return metric_files


def compute_lesion_statistics(lesion_mask, metric_data, ignore_zeros=False):
    """
    Compute statistics for a lesion in a metric map.
    
    Parameters
    ----------
    lesion_mask : ndarray
        Binary mask of the lesion.
    metric_data : ndarray
        Metric map data.
    ignore_zeros : bool
        If True, ignore zero values in statistics computation.
    
    Returns
    -------
    stats : dict
        Dictionary with mean, std, and median values.
    """
    # Extract metric values for the lesion
    lesion_values = metric_data[lesion_mask]
    
    # Filter out zeros if requested
    if ignore_zeros:
        lesion_values = lesion_values[lesion_values != 0]
    
    # Handle empty lesions or all-zero lesions
    if len(lesion_values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan
        }
    
    # Compute statistics
    stats = {
        'mean': np.mean(lesion_values),
        'std': np.std(lesion_values),
        'median': np.median(lesion_values)
    }
    
    return stats


def main():
    """Main function."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    
    # Validate inputs
    assert_inputs_exist(parser, [args.in_lesions])
    assert_outputs_exist(parser, args, [args.out_csv])
    
    if not os.path.isdir(args.in_metrics_folder):
        parser.error(f"Metrics folder does not exist: {args.in_metrics_folder}")
    
    # Get metric files
    logging.info("Finding metric files...")
    metric_files = get_metric_files(args.in_metrics_folder, args.metrics)
    logging.info(f"Found {len(metric_files)} metric file(s) to process:")
    for filepath, metric_name in metric_files:
        logging.info(f"  - {metric_name}: {filepath}")
    
    # Load lesion labels
    logging.info("Loading lesion labels...")
    lesion_img = nib.load(args.in_lesions)
    lesion_data = get_data_as_labels(lesion_img)
    voxel_sizes = lesion_img.header.get_zooms()[0:3]
    voxel_volume_mm3 = np.prod(voxel_sizes)
    
    # Check header compatibility for all metric files
    metric_paths = [fp for fp, _ in metric_files]
    assert_headers_compatible(parser, [args.in_lesions] + metric_paths)
    
    # Load all metric maps
    logging.info("Loading metric maps...")
    metric_data_dict = {}
    for filepath, metric_name in metric_files:
        metric_img = nib.load(filepath)
        metric_data_dict[metric_name] = np.asarray(metric_img.dataobj)
        logging.debug(f"Loaded {metric_name}: shape {metric_data_dict[metric_name].shape}")
    
    # Get unique lesion labels (excluding background 0)
    lesion_ids = np.unique(lesion_data)
    lesion_ids = lesion_ids[lesion_ids != 0]
    
    logging.info(f"Found {len(lesion_ids)} lesions to analyze.")
    
    # Process each lesion
    results = []
    for lesion_id in lesion_ids:
        logging.debug(f"Processing lesion {lesion_id}...")
        
        # Extract lesion mask
        lesion_mask = (lesion_data == lesion_id)
        
        # Compute volume
        volume_voxels = np.sum(lesion_mask)
        volume_mm3 = volume_voxels * voxel_volume_mm3
        
        # Skip if below minimum volume threshold
        if volume_mm3 < args.min_lesion_vol:
            logging.debug(f"Skipping lesion {lesion_id}: "
                         f"volume {volume_mm3:.2f} mm³ < {args.min_lesion_vol} mm³")
            continue
        
        # Initialize result dictionary for this lesion
        result = {
            'lesion_id': int(lesion_id),
            'volume_voxels': int(volume_voxels),
            'volume_mm3': float(volume_mm3)
        }
        
        # Add subject ID if provided
        if args.add_sid:
            result['subject_id'] = args.add_sid
        
        # Compute statistics for each metric
        for metric_name, metric_data in metric_data_dict.items():
            stats = compute_lesion_statistics(
                lesion_mask, metric_data, ignore_zeros=args.ignore_zeros
            )
            
            # Add to result dictionary with metric name prefix
            result[f'{metric_name}_mean'] = stats['mean']
            result[f'{metric_name}_std'] = stats['std']
            result[f'{metric_name}_median'] = stats['median']
        
        results.append(result)
        logging.info(f"Lesion {lesion_id}: {volume_mm3:.2f} mm³ "
                    f"({volume_voxels} voxels)")
    
    # Create DataFrame
    logging.info(f"Creating output CSV with {len(results)} lesions...")
    df = pd.DataFrame(results)
    
    # Reorder columns: subject_id (if present), lesion_id, volume info, then metrics alphabetically
    if args.add_sid:
        volume_cols = ['subject_id', 'lesion_id', 'volume_voxels', 'volume_mm3']
    else:
        volume_cols = ['lesion_id', 'volume_voxels', 'volume_mm3']
    metric_cols = [col for col in df.columns if col not in volume_cols]
    metric_cols.sort()
    df = df[volume_cols + metric_cols]
    
    # Save to CSV
    df.to_csv(args.out_csv, index=False, float_format='%.6f')
    logging.info(f"Saved results to {args.out_csv}")
    
    # Print summary statistics
    logging.info("\n" + "="*60)
    logging.info("Summary Statistics:")
    logging.info("="*60)
    logging.info(f"Total lesions processed: {len(results)}")
    logging.info(f"Total lesion volume: {df['volume_mm3'].sum():.2f} mm³")
    logging.info(f"Mean lesion volume: {df['volume_mm3'].mean():.2f} mm³")
    logging.info(f"Median lesion volume: {df['volume_mm3'].median():.2f} mm³")
    logging.info(f"Min lesion volume: {df['volume_mm3'].min():.2f} mm³")
    logging.info(f"Max lesion volume: {df['volume_mm3'].max():.2f} mm³")
    logging.info("="*60)
    
    # Print metric-wise summary
    for metric_name in metric_data_dict.keys():
        mean_col = f'{metric_name}_mean'
        if mean_col in df.columns:
            logging.info(f"\n{metric_name}:")
            logging.info(f"  Mean across lesions: {df[mean_col].mean():.6f}")
            logging.info(f"  Std across lesions: {df[mean_col].std():.6f}")
            logging.info(f"  Min: {df[mean_col].min():.6f}")
            logging.info(f"  Max: {df[mean_col].max():.6f}")
    
    logging.info("\nExtraction complete!")


if __name__ == "__main__":
    main()
