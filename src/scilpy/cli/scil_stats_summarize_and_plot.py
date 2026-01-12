#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize statistics from a CSV file and create boxplot visualizations.

This script reads a CSV file with ROI-based metrics and generates:
1. A summary CSV with mean, median, and std for each metric grouped by ROI
2. Boxplot figures for each metric showing distribution across ROIs

Expected CSV format:
    sid, roi, Volume, Diameter_mean, Diameter_median, ...
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_csv',
                   help='Input CSV file with ROI-based metrics.')
    p.add_argument('out_summary',
                   help='Output summary CSV file.')
    p.add_argument('out_figures_dir',
                   help='Output directory for boxplot figures.')

    p.add_argument('--metrics', nargs='+',
                   help='List of metric column names to process. '
                        'If not provided, will auto-detect numeric columns '
                        '(excluding sid and roi).')
    p.add_argument('--figsize', nargs=2, type=float, default=[12, 6],
                   help='Figure size in inches (width height) [%(default)s].')
    p.add_argument('--dpi', type=int, default=150,
                   help='DPI for saved figures [%(default)s].')
    p.add_argument('--roi_order', nargs='+',
                   help='Specific order for ROIs in plots. If not provided, '
                        'will use alphabetical order.')
    p.add_argument('--skip_confidence', action='store_true',
                   help='Skip the Confidence column if present.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_csv)
    assert_outputs_exist(parser, args, args.out_summary)

    # Create output directory for figures
    os.makedirs(args.out_figures_dir, exist_ok=True)

    # Read CSV
    df = pd.read_csv(args.in_csv)

    # Identify metric columns
    if args.metrics:
        metric_cols = args.metrics
    else:
        # Auto-detect: numeric columns excluding 'sid' and 'roi'
        exclude_cols = ['sid', 'roi']
        if args.skip_confidence:
            exclude_cols.append('Confidence')
        
        metric_cols = [col for col in df.columns 
                      if col not in exclude_cols and 
                      pd.api.types.is_numeric_dtype(df[col])]

    # Convert metric columns to numeric (handle any non-numeric values)
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute summary statistics by ROI for each metric
    summary_data = []
    
    for roi in df['roi'].unique():
        roi_data = df[df['roi'] == roi]
        row = {'roi': roi}
        
        for metric in metric_cols:
            values = roi_data[metric].dropna()
            if len(values) > 0:
                row[f'{metric}_mean'] = values.mean()
                row[f'{metric}_median'] = values.median()
                row[f'{metric}_std'] = values.std()
                row[f'{metric}_count'] = len(values)
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_median'] = np.nan
                row[f'{metric}_std'] = np.nan
                row[f'{metric}_count'] = 0
        
        summary_data.append(row)

    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by ROI name
    summary_df = summary_df.sort_values('roi').reset_index(drop=True)
    
    # Save summary
    summary_df.to_csv(args.out_summary, index=False)

    # Determine ROI order for plots
    if args.roi_order:
        roi_order = [roi for roi in args.roi_order if roi in df['roi'].unique()]
    else:
        roi_order = sorted(df['roi'].unique())

    # Create boxplots for each metric
    sns.set_style("whitegrid")
    
    for metric in metric_cols:
        # Filter out NaN values
        plot_df = df[['roi', metric]].dropna()
        
        if len(plot_df) == 0:
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=tuple(args.figsize))
        
        # Create boxplot
        sns.boxplot(data=plot_df, x='roi', y=metric, order=roi_order, ax=ax)
        
        # Customize plot
        ax.set_xlabel('ROI', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' '), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ")} Distribution by ROI', 
                    fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_filename = os.path.join(args.out_figures_dir, 
                                      f'{metric}_boxplot.png')
        plt.savefig(output_filename, dpi=args.dpi, bbox_inches='tight')
        plt.close()

    print(f"Summary statistics saved to: {args.out_summary}")
    print(f"Boxplot figures saved to: {args.out_figures_dir}/")
    print(f"Processed {len(metric_cols)} metrics across {len(roi_order)} ROIs")


if __name__ == "__main__":
    main()
