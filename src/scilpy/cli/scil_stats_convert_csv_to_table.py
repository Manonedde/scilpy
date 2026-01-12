#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert multiple CSV summary files into a single consolidated table.

This script takes multiple CSV files (typically with ROIs as columns and 
metrics as rows) and converts them into a single long-format table with
columns for subject ID, ROI names, and metric values.

Input CSV format (wide format):
    - First column: metric names (e.g., "Volume", "Diameter,mean")
    - Other columns: ROI names as headers (e.g., "LCAR", "RCAR", "BAS")
    - Values in cells

Output CSV format (long format):
    - sid: subject/session identifier
    - roi: ROI name
    - Volume, Diameter_mean, Diameter_median, etc.: metric columns

This is useful for statistical analysis and data visualization where long-format
data is preferred.
"""

import argparse
import os
import glob
import re

import pandas as pd

from scilpy.io.utils import (add_overwrite_arg,
                             assert_outputs_exist)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_csv_dir',
                   help='Directory containing input CSV files.')
    p.add_argument('out_csv',
                   help='Output consolidated CSV file.')

    p.add_argument('--pattern', default='*_SUMMARY.csv',
                   help='Glob pattern to match input CSV files '
                        '[%(default)s].')
    p.add_argument('--sid_pattern', 
                   default=r'(sub-\d+_ses-[\d-]+)',
                   help='Regex pattern to extract subject/session ID from '
                        'filename [%(default)s].')
    p.add_argument('--skip_empty_rois', action='store_true',
                   help='Skip ROIs with empty/missing values.')
    p.add_argument('--metric_separator', default=',',
                   help='Separator used in metric names (e.g., "Diameter,mean") '
                        '[%(default)s].')
    p.add_argument('--output_separator', default='_',
                   help='Separator to use in output column names '
                        '(e.g., "Diameter_mean") [%(default)s].')

    add_overwrite_arg(p)

    return p


def extract_sid_from_filename(filename, pattern):
    """
    Extract subject/session ID from filename using regex pattern.
    
    Parameters
    ----------
    filename : str
        Input filename
    pattern : str
        Regex pattern to extract ID
        
    Returns
    -------
    str
        Extracted subject/session ID, or filename if pattern doesn't match
    """
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    else:
        # Fallback: use basename without extension
        return os.path.splitext(os.path.basename(filename))[0]


def clean_metric_name(metric_name, separator='_'):
    """
    Clean and format metric names for column headers.
    
    Parameters
    ----------
    metric_name : str
        Original metric name (e.g., "Diameter,mean")
    separator : str
        Separator to use in output (e.g., "_")
        
    Returns
    -------
    str
        Cleaned metric name (e.g., "Diameter_mean")
    """
    # Replace common separators with the desired one
    cleaned = metric_name.replace(',', separator)
    cleaned = cleaned.replace(' ', separator)
    cleaned = cleaned.replace('__', separator)
    # Remove leading/trailing separators
    cleaned = cleaned.strip(separator)
    return cleaned


def read_wide_csv(csv_file, metric_sep=','):
    """
    Read a wide-format CSV and parse it.
    
    Parameters
    ----------
    csv_file : str
        Path to CSV file
    metric_sep : str
        Separator used in metric names
        
    Returns
    -------
    pd.DataFrame
        Parsed dataframe with cleaned structure
    """
    # Read without header to get true structure
    df_raw = pd.read_csv(csv_file, header=None)
    
    # Header row: column 0 is empty, ROI names start from column 1
    roi_names = df_raw.iloc[0, 1:].dropna().tolist()
    
    # Data rows start from row 1
    df_data = df_raw.iloc[1:, :].reset_index(drop=True)
    
    # Process each row based on its structure:
    # - Volume/Confidence: col0=metric, col1+=ROI values
    # - Diameter: col0="Diameter", col1=subcategory, col2+=ROI values (shifted!)
    
    result_rows = []
    for idx in range(len(df_data)):
        metric_main = str(df_data.iloc[idx, 0]).strip()
        potential_sub = str(df_data.iloc[idx, 1]).strip()
        
        # Diameter rows have subcategories in column 1
        if metric_main == 'Diameter' and potential_sub in ['mean', 'median', 'max', 'min', 'STD', 'N']:
            # Diameter row: subcategory in col1, ROI values start at col2
            metric_name = f"{metric_main}{metric_sep}{potential_sub}"
            roi_values = df_data.iloc[idx, 2:2+len(roi_names)].tolist()
        else:
            # Volume/Confidence row: ROI values start at col1
            metric_name = metric_main
            roi_values = df_data.iloc[idx, 1:1+len(roi_names)].tolist()
        
        result_rows.append([metric_name] + roi_values)
    
    # Create dataframe
    result_df = pd.DataFrame(result_rows, columns=['metric'] + roi_names)
    result_df = result_df.set_index('metric')
    
    return result_df


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Find all CSV files
    search_pattern = os.path.join(args.in_csv_dir, args.pattern)
    csv_files = sorted(glob.glob(search_pattern))
    
    if not csv_files:
        parser.error(f"No CSV files found matching pattern: {search_pattern}")
    
    assert_outputs_exist(parser, args, args.out_csv)

    # Process each CSV file
    all_data = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        sid = extract_sid_from_filename(filename, args.sid_pattern)
        
        try:
            # Read the wide-format CSV
            df_wide = read_wide_csv(csv_file, args.metric_separator)
            
            # Transpose so ROIs become rows and metrics become columns
            df_transposed = df_wide.T
            
            # Reset index to make ROI a column
            df_transposed = df_transposed.reset_index()
            df_transposed = df_transposed.rename(columns={'index': 'roi'})
            
            # Clean column names (metric names)
            df_transposed.columns = [
                clean_metric_name(col, args.output_separator) 
                if col != 'roi' else col 
                for col in df_transposed.columns
            ]
            
            # Add subject/session ID
            df_transposed.insert(0, 'sid', sid)
            
            # Skip empty ROIs if requested
            if args.skip_empty_rois:
                # Remove rows where all metric values are empty/NaN
                metric_cols = [c for c in df_transposed.columns 
                              if c not in ['sid', 'roi']]
                df_transposed = df_transposed.dropna(subset=metric_cols, how='all')
            
            all_data.append(df_transposed)
            
        except Exception as e:
            print(f"Warning: Error processing {filename}: {e}")
            continue
    
    if not all_data:
        parser.error("No data was successfully processed from CSV files.")
    
    # Concatenate all dataframes
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by sid and roi for better readability
    final_df = final_df.sort_values(['sid', 'roi']).reset_index(drop=True)
    
    # Save to output CSV
    final_df.to_csv(args.out_csv, index=False)
    
    
    # Print summary
    n_subjects = final_df['sid'].nunique()
    n_rois = final_df['roi'].nunique()
    metric_cols = [c for c in final_df.columns if c not in ['sid', 'roi']]
    
    print("\nSummary:"
          f"\n  Number of subjects/sessions: {n_subjects}"
          f"\n  Number of ROIs: {n_rois}"
          f"\n  Metrics included: {', '.join(metric_cols)}")


if __name__ == "__main__":
    main()
