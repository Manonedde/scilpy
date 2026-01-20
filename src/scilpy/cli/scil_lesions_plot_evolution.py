#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the evolution of lesion volumes across sessions.

This script takes multiple harmonized lesion label maps (from
scil_lesions_harmonize_labels.py) and plots the volume evolution of each
lesion across sessions. It handles both positive lesion labels and negative
pre-lesion labels.

The script will:
1. Count voxels for each lesion ID (positive and negative) in each session
2. Plot volume evolution over time for each lesion
3. Distinguish between pre-lesion (negative IDs), baseline lesions (present in first session),
   new lesions (appearing later), and confluent lesions (merged from multiple sources)
4. Optionally detect confluent lesions using backwards-mapping

Confluent Lesions:
When --detect_confluent is enabled, the script identifies lesions that result
from the merger of multiple distinct lesions from the first session. This uses
backwards-mapping: for each lesion in subsequent sessions, we check if its 
footprint overlaps with multiple lesions from the first session. Confluent 
lesions are treated as single entities throughout the analysis to maintain 
clean statistics.

Example usage:
    scil_lesions_plot_evolution.py session1.nii.gz session2.nii.gz \\
        session3.nii.gz output_plot.png --detect_confluent
"""

import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_headers_compatible)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, 
        formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument('in_images', nargs='+',
                   help='Input lesion label maps in chronological order, '
                        'in NIfTI format.')
    p.add_argument('out_plot',
                   help='Output plot base filename (without extension). '
                        'Will generate both .html and .png files.')
    p.add_argument('--out_csv',
                   help='Output CSV filename for lesion volume data. '
                        'If not provided, no CSV will be saved.')
    p.add_argument('--out_trajectory_plot',
                   help='Output filename for lesion trajectory classification plot '
                        '(growth/shrinkage/mixed). Shows all lesions categorized by '
                        'their volume change pattern across sessions.')
    p.add_argument('--trajectory_threshold', type=float, default=3.0,
                   help='Absolute delta threshold (in mm³) for trajectory classification. '
                        'Only changes with |Δ| > threshold are considered for '
                        'classifying lesions as growth/shrinkage/mixed [%(default)s].')
    p.add_argument('--delta_method', default='baseline',
                   choices=['baseline', 'session'],
                   help='Method for calculating delta (volume change): '
                        'baseline = compare to first appearance, '
                        'session = compare to previous session [%(default)s].')
    
    p.add_argument('--session_names', nargs='+',
                   help='Custom session names for x-axis. If not provided, '
                        'will use Session 1, Session 2, etc.')
    p.add_argument('--voxel_size', type=float, nargs=3,
                   help='Voxel size in mm (x, y, z) to convert voxel counts '
                        'to volume in mm³. If not provided, uses voxel size '
                        'from first image header.')
    p.add_argument('--top_n_lesions', type=int,
                   help='Only plot the top N lesions by maximum volume. '
                        'Default: plot all lesions.')
    p.add_argument('--min_volume', type=float, default=0,
                   help='Minimum volume (in voxels or mm³) for a lesion to '
                        'be included in the plot [%(default)s].')
    p.add_argument('--width', type=int, default=1400,
                   help='Plot width in pixels for main lesion evolution plot [%(default)s].')
    p.add_argument('--height', type=int, default=600,
                   help='Plot height in pixels for main lesion evolution plot [%(default)s].')
    p.add_argument('--trajectory_width', type=int, default=1200,
                   help='Plot width in pixels for trajectory classification plot [%(default)s].')
    p.add_argument('--trajectory_height', type=int, default=1400,
                   help='Plot height in pixels for trajectory classification plot [%(default)s].')
    p.add_argument('--colormap', default='nipy_spectral',
                   choices=['nipy_spectral', 'jet', 'turbo', 'tab20', 'hsv', 'rainbow'],
                   help='Colormap for lesion colors [%(default)s].')
    p.add_argument('--separate_pre_lesions', action='store_true',
                   help='Create separate plots for pre-lesions and baseline/new lesions.')
    p.add_argument('--growth_threshold', type=float, default=3.0,
                   help='Volume change threshold (in mm³) to consider a lesion '
                        'as changing (growing or shrinking) compared to its baseline. '
                        'Lesions with |Δ| > threshold will be included [%(default)s].')
    p.add_argument('--highlight_changes', action='store_true',
                   help='In the changing lesions plot, display all lines in grey '
                        'and highlight only the points where |Δ| > threshold.')
    p.add_argument('--detect_confluent', action='store_true',
                   help='Detect confluent lesions using backwards-mapping. '
                        'A lesion is classified as confluent if it overlaps with '
                        'multiple distinct lesions from the first session. Confluent '
                        'lesions are treated as a single entity for analysis.')
    p.add_argument('--min_confluence_overlap', type=int, default=1,
                   help='Minimum number of overlapping voxels to consider a confluence '
                        '[%(default)s].')
    
    add_overwrite_arg(p)
    return p


def count_lesion_volumes(images, voxel_volume=1.0, filenames=None, confluent_lesions=None):
    """
    Count voxel volumes for each lesion ID across all sessions.
    
    Parameters
    ----------
    images : list of nibabel.Nifti1Image
        List of lesion label images.
    voxel_volume : float
        Volume of a single voxel in mm³.
    filenames : list of str, optional
        List of filenames corresponding to images.
    confluent_lesions : dict, optional
        Dictionary mapping confluent lesion IDs to their source lesion IDs.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns: Session, Lesion_ID, Volume, Status
        Status is 'Pre-lesion' for negative IDs, 'Confluent' for merged lesions, 'Baseline' for lesions present in first session, 'New' for lesions appearing later.
    """
    if confluent_lesions is None:
        confluent_lesions = {}
    
    data_records = []
    
    # First pass: identify which lesions are present in the first session (baseline lesions)
    first_session_data = np.asanyarray(images[0].dataobj)
    baseline_lesions = set(np.unique(first_session_data))
    baseline_lesions.discard(0)  # Remove background
    baseline_lesions = {abs(int(label)) for label in baseline_lesions}  # Use absolute IDs
    
    for session_idx, img in enumerate(images):
        data = np.asanyarray(img.dataobj)
        
        # Extract subject and session info from filename if available
        subject_id = 'Unknown'
        session_name = f'Session {session_idx + 1}'
        
        if filenames and session_idx < len(filenames):
            filename = os.path.basename(filenames[session_idx])
            # Try to extract subject ID (e.g., sub-002-ms)
            if 'sub-' in filename:
                subject_id = filename.split('_')[0]
            # Try to extract session name (e.g., ses-1, ses-2)
            if 'ses-' in filename:
                import re
                match = re.search(r'ses-(\d+)', filename)
                if match:
                    session_name = f'ses-{match.group(1)}'
        
        # Get all unique labels (including negative ones)
        unique_labels = np.unique(data)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background
        
        for label_id in unique_labels:
            voxel_count = np.sum(data == label_id)
            volume = voxel_count * voxel_volume
            
            # Determine status based on sign, confluent detection, and first appearance
            if label_id < 0:
                status = 'Pre-lesion'
                abs_id = abs(label_id)
            else:
                abs_id = label_id
                # Check if this is a confluent lesion
                if abs_id in confluent_lesions:
                    status = 'Confluent'
                # Check if lesion was present in baseline (first session)
                elif abs_id in baseline_lesions:
                    status = 'Baseline'
                else:
                    # Lesion appears after baseline, so it's new
                    status = 'New'
            
            data_records.append({
                'Subject': subject_id,
                'Session': session_idx + 1,
                'Session_Name': session_name,
                'Lesion_ID': int(label_id),
                'Absolute_ID': int(abs_id),
                'Volume': volume,
                'Status': status,
                'Merged_From': str(confluent_lesions.get(abs_id, [])) if abs_id in confluent_lesions else ''
            })
    
    return pd.DataFrame(data_records)


def detect_confluent_lesions(images, min_overlap=1):
    """
    Detect confluent lesions using backwards-mapping.
    
    A lesion is considered confluent if it's a single blob in a later session
    that overlaps with multiple distinct lesions from an earlier session.
    Uses backwards-mapping: for each lesion in session N, check if its footprint
    overlaps with multiple lesions in session 1.
    
    Parameters
    ----------
    images : list of nibabel.Nifti1Image
        List of lesion label images in chronological order.
    min_overlap : int
        Minimum number of overlapping voxels to consider a merge.
    
    Returns
    -------
    confluent_lesions : dict
        Dictionary mapping lesion IDs to the list of original lesion IDs they merged from.
        Format: {lesion_id: [source_id1, source_id2, ...]}
    """
    if len(images) < 2:
        return {}
    
    confluent_lesions = {}
    
    # Get data arrays
    first_session = np.asanyarray(images[0].dataobj)
    
    # For each subsequent session, check for merges
    for session_idx in range(1, len(images)):
        current_session = np.asanyarray(images[session_idx].dataobj)
        
        # Get all positive lesion IDs in current session (excluding pre-lesions)
        current_labels = np.unique(current_session)
        current_labels = current_labels[(current_labels > 0)]
        
        # For each lesion in current session
        for lesion_id in current_labels:
            # Get the footprint of this lesion
            lesion_mask = (current_session == lesion_id)
            
            # Check which lesions from first session overlap with this footprint
            overlapping_first_session = first_session[lesion_mask]
            overlapping_labels = np.unique(overlapping_first_session)
            overlapping_labels = overlapping_labels[overlapping_labels > 0]
            
            # Count voxels for each overlapping label
            overlap_counts = {}
            for overlap_label in overlapping_labels:
                count = np.sum(overlapping_first_session == overlap_label)
                if count >= min_overlap:
                    overlap_counts[int(overlap_label)] = count
            
            # If this lesion overlaps with multiple lesions from first session, it's confluent
            if len(overlap_counts) > 1:
                confluent_lesions[int(lesion_id)] = sorted(overlap_counts.keys())
    
    return confluent_lesions


def rgba_to_plotly(rgba):
    """Convert matplotlib RGBA to plotly RGB string."""
    return f'rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})'


def plot_lesion_evolution(df, session_names, width, height, 
                          separate_pre_lesions=False, volume_unit='voxels',
                          growth_threshold=3.0, highlight_changes=False,
                          colormap='nipy_spectral', color_map=None):
    """
    Plot lesion volume evolution across sessions using Plotly.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with lesion volume data.
    session_names : list of str
        Names for each session.
    width : int
        Plot width in pixels.
    height : int
        Plot height in pixels.
    separate_pre_lesions : bool
        If True, create separate subplots for pre-lesions and baseline/new lesions.
    volume_unit : str
        Unit for volume ('voxels' or 'mm³').
    growth_threshold : float
        Volume change threshold to identify growing lesions.
    highlight_changes : bool
        If True, in changing lesions plot, show all lines in grey and 
        highlight only points with |Δ| > threshold.
    colormap : str
        Name of matplotlib colormap to use for lesion colors.
    color_map : dict, optional
        Pre-computed color mapping. If None, will be generated.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The created figure.
    """
    
    # Identify growing/shrinking lesions (all lesions with |change| > threshold)
    changing_lesions = set()
    for abs_id in df[df['Status'].isin(['Baseline', 'New', 'Confluent'])]['Absolute_ID'].unique():
        lesion_data = df[(df['Absolute_ID'] == abs_id) & (df['Status'].isin(['Baseline', 'New', 'Confluent']))].sort_values('Session')
        if len(lesion_data) > 1:
            baseline_volume = lesion_data.iloc[0]['Volume']
            # Check for maximum absolute change (can be positive or negative)
            max_abs_change = (lesion_data['Volume'] - baseline_volume).abs().max()
            if max_abs_change > growth_threshold:
                changing_lesions.add(abs_id)
    
    # Generate color map if not provided
    if color_map is None:
        all_lesion_ids = sorted(df['Absolute_ID'].unique())
        n_lesions = len(all_lesion_ids)
        
        # Use continuous colormap for any number of lesions
        cmap = plt.cm.get_cmap(colormap)
        colors = cmap(np.linspace(0, 0.95, n_lesions))  # 0.95 to avoid extreme end
        color_map = {lesion_id: rgba_to_plotly(colors[i]) for i, lesion_id in enumerate(all_lesion_ids)}
    
    # Create figure with 3 subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('All Lesions', 
                       f'Changing Lesions (|Δ| > {growth_threshold} {volume_unit})',
                       'New Lesions Evolution'),
        horizontal_spacing=0.15, vertical_spacing=0.14
    )
    
    # Subplot 1: All lesions (Baseline, New, and Confluent)
    df_active = df[df['Status'].isin(['Baseline', 'New', 'Confluent'])]
    if not df_active.empty:
        for lesion_id in sorted(df_active['Absolute_ID'].unique()):
            lesion_data = df_active[df_active['Absolute_ID'] == lesion_id].sort_values('Session')
            fig.add_trace(
                go.Scatter(
                    x=lesion_data['Session'],
                    y=lesion_data['Volume'],
                    mode='lines+markers',
                    name=f'Lesion {lesion_id}',
                    line=dict(color=color_map[lesion_id], width=2),
                    marker=dict(size=6),
                    legendgroup='all',
                    showlegend=True,
                    hovertemplate='Lesion: %{text}<br>Session: %{x}<br>Volume: %{y:.2f}<extra></extra>',
                    text=[str(lesion_id)] * len(lesion_data)
                ),
                row=1, col=1
            )
    
    # Subplot 2: Changing lesions (growth > threshold)
    if changing_lesions:
        df_changing = df_active[df_active['Absolute_ID'].isin(changing_lesions)]
        
        if highlight_changes:
            # Plot all lesions in grey first
            for lesion_id in sorted(changing_lesions):
                lesion_data = df_changing[df_changing['Absolute_ID'] == lesion_id].sort_values('Session')
                fig.add_trace(
                    go.Scatter(
                        x=lesion_data['Session'],
                        y=lesion_data['Volume'],
                        mode='lines+markers',
                        name=f'Lesion {lesion_id} baseline',
                        line=dict(color='lightgrey', width=2),
                        marker=dict(size=6),
                        showlegend=False,
                        legendgroup='changing',
                        hovertemplate='Lesion: %{text}<br>Session: %{x}<br>Volume: %{y:.2f}<extra></extra>',
                        text=[str(lesion_id)] * len(lesion_data)
                    ),
                    row=1, col=2
                )
            
            # Highlight only points where |Δ| > threshold
            for lesion_id in sorted(changing_lesions):
                lesion_data = df_changing[df_changing['Absolute_ID'] == lesion_id].sort_values('Session')
                
                # Calculate delta for each point
                baseline_volume = lesion_data.iloc[0]['Volume']
                deltas = lesion_data['Volume'] - baseline_volume
                
                # Find points exceeding threshold
                exceeding_mask = deltas.abs() > growth_threshold
                exceeding_data = lesion_data[exceeding_mask]
                
                if not exceeding_data.empty:
                    # Color code: red for growth, blue for shrinkage
                    for _, row in exceeding_data.iterrows():
                        delta = row['Volume'] - baseline_volume
                        point_color = 'red' if delta > 0 else 'blue'
                        fig.add_trace(
                            go.Scatter(
                                x=[row['Session']],
                                y=[row['Volume']],
                                mode='markers',
                                name=f'Lesion {lesion_id}',
                                marker=dict(size=12, color=point_color, 
                                          line=dict(color='black', width=2)),
                                showlegend=False,
                                legendgroup='changing',
                                hovertemplate='Lesion: %{text}<br>Session: %{x}<br>Volume: %{y:.2f}<extra></extra>',
                                text=[str(lesion_id)]
                            ),
                            row=1, col=2
                        )
        else:
            # Original behavior: colored lines
            for lesion_id in sorted(changing_lesions):
                lesion_data = df_changing[df_changing['Absolute_ID'] == lesion_id].sort_values('Session')
                fig.add_trace(
                    go.Scatter(
                        x=lesion_data['Session'],
                        y=lesion_data['Volume'],
                        mode='lines+markers',
                        name=f'Lesion {lesion_id}',
                        line=dict(color=color_map[lesion_id], width=2.5),
                        marker=dict(size=6),
                        showlegend=False,
                        legendgroup='changing',
                        hovertemplate='Lesion: %{text}<br>Session: %{x}<br>Volume: %{y:.2f}<extra></extra>',
                        text=[str(lesion_id)] * len(lesion_data)
                    ),
                    row=1, col=2
                )
    else:
        # Add annotation for no changing lesions
        fig.add_annotation(
            text=f'No changing lesions<br>(threshold: {growth_threshold} {volume_unit})',
            xref='x2', yref='y2',
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12),
            row=1, col=2
        )
    
    # Subplot 3: New lesions evolution (pre-lesions with their baseline/new phases)
    lesions_with_pre = df[df['Status'] == 'Pre-lesion']['Absolute_ID'].unique()
    if len(lesions_with_pre) > 0:
        for abs_id in sorted(lesions_with_pre):
            lesion_df = df[df['Absolute_ID'] == abs_id].sort_values('Session')
            
            # Combine pre-lesion and baseline/new data to create continuous line
            all_sessions = []
            all_volumes = []
            
            # Get pre-lesion data
            pre_data = lesion_df[lesion_df['Status'] == 'Pre-lesion']
            if not pre_data.empty:
                all_sessions.extend(pre_data['Session'].tolist())
                all_volumes.extend(pre_data['Volume'].tolist())
            
            # Get baseline/new data
            active_data = lesion_df[lesion_df['Status'].isin(['Baseline', 'New'])]
            if not active_data.empty:
                all_sessions.extend(active_data['Session'].tolist())
                all_volumes.extend(active_data['Volume'].tolist())
            
            # Sort by session to ensure proper line connection
            if all_sessions:
                sorted_indices = np.argsort(all_sessions)
                sorted_sessions = np.array(all_sessions)[sorted_indices]
                sorted_volumes = np.array(all_volumes)[sorted_indices]
                
                # Plot as continuous line with markers
                fig.add_trace(
                    go.Scatter(
                        x=sorted_sessions,
                        y=sorted_volumes,
                        mode='lines+markers',
                        name=f'Lesion {abs_id}',
                        line=dict(color=color_map[abs_id], width=2),
                        marker=dict(size=6),
                        showlegend=False,
                        legendgroup='new',
                        hovertemplate='Lesion: %{text}<br>Session: %{x}<br>Volume: %{y:.2f}<extra></extra>',
                        text=[str(abs_id)] * len(sorted_sessions)
                    ),
                    row=1, col=3
                )
                
                # Add special markers for pre-lesion points
                if not pre_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=pre_data['Session'],
                            y=pre_data['Volume'],
                            mode='markers',
                            name=f'Lesion {abs_id} pre',
                            marker=dict(size=10, color=color_map[abs_id], symbol='square',
                                      line=dict(color='black', width=1.5)),
                            showlegend=False,
                            legendgroup='new',
                            hovertemplate='Lesion: %{text} (pre-lesion)<br>Session: %{x}<br>Volume: %{y:.2f}<extra></extra>',
                            text=[str(abs_id)] * len(pre_data)
                        ),
                        row=1, col=3
                    )
    else:
        # Add annotation for no new lesions
        fig.add_annotation(
            text='No new lesions found',
            xref='x3', yref='y3',
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12),
            row=1, col=3
        )
    
    # Update layout
    fig.update_xaxes(title_text='Session', row=1, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text='Session', row=1, col=2, gridcolor='lightgray')
    fig.update_xaxes(title_text='Session', row=1, col=3, gridcolor='lightgray')
    
    fig.update_yaxes(title_text=f'Volume ({volume_unit})', row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text=f'Volume ({volume_unit})', row=1, col=2, gridcolor='lightgray')
    fig.update_yaxes(title_text=f'Volume ({volume_unit})', row=1, col=3, gridcolor='lightgray')
    
    if session_names:
        tickvals = list(range(1, len(session_names) + 1))
        fig.update_xaxes(tickvals=tickvals, ticktext=session_names, row=1, col=1)
        fig.update_xaxes(tickvals=tickvals, ticktext=session_names, row=1, col=2)
        fig.update_xaxes(tickvals=tickvals, ticktext=session_names, row=1, col=3)
    
    fig.update_layout(
        height=height,
        width=width,
        showlegend=True,
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02, font=dict(size=8)),
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig


def plot_trajectory_classification(df, session_names, width, height, 
                                   color_map, volume_unit='mm³', threshold=2.5,
                                   colormap='nipy_spectral'):
    """
    Plot lesions classified by their trajectory type: growth, shrinkage, or mixed using Plotly.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with lesion volume data including Delta column.
    session_names : list of str
        Names for each session.
    width : int
        Plot width in pixels.
    height : int
        Plot height in pixels.
    color_map : dict
        Color mapping for lesion IDs.
    volume_unit : str
        Unit for volume ('voxels' or 'mm³').
    threshold : float
        Absolute delta threshold for considering a change significant.
    colormap : str
        Name of matplotlib colormap (not used if color_map provided).
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The created figure.
    """
    
    # Classify lesions by trajectory
    growth_only = set()
    shrinkage_only = set()
    mixed = set()
    stable = set()
    
    df_active = df[df['Status'].isin(['Baseline', 'New', 'Confluent'])]
    
    for abs_id in df_active['Absolute_ID'].unique():
        lesion_data = df_active[df_active['Absolute_ID'] == abs_id].sort_values('Session')
        
        if len(lesion_data) > 1:
            # Get all deltas (excluding NaN)
            deltas = lesion_data['Delta'].dropna()
            
            if len(deltas) > 1:
                # Check if any significant changes occur (above threshold)
                has_positive = (deltas > threshold).any()
                has_negative = (deltas < -threshold).any()
                
                if has_positive and has_negative:
                    mixed.add(abs_id)
                elif has_positive:
                    growth_only.add(abs_id)
                elif has_negative:
                    shrinkage_only.add(abs_id)
                else:
                    stable.add(abs_id)
            else:
                stable.add(abs_id)
        else:
            stable.add(abs_id)
    
    # Create figure with 4 subplots in 2x2 grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(f'Growth Only (n={len(growth_only)})<br>(Δ > {threshold} {volume_unit})',
                       f'Shrinkage Only (n={len(shrinkage_only)})<br>(Δ < -{threshold} {volume_unit})',
                       f'Mixed (Growth + Shrinkage) (n={len(mixed)})<br>(|Δ| > {threshold} {volume_unit})',
                       f'Stable (n={len(stable)})<br>(|Δ| ≤ {threshold} {volume_unit})'),
        horizontal_spacing=0.15,
        vertical_spacing=0.14
    )
    
    # Subplot 1: Growth only
    if growth_only:
        for lesion_id in sorted(growth_only):
            lesion_data = df_active[df_active['Absolute_ID'] == lesion_id].sort_values('Session')
            # Remove NaN values for plotting
            lesion_data_clean = lesion_data.dropna(subset=['Volume'])
            
            fig.add_trace(
                go.Scatter(
                    x=lesion_data_clean['Session'],
                    y=lesion_data_clean['Volume'],
                    mode='lines+markers',
                    name=f'Lesion {lesion_id}',
                    line=dict(color=color_map[lesion_id], width=2),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate='Lesion: %{text}<br>Session: %{x}<br>Volume: %{y:.2f}<extra></extra>',
                    text=[str(lesion_id)] * len(lesion_data_clean)
                ),
                row=1, col=1
            )
            
            # Annotate first and last points
            if len(lesion_data_clean) > 0:
                first_row = lesion_data_clean.iloc[0]
                last_row = lesion_data_clean.iloc[-1]
                
                fig.add_annotation(
                    x=first_row['Session'], y=first_row['Volume'],
                    text=f"{first_row['Volume']:.1f}",
                    showarrow=False,
                    font=dict(size=8),
                    xshift=-10, yshift=-10,
                    row=1, col=1
                )
                fig.add_annotation(
                    x=last_row['Session'], y=last_row['Volume'],
                    text=f"{last_row['Volume']:.1f}",
                    showarrow=False,
                    font=dict(size=8),
                    xshift=10, yshift=-10,
                    row=1, col=1
                )
    else:
        fig.add_annotation(
            text='No growth-only lesions',
            xref='x', yref='y',
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12),
            row=1, col=1
        )
    
    # Subplot 2: Shrinkage only
    if shrinkage_only:
        for lesion_id in sorted(shrinkage_only):
            lesion_data = df_active[df_active['Absolute_ID'] == lesion_id].sort_values('Session')
            # Remove NaN values for plotting
            lesion_data_clean = lesion_data.dropna(subset=['Volume'])
            
            fig.add_trace(
                go.Scatter(
                    x=lesion_data_clean['Session'],
                    y=lesion_data_clean['Volume'],
                    mode='lines+markers',
                    name=f'Lesion {lesion_id}',
                    line=dict(color=color_map[lesion_id], width=2),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate='Lesion: %{text}<br>Session: %{x}<br>Volume: %{y:.2f}<extra></extra>',
                    text=[str(lesion_id)] * len(lesion_data_clean)
                ),
                row=1, col=2
            )
            
            # Annotate first and last points
            if len(lesion_data_clean) > 0:
                first_row = lesion_data_clean.iloc[0]
                last_row = lesion_data_clean.iloc[-1]
                
                fig.add_annotation(
                    x=first_row['Session'], y=first_row['Volume'],
                    text=f"{first_row['Volume']:.1f}",
                    showarrow=False,
                    font=dict(size=8),
                    xshift=-10, yshift=-10,
                    row=1, col=2
                )
                fig.add_annotation(
                    x=last_row['Session'], y=last_row['Volume'],
                    text=f"{last_row['Volume']:.1f}",
                    showarrow=False,
                    font=dict(size=8),
                    xshift=10, yshift=-10,
                    row=1, col=2
                )
    else:
        fig.add_annotation(
            text='No shrinkage-only lesions',
            xref='x2', yref='y2',
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12),
            row=1, col=2
        )
    
    # Subplot 3: Mixed (both growth and shrinkage)
    if mixed:
        for lesion_id in sorted(mixed):
            lesion_data = df_active[df_active['Absolute_ID'] == lesion_id].sort_values('Session')
            # Remove NaN values for plotting
            lesion_data_clean = lesion_data.dropna(subset=['Volume'])
            
            fig.add_trace(
                go.Scatter(
                    x=lesion_data_clean['Session'],
                    y=lesion_data_clean['Volume'],
                    mode='lines+markers',
                    name=f'Lesion {lesion_id}',
                    line=dict(color=color_map[lesion_id], width=2),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate='Lesion: %{text}<br>Session: %{x}<br>Volume: %{y:.2f}<extra></extra>',
                    text=[str(lesion_id)] * len(lesion_data_clean)
                ),
                row=2, col=1
            )
            
            # Annotate first and last points
            if len(lesion_data_clean) > 0:
                first_row = lesion_data_clean.iloc[0]
                last_row = lesion_data_clean.iloc[-1]
                
                fig.add_annotation(
                    x=first_row['Session'], y=first_row['Volume'],
                    text=f"{first_row['Volume']:.1f}",
                    showarrow=False,
                    font=dict(size=8),
                    xshift=-10, yshift=-10,
                    row=2, col=1
                )
                fig.add_annotation(
                    x=last_row['Session'], y=last_row['Volume'],
                    text=f"{last_row['Volume']:.1f}",
                    showarrow=False,
                    font=dict(size=8),
                    xshift=10, yshift=-10,
                    row=2, col=1
                )
    else:
        fig.add_annotation(
            text='No mixed trajectory lesions',
            xref='x3', yref='y3',
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12),
            row=1, col=2
        )
    
    # Subplot 4: Stable lesions
    if stable:
        for lesion_id in sorted(stable):
            lesion_data = df_active[df_active['Absolute_ID'] == lesion_id].sort_values('Session')
            # Remove NaN values for plotting
            lesion_data_clean = lesion_data.dropna(subset=['Volume'])
            
            fig.add_trace(
                go.Scatter(
                    x=lesion_data_clean['Session'],
                    y=lesion_data_clean['Volume'],
                    mode='lines+markers',
                    name=f'Lesion {lesion_id}',
                    line=dict(color=color_map[lesion_id], width=2),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate='Lesion: %{text}<br>Session: %{x}<br>Volume: %{y:.2f}<extra></extra>',
                    text=[str(lesion_id)] * len(lesion_data_clean)
                ),
                row=2, col=2
            )
            
            # Annotate first and last points
            if len(lesion_data_clean) > 0:
                first_row = lesion_data_clean.iloc[0]
                last_row = lesion_data_clean.iloc[-1]
                
                fig.add_annotation(
                    x=first_row['Session'], y=first_row['Volume'],
                    text=f"{first_row['Volume']:.1f}",
                    showarrow=False,
                    font=dict(size=8),
                    xshift=-10, yshift=-10,
                    row=2, col=2
                )
                fig.add_annotation(
                    x=last_row['Session'], y=last_row['Volume'],
                    text=f"{last_row['Volume']:.1f}",
                    showarrow=False,
                    font=dict(size=8),
                    xshift=10, yshift=-10,
                    row=2, col=2
                )
    else:
        fig.add_annotation(
            text='No stable lesions',
            xref='x4', yref='y4',
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12),
            row=2, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text='Session', row=1, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text='Session', row=1, col=2, gridcolor='lightgray')
    fig.update_xaxes(title_text='Session', row=2, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text='Session', row=2, col=2, gridcolor='lightgray')
    
    fig.update_yaxes(title_text=f'Volume ({volume_unit})', row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text=f'Volume ({volume_unit})', row=1, col=2, gridcolor='lightgray')
    fig.update_yaxes(title_text=f'Volume ({volume_unit})', row=2, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text=f'Volume ({volume_unit})', row=2, col=2, gridcolor='lightgray')
    
    if session_names:
        tickvals = list(range(1, len(session_names) + 1))
        fig.update_xaxes(tickvals=tickvals, ticktext=session_names, row=1, col=1)
        fig.update_xaxes(tickvals=tickvals, ticktext=session_names, row=1, col=2)
        fig.update_xaxes(tickvals=tickvals, ticktext=session_names, row=2, col=1)
        fig.update_xaxes(tickvals=tickvals, ticktext=session_names, row=2, col=2)
    
    fig.update_layout(
        title_text='Lesion Trajectory Classification',
        height=height,
        width=width,
        showlegend=False,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig, {'growth_only': growth_only, 'shrinkage_only': shrinkage_only, 
                 'mixed': mixed, 'stable': stable}


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    assert_inputs_exist(parser, args.in_images)
    assert_outputs_exist(parser, args, args.out_plot)
    if args.out_csv:
        assert_outputs_exist(parser, args, args.out_csv)
    if args.out_trajectory_plot:
        assert_outputs_exist(parser, args, args.out_trajectory_plot)
    assert_headers_compatible(parser, args.in_images)
    
    # Load images
    imgs = [nib.load(filename) for filename in args.in_images]
    
    # Calculate voxel volume
    if args.voxel_size:
        voxel_volume = np.prod(args.voxel_size)
        volume_unit = 'mm³'
    else:
        voxel_size = imgs[0].header.get_zooms()[:3]
        voxel_volume = np.prod(voxel_size)
        volume_unit = 'mm³'
    
    # Detect confluent lesions if requested
    confluent_lesions = {}
    if args.detect_confluent:
        print("Detecting confluent lesions using backwards-mapping...")
        confluent_lesions = detect_confluent_lesions(imgs, args.min_confluence_overlap)
        if confluent_lesions:
            print(f"Found {len(confluent_lesions)} confluent lesion(s):")
            for lesion_id, sources in confluent_lesions.items():
                print(f"  Lesion {lesion_id} merged from lesions: {sources}")
        else:
            print("No confluent lesions detected.")
    
    # Count lesion volumes
    df = count_lesion_volumes(imgs, voxel_volume, args.in_images, confluent_lesions)
    
    if df.empty:
        raise ValueError("No lesions found in input images.")
    
    # Calculate delta (volume change) compared to first appearance (baseline)
    # and fill missing sessions with NaN
    df_with_delta = []
    n_sessions = len(args.in_images)
    all_sessions = list(range(1, n_sessions + 1))
    
    for abs_id in df['Absolute_ID'].unique():
        lesion_data = df[df['Absolute_ID'] == abs_id].sort_values('Session')
        
        # Get subject and first session name for filling missing entries
        subject_id = lesion_data.iloc[0]['Subject']
        
        # For active and confluent lesions, calculate delta based on method
        active_data = lesion_data[lesion_data['Status'].isin(['Baseline', 'New', 'Confluent'])]
        if not active_data.empty:
            if args.delta_method == 'baseline':
                # Delta compared to first active/confluent appearance (baseline)
                baseline_volume = active_data.iloc[0]['Volume']
                lesion_data['Delta'] = lesion_data['Volume'] - baseline_volume
                # Set delta to 0 for pre-lesion points
                lesion_data.loc[lesion_data['Status'] == 'Pre-lesion', 'Delta'] = 0
            else:  # session-to-session
                # Delta compared to previous session
                lesion_data = lesion_data.sort_values('Session').reset_index(drop=True)
                lesion_data['Delta'] = lesion_data['Volume'].diff()
                # First session has delta = 0 (no previous session to compare)
                lesion_data.loc[lesion_data.index[0], 'Delta'] = 0
        else:
            lesion_data['Delta'] = 0
        
        # Find missing sessions
        present_sessions = set(lesion_data['Session'].values)
        missing_sessions = [s for s in all_sessions if s not in present_sessions]
        
        # Add missing sessions with NaN values
        for missing_session in missing_sessions:
            # Determine status: if lesion has appeared, missing means it disappeared
            # If it hasn't appeared yet, it's just not present
            if missing_session > lesion_data['Session'].max():
                # After last appearance - disappeared, use the lesion's original status
                status = active_data.iloc[0]['Status'] if not active_data.empty else 'New'
            elif not active_data.empty and missing_session >= active_data.iloc[0]['Session']:
                # After first appearance - disappeared, use the lesion's original status
                status = active_data.iloc[0]['Status']
            else:
                # Before any appearance
                continue  # Don't add entries before lesion appears
            
            # Get the Merged_From value from existing data for this lesion
            merged_from_val = lesion_data['Merged_From'].iloc[0] if 'Merged_From' in lesion_data.columns else ''
            
            missing_entry = {
                'Subject': subject_id,
                'Session': missing_session,
                'Session_Name': f'ses-{missing_session}',
                'Lesion_ID': int(abs_id),
                'Absolute_ID': int(abs_id),
                'Volume': np.nan,
                'Status': status,
                'Delta': np.nan,
                'Merged_From': merged_from_val
            }
            lesion_data = pd.concat([lesion_data, pd.DataFrame([missing_entry])], ignore_index=True)
        
        # Sort by session after adding missing entries
        lesion_data = lesion_data.sort_values('Session')
        df_with_delta.append(lesion_data)
    
    df = pd.concat(df_with_delta, ignore_index=True)
    
    # Calculate intra-lesion std for each lesion
    intra_std_dict = {}
    for abs_id in df['Absolute_ID'].unique():
        lesion_volumes = df[df['Absolute_ID'] == abs_id]['Volume'].dropna()
        if len(lesion_volumes) > 1:
            intra_std_dict[abs_id] = lesion_volumes.std()
        else:
            intra_std_dict[abs_id] = 0.0
    
    # Add intra_std column to dataframe
    df['Intra_Std'] = df['Absolute_ID'].map(intra_std_dict)
    
    # Classify lesions by trajectory group
    trajectory_dict = {}
    df_active = df[df['Status'].isin(['Baseline', 'New', 'Confluent'])]
    
    for abs_id in df['Absolute_ID'].unique():
        lesion_data = df_active[df_active['Absolute_ID'] == abs_id].sort_values('Session')
        
        if len(lesion_data) > 1:
            # Get all deltas (excluding NaN)
            deltas = lesion_data['Delta'].dropna()
            
            if len(deltas) > 1:
                # Check if any significant changes occur (above threshold)
                has_positive = (deltas > args.trajectory_threshold).any()
                has_negative = (deltas < -args.trajectory_threshold).any()
                
                if has_positive and has_negative:
                    trajectory_dict[abs_id] = 'Mixed'
                elif has_positive:
                    trajectory_dict[abs_id] = 'Growth'
                elif has_negative:
                    trajectory_dict[abs_id] = 'Shrinkage'
                else:
                    trajectory_dict[abs_id] = 'Stable'
            else:
                trajectory_dict[abs_id] = 'Stable'
        else:
            trajectory_dict[abs_id] = 'Stable'
    
    # Add trajectory group column to dataframe
    df['Trajectory_Group'] = df['Absolute_ID'].map(trajectory_dict)
    
    # Save CSV if requested
    if args.out_csv:
        # Reorder columns for better readability
        output_columns = ['Subject', 'Session', 'Session_Name', 'Absolute_ID', 
                         'Lesion_ID', 'Status', 'Volume', 'Delta', 'Intra_Std', 'Trajectory_Group', 'Merged_From']
        df_output = df[output_columns].sort_values(['Absolute_ID', 'Session'])
        df_output.to_csv(args.out_csv, index=False, float_format='%.3f')
        print(f"CSV data saved to: {args.out_csv}")
    
    # Filter by minimum volume
    if args.min_volume > 0:
        # Get max volume per lesion
        max_volumes = df.groupby('Absolute_ID')['Volume'].max()
        valid_lesions = max_volumes[max_volumes >= args.min_volume].index
        df = df[df['Absolute_ID'].isin(valid_lesions)]
    
    # Filter top N lesions
    if args.top_n_lesions:
        max_volumes = df.groupby('Absolute_ID')['Volume'].max()
        top_lesions = max_volumes.nlargest(args.top_n_lesions).index
        df = df[df['Absolute_ID'].isin(top_lesions)]
    
    # Create a consistent color map for all lesions using continuous colormap
    all_lesion_ids = sorted(df['Absolute_ID'].unique())
    n_lesions = len(all_lesion_ids)
    
    # Generate unique colors by combining multiple colormaps if needed
    if n_lesions <= 20:
        # Use single colormap for small number of lesions
        cmap = plt.cm.get_cmap(args.colormap)
        colors = [cmap(i / max(1, n_lesions - 1)) for i in range(n_lesions)]
    else:
        # Combine multiple colormaps for better color distinction
        # Use complementary colormaps to maximize color diversity
        colors_list = []
        
        # First 20 colors from primary colormap
        cmap1 = plt.cm.get_cmap('gist_earth')
        n_from_cmap1 = min(20, n_lesions)
        colors_list.extend([cmap1(i / 19) for i in range(n_from_cmap1)])
        
        if n_lesions > 20:
            # Next colors from gist_stern
            cmap2 = plt.cm.get_cmap('gist_stern')
            n_from_cmap2 = min(20, n_lesions - 20)
            colors_list.extend([cmap2(i / 19) for i in range(n_from_cmap2)])
        
        if n_lesions > 40:
            # Additional colors from gist_rainbow
            cmap3 = plt.cm.get_cmap('gist_rainbow')
            n_from_cmap3 = n_lesions - 40
            colors_list.extend([cmap3(i / max(1, n_from_cmap3 - 1)) for i in range(n_from_cmap3)])
        
        colors = colors_list[:n_lesions]
    
    # Create color map ensuring each lesion ID gets a unique color (convert to plotly RGB)
    color_map = {lesion_id: rgba_to_plotly(colors[i]) for i, lesion_id in enumerate(all_lesion_ids)}
    
    # Set session names
    if args.session_names:
        if len(args.session_names) != len(args.in_images):
            parser.error(f"Number of session names ({len(args.session_names)}) "
                        f"must match number of input images ({len(args.in_images)})")
        session_names = args.session_names
    else:
        session_names = [f'Session {i+1}' for i in range(len(args.in_images))]
    
    # Create plot
    fig = plot_lesion_evolution(df, session_names, args.width, args.height,
                                args.separate_pre_lesions, 
                                volume_unit, args.growth_threshold,
                                args.highlight_changes, args.colormap, color_map)
    
    # Determine output filenames (remove extension if provided)
    out_base = os.path.splitext(args.out_plot)[0]
    out_html = f"{out_base}.html"
    out_png = f"{out_base}.png"
    
    # Save plot as HTML
    fig.write_html(out_html)
    print(f"Interactive HTML plot saved to: {out_html}")
    
    # Save plot as PNG
    fig.write_image(out_png, width=args.width, height=args.height)
    print(f"Static PNG plot saved to: {out_png}")
    
    # Create and save trajectory classification plot if requested
    if args.out_trajectory_plot:
        traj_width = args.trajectory_width if hasattr(args, 'trajectory_width') else args.width
        traj_height = args.trajectory_height if hasattr(args, 'trajectory_height') else args.height
        fig_traj, trajectory_stats = plot_trajectory_classification(
            df, session_names, traj_width, traj_height,
            color_map, volume_unit, args.trajectory_threshold, args.colormap)
        
        # Determine trajectory output filenames
        traj_base = os.path.splitext(args.out_trajectory_plot)[0]
        traj_html = f"{traj_base}.html"
        traj_png = f"{traj_base}.png"
        
        # Save trajectory plot as HTML
        fig_traj.write_html(traj_html)
        print(f"Trajectory classification HTML plot saved to: {traj_html}")
        
        # Save trajectory plot as PNG
        fig_traj.write_image(traj_png, width=traj_width, height=traj_height)
        print(f"Trajectory classification PNG plot saved to: {traj_png}")
        
        # Print trajectory statistics
        print("\n=== Trajectory Classification ===")
        print(f"(Using threshold: |Δ| > {args.trajectory_threshold} {volume_unit})")
        print(f"Growth only: {len(trajectory_stats['growth_only'])}")
        if trajectory_stats['growth_only']:
            print(f"  IDs: {sorted(trajectory_stats['growth_only'])}")
        print(f"Shrinkage only: {len(trajectory_stats['shrinkage_only'])}")
        if trajectory_stats['shrinkage_only']:
            print(f"  IDs: {sorted(trajectory_stats['shrinkage_only'])}")
        print(f"Mixed (growth + shrinkage): {len(trajectory_stats['mixed'])}")
        if trajectory_stats['mixed']:
            print(f"  IDs: {sorted(trajectory_stats['mixed'])}")
        print(f"Stable (no significant change): {len(trajectory_stats['stable'])}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total number of unique lesions: {df['Absolute_ID'].nunique()}")
    print(f"Number of sessions: {len(args.in_images)}")
    print(f"\nLesions with pre-lesion phase: "
          f"{df[df['Status'] == 'Pre-lesion']['Absolute_ID'].nunique()}")
    print(f"Lesions with active phase: "
          f"{df[df['Status'].isin(['Baseline', 'New'])]['Absolute_ID'].nunique()}")
    
    # Print confluent lesions if detected
    if args.detect_confluent:
        confluent_count = df[df['Status'] == 'Confluent']['Absolute_ID'].nunique()
        print(f"Confluent lesions (merged from multiple sources): {confluent_count}")
        if confluent_count > 0:
            confluent_ids = df[df['Status'] == 'Confluent']['Absolute_ID'].unique()
            print(f"  IDs: {sorted(confluent_ids)}")
    
    # Count growing lesions (exclude confluent for cleaner statistics)
    changing_lesions = set()
    for abs_id in df[df['Status'].isin(['Baseline', 'New', 'Confluent'])]['Absolute_ID'].unique():
        lesion_data = df[(df['Absolute_ID'] == abs_id) & (df['Status'].isin(['Baseline', 'New', 'Confluent']))].sort_values('Session')
        if len(lesion_data) > 1:
            baseline_volume = lesion_data.iloc[0]['Volume']
            max_abs_change = (lesion_data['Volume'] - baseline_volume).abs().max()
            if max_abs_change > args.growth_threshold:
                changing_lesions.add(abs_id)
    
    print(f"Changing lesions (|Δ| > {args.growth_threshold} {volume_unit}): {len(changing_lesions)}")
    if changing_lesions:
        print(f"  IDs: {sorted(changing_lesions)}")
    
    # Volume statistics
    print(f"\n=== Volume Statistics ({volume_unit}) ===")
    status_list = ['Pre-lesion', 'Baseline', 'New']
    if args.detect_confluent and not df[df['Status'] == 'Confluent'].empty:
        status_list.append('Confluent')
    
    for status in status_list:
        status_df = df[df['Status'] == status]
        if not status_df.empty:
            print(f"\n{status}:")
            print(f"  Mean volume: {status_df['Volume'].mean():.2f}")
            print(f"  Median volume: {status_df['Volume'].median():.2f}")
            print(f"  Min volume: {status_df['Volume'].min():.2f}")
            print(f"  Max volume: {status_df['Volume'].max():.2f}")
            
            # Calculate intra-lesion variability (std across sessions for each lesion)
            intra_lesion_stds = []
            for abs_id in status_df['Absolute_ID'].unique():
                lesion_volumes = status_df[status_df['Absolute_ID'] == abs_id]['Volume'].dropna()
                if len(lesion_volumes) > 1:
                    intra_lesion_stds.append(lesion_volumes.std())
            
            if intra_lesion_stds:
                mean_intra_std = np.mean(intra_lesion_stds)
                print(f"  Mean intra-lesion std (across sessions): {mean_intra_std:.2f}")


if __name__ == "__main__":
    main()
