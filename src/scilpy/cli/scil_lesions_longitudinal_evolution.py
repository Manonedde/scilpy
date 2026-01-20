#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script processes longitudinal lesion data to track lesion evolution
across multiple sessions. It provides tools to:

1. Add pre-lesion labels: Track lesion presence before their first appearance
   using negative-valued labels.

2. Detect confluent lesions: Identify lesions that result from the merger of
   multiple distinct lesions from the first session using backwards-mapping.
   These merged lesions are marked as distinct entities.

3. Create confluent maps: Generate separate label maps showing confluent lesions
   grouped with their source lesions using new labels (ID + 1000). Original maps
   preserve their labels.

4. Fill intermediate missing lesions: Correct segmentation errors where lesions
   were accidentally not segmented in intermediate timepoints. This only fills
   single-session gaps (not segmentation errors spanning multiple sessions).

5. Validate new lesion distance: Ensure that newly appearing lesions are at least
   a minimum distance (default: 2 mm) away from any pre-existing lesions at all
   timepoints prior to their first appearance. This constraint validates that
   detected lesions are truly new and not artifacts or mislabelings.

All images should be co-registered and have the same shape.

Usage:
    For pre-lesion tracking:
        scil_lesions_longitudinal_evolution.py ses-1.nii.gz ses-2.nii.gz ... \\
                                                  output_dir --add_pre_lesion_labels

    For detecting and relabeling confluent lesions:
        scil_lesions_longitudinal_evolution.py ses-1.nii.gz ses-2.nii.gz ... \\
                                                  output_dir --detect_confluent \\
                                                  --min_confluence_overlap 1

    For validating new lesion distance:
        scil_lesions_longitudinal_evolution.py ses-1.nii.gz ses-2.nii.gz ... \\
                                                  output_dir \\
                                                  --validate_lesion_distance \\
                                                  --min_distance_mm 2.0

    For filling intermediate missing lesions:
        scil_lesions_longitudinal_evolution.py ses-1.nii.gz ses-2.nii.gz ... \\
                                                  output_dir \\
                                                  --fill_intermediate_missing
"""

import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             assert_headers_compatible)

EPILOG = """
References:
    [1] Köhler, Caroline, et al. "Exploring individual multiple sclerosis
    lesion volume change over time: development of an algorithm for the
    analyses of longitudinal quantitative MRI measures."
    NeuroImage: Clinical 21 (2019): 101623.
    
    [2] Sormani, Maria P., et al. "Magnetic Resonance Imaging as a Potential
    Measure of Lesion Burden and Brain Atrophy in Multiple Sclerosis."
    Nature Reviews Neurology 17.8 (2021): 465-481.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC8453433/
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_images', nargs='+',
                   help='Input lesion label files in NIfTI format, '
                        'in chronological order.')
    p.add_argument('out_dir',
                   help='Output directory for processed lesion files.')
    
    p.add_argument('--all', action='store_true',
                   help='Apply all transformations: add pre-lesion labels, '
                        'detect and relabel confluent lesions, and fill '
                        'intermediate missing lesions.')

    g1 = p.add_argument_group('Group 1: Lesion evolution across sessions',
                              'Track and correct lesion presence across timepoints')
    g1.add_argument('--add_pre_lesion_labels', action='store_true',
                    help='Add pre-lesion labels for each lesion to all '
                         'sessions before it appears for the first time. '
                         'Pre-lesion labels use negative values of the lesion ID.')
    g1.add_argument('--fill_intermediate_missing', action='store_true',
                    help='Fill missing lesions in intermediate sessions. If a '
                         'lesion exists in session N and N+2 but is missing in '
                         'N+1, copy it from session N (previous session). This '
                         'corrects segmentation errors where lesions were '
                         'accidentally not segmented in intermediate timepoints.')

    g2 = p.add_argument_group('Group 2: Detect confluent lesions',
                              'Identify lesions resulting from mergers or contact')
    g2.add_argument('--detect_confluent', action='store_true',
                    help='Detect confluent and contact lesions using backwards-mapping. '
                         'Lesions are classified as CONFLUENT if overlap increases >4 voxels '
                         'between sessions (merging), or CONTACT if overlap is stable (touching). '
                         'Original maps keep their labels; separate *_confluent maps group all '
                         'source lesions with the confluent label (ID+1000). Type and overlap '
                         'evolution are recorded in the CSV report.')
    g2.add_argument('--min_confluence_overlap', type=int, default=4,
                    help='Minimum number of overlapping voxels to consider a confluence '
                         '[%(default)s].')

    g3 = p.add_argument_group('Group 3: Validation of new lesions',
                              'Validate spatial constraints for newly appearing lesions')
    g3.add_argument('--validate_lesion_distance', action='store_true',
                    help='Validate that each NEW lesion is at least min_distance_mm away '
                         'from any pre-existing lesion at all timepoints prior to '
                         'lesion onset. Lesions violating this constraint are reported.')
    g3.add_argument('--min_distance_mm', type=float, default=2.0,
                    help='Minimum distance in millimeters required between a new lesion '
                         'and pre-existing lesions [%(default)s].')



    add_overwrite_arg(p)
    return p


# -------------------------------------------------------------------------
# FUNCTION: Generate lesion information CSV report
# -------------------------------------------------------------------------
def generate_lesion_report(relabeled_data, output_dir, affine=None, confluent_lesions=None):
    """
    Generate a CSV report with information about all lesions.
    
    For each lesion, the report includes:
    - Lesion ID
    - First appearance session
    - Number of voxels at each session
    - Total voxel count across all sessions
    - Volume in mm³ (if affine is provided)
    - Confluent status and associated lesions (if confluent_lesions provided)
    
    Parameters
    ----------
    relabeled_data : list of np.ndarray
        List of 3D label arrays (same shape), one per session,
        in chronological order.
    output_dir : str
        Directory where the CSV file will be saved.
    affine : np.ndarray, optional
        4x4 affine transformation matrix for computing volume in mm³.
        If not provided, only voxel counts are reported.
    confluent_lesions : dict, optional
        Dictionary mapping confluent lesion IDs to their source lesion IDs.
        If provided, adds confluent information to the report.
    
    Returns
    -------
    csv_path : str
        Path to the generated CSV file.
    """
    n_sessions = len(relabeled_data)
    
    # Find all unique lesion labels and their first appearance
    lesion_info = {}
    
    for data in relabeled_data:
        for lesion_id in np.unique(data[data > 0]):
            lesion_id = int(lesion_id)
            if lesion_id not in lesion_info:
                lesion_info[lesion_id] = {
                    'first_session': None,
                    'voxel_counts': [0] * n_sessions
                }
    
    # Count voxels per lesion per session and find first appearance
    for session_idx in range(n_sessions):
        for lesion_id in np.unique(relabeled_data[session_idx][relabeled_data[session_idx] > 0]):
            lesion_id = int(lesion_id)
            voxel_count = np.sum(relabeled_data[session_idx] == lesion_id)
            lesion_info[lesion_id]['voxel_counts'][session_idx] = voxel_count
            
            if lesion_info[lesion_id]['first_session'] is None:
                lesion_info[lesion_id]['first_session'] = session_idx
    
    # Build dataframe
    rows = []
    for lesion_id in sorted(lesion_info.keys()):
        lesion_data = lesion_info[lesion_id]
        row = {
            'Lesion_ID': lesion_id,
            'First_Session': lesion_data['first_session'],
            'Total_Voxels': sum(lesion_data['voxel_counts'])
        }
        
        # Add confluent information if available
        if confluent_lesions is not None:
            if lesion_id in confluent_lesions:
                confluent_info = confluent_lesions[lesion_id]
                row['Is_Confluent'] = True
                row['Confluent_Type'] = confluent_info['type']
                row['Confluent_Label'] = lesion_id + 1000
                row['Merged_From'] = ','.join(map(str, confluent_info['sources']))
                # Add overlap counts for each source
                overlap_strings = [f"{src}({confluent_info['overlaps'][src]}v)" for src in confluent_info['sources']]
                row['Merged_From_Overlaps'] = ','.join(overlap_strings)
                
                # Add overlap evolution across sessions
                overlap_evolution = confluent_info['overlap_evolution']
                evolution_strings = []
                for sess_idx in sorted(overlap_evolution.keys()):
                    sess_overlaps = overlap_evolution[sess_idx]
                    sess_str = f"S{sess_idx}:" + '+'.join([f"{src}({count})" for src, count in sorted(sess_overlaps.items())])
                    evolution_strings.append(sess_str)
                row['Overlap_Evolution'] = '; '.join(evolution_strings)
            else:
                row['Is_Confluent'] = False
                row['Confluent_Type'] = ''
                row['Confluent_Label'] = ''
                row['Merged_From'] = ''
                row['Merged_From_Overlaps'] = ''
                row['Overlap_Evolution'] = ''
        
        # Add voxel counts per session
        for session_idx in range(n_sessions):
            row[f'Session_{session_idx}_Voxels'] = lesion_data['voxel_counts'][session_idx]
        
        # Add volume in mm³ if affine is provided
        if affine is not None:
            voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))
            total_voxels = sum(lesion_data['voxel_counts'])
            volume_mm3 = total_voxels * voxel_volume
            row['Volume_mm3'] = volume_mm3
        
        rows.append(row)
    
    # Create dataframe
    df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'longitudinal_lesions_report.csv')
    df.to_csv(csv_path, index=False)
    
    return csv_path


# -------------------------------------------------------------------------
# FUNCTION: Validate lesion distance from pre-existing lesions
# -------------------------------------------------------------------------
def validate_lesion_distance(relabeled_data, affine, min_distance_mm=2.0):
    """
    Validate that NEW lesions are at least min_distance_mm away from any 
    pre-existing lesion at all timepoints prior to lesion onset.
    
    This function validates only NEW lesions (i.e., lesions that first appear 
    in sessions after the initial session). For each new lesion, it checks that 
    the lesion maintains a minimum spatial distance from all lesions that existed 
    before its first appearance, ensuring that detected lesions are truly new and 
    not artifacts or mislabelings.
    
    Parameters
    ----------
    relabeled_data : list of np.ndarray
        List of 3D label arrays (same shape), one per session,
        in chronological order.
    affine : np.ndarray
        4x4 affine transformation matrix for converting voxel coordinates
        to physical space (mm). Since all images are co-registered, a single
        affine is used for all sessions.
    min_distance_mm : float
        Minimum distance in millimeters [%(default)s].
    
    Returns
    -------
    invalid_lesions : dict
        Dictionary mapping NEW lesion IDs to information about distance violations.
        Format: {lesion_id: {'first_session': int, 'violation_voxels': int,
                             'min_distance_mm': float}}
    """
    n_sessions = len(relabeled_data)
    invalid_lesions = {}
    
    # Find all unique lesion labels and their first appearance
    lesion_first_appearance = {}
    for data in relabeled_data:
        for lesion_id in np.unique(data[data > 0]):
            if int(lesion_id) not in lesion_first_appearance:
                lesion_first_appearance[int(lesion_id)] = None
    
    for session_idx in range(n_sessions):
        for lesion_id in np.unique(relabeled_data[session_idx][relabeled_data[session_idx] > 0]):
            lesion_id = int(lesion_id)
            if lesion_first_appearance[lesion_id] is None:
                lesion_first_appearance[lesion_id] = session_idx
    
    # For each NEW lesion (appearing after first session), check distance from pre-existing lesions
    for lesion_id, first_session in lesion_first_appearance.items():
        # Only validate new lesions (first appearance after the initial session)
        if first_session is None or first_session == 0:
            # Skip lesions that don't exist or appear only in the first session
            continue
        
        # Get voxel coordinates of this lesion in its first appearance
        lesion_mask = (relabeled_data[first_session] == lesion_id)
        lesion_voxels = np.argwhere(lesion_mask)
        
        # Convert to physical coordinates (mm)
        lesion_coords_mm = []
        for voxel in lesion_voxels:
            voxel_homogeneous = np.append(voxel, 1)
            coord_mm = affine @ voxel_homogeneous
            lesion_coords_mm.append(coord_mm[:3])
        lesion_coords_mm = np.array(lesion_coords_mm)
        
        # Get all pre-existing lesion voxels (from sessions before first_session)
        pre_existing_mask = np.zeros_like(relabeled_data[first_session], dtype=bool)
        for prev_session in range(first_session):
            pre_existing_mask |= relabeled_data[prev_session] > 0
        
        pre_existing_voxels = np.argwhere(pre_existing_mask)
        
        if len(pre_existing_voxels) == 0:
            # No pre-existing lesions, skip validation
            continue
        
        # Convert pre-existing voxels to physical coordinates (mm)
        pre_existing_coords_mm = []
        for voxel in pre_existing_voxels:
            voxel_homogeneous = np.append(voxel, 1)
            coord_mm = affine @ voxel_homogeneous
            pre_existing_coords_mm.append(coord_mm[:3])
        pre_existing_coords_mm = np.array(pre_existing_coords_mm)
        
        # Compute distances from lesion voxels to nearest pre-existing lesion voxel
        distances = cdist(lesion_coords_mm, pre_existing_coords_mm)
        min_distances = np.min(distances, axis=1)
        
        # Check if any voxel violates the minimum distance requirement
        violation_voxels = np.sum(min_distances < min_distance_mm)
        
        if violation_voxels > 0:
            invalid_lesions[lesion_id] = {
                'first_session': first_session,
                'violation_voxels': int(violation_voxels),
                'min_distance_mm': float(np.min(min_distances))
            }
    
    return invalid_lesions


# -------------------------------------------------------------------------
# FUNCTION: Add pre-lesion labels
# -------------------------------------------------------------------------
def add_pre_lesion_labels(relabeled_data):
    """
    Add pre-lesion labels for each new-lesion to all sessions before it 
    appears for the first time.
    
    For each lesion label that appears in the dataset, this function:
    1. Finds the first session where the lesion appears
    2. Identifies the spatial location of that lesion
    3. Adds a "pre-lesion" label (negative value of lesion ID) to that 
       location in all previous sessions
    
    Parameters
    ----------
    relabeled_data : list of np.ndarray
        List of 3D label arrays (same shape), one per session, 
        in chronological order.
    
    Returns
    -------
    data_with_pre_lesions : list of np.ndarray
        List of label arrays with pre-lesion labels added.
        Pre-lesion labels use negative values of the corresponding lesion ID.
    """
    # Convert to int32 to support negative pre-lesion labels
    data_with_pre_lesions = [data.astype(np.int32) for data in relabeled_data]
    n_sessions = len(relabeled_data)
    
    # Find all unique lesion labels across all sessions
    all_labels = set()
    for data in relabeled_data:
        all_labels.update(np.unique(data[data > 0]))
    
    # For each lesion label, find when it first appears
    for lesion_id in sorted(all_labels):
        first_appearance = None
        lesion_mask = None
        
        # Find the first session where this lesion appears
        for session_idx in range(n_sessions):
            if lesion_id in relabeled_data[session_idx]:
                first_appearance = session_idx
                # Get the mask of this lesion in its first appearance
                lesion_mask = (relabeled_data[session_idx] == lesion_id)
                break
        
        if first_appearance is None or first_appearance == 0:
            # Lesion doesn't exist or appears in first session (no pre-lesion needed)
            continue
        
        # Add pre-lesion label to all sessions before first appearance
        pre_lesion_label = -int(lesion_id)  # Use negative value for pre-lesion
        
        for session_idx in range(first_appearance):
            # Only add pre-lesion label where there's no existing lesion
            pre_lesion_region = lesion_mask & (data_with_pre_lesions[session_idx] == 0)
            data_with_pre_lesions[session_idx][pre_lesion_region] = pre_lesion_label
    
    return data_with_pre_lesions


# -------------------------------------------------------------------------
# FUNCTION: Detect confluent lesions with overlap analysis
# -------------------------------------------------------------------------
def detect_confluent_lesions(relabeled_data, min_overlap=1):
    """
    Detect confluent and contact lesions using backwards-mapping and overlap evolution.
    
    A lesion is considered:
    - CONFLUENT: if it overlaps with multiple lesions from first session AND 
                 overlap increases by >4 voxels between sessions (merging)
    - CONTACT: if it overlaps with multiple lesions from first session BUT
               overlap remains stable (just touching, not merging)
    
    Uses backwards-mapping: for each lesion in session N, check if its footprint
    overlaps with multiple lesions in session 1, then analyze overlap evolution.
    
    Parameters
    ----------
    relabeled_data : list of np.ndarray
        List of 3D label arrays (same shape), one per session,
        in chronological order.
    min_overlap : int
        Minimum number of overlapping voxels to consider a merge.
    
    Returns
    -------
    confluent_lesions : dict
        Dictionary mapping lesion IDs to info about confluent lesions.
        Format: {lesion_id: {'sources': [source_id1, source_id2, ...],
                             'overlaps': {source_id1: voxel_count1, ...},
                             'type': 'confluent' or 'contact',
                             'overlap_evolution': {session_idx: {source_id: count, ...}, ...}}}
    """
    if len(relabeled_data) < 2:
        return {}
    
    confluent_lesions = {}
    
    # Get first session data
    first_session = relabeled_data[0]
    
    # For each subsequent session, check for merges
    for session_idx in range(1, len(relabeled_data)):
        current_session = relabeled_data[session_idx]
        
        # Get all positive lesion IDs in current session
        current_labels = np.unique(current_session)
        current_labels = current_labels[current_labels > 0]
        
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
            
            # If this lesion overlaps with multiple lesions from first session
            if len(overlap_counts) > 1:
                # Track overlap evolution across all sessions
                overlap_evolution = {}
                
                for sess_idx in range(len(relabeled_data)):
                    sess_data = relabeled_data[sess_idx]
                    
                    # Check if this lesion exists in this session
                    if np.any(sess_data == lesion_id):
                        lesion_mask_sess = (sess_data == lesion_id)
                        first_sess_view = first_session[lesion_mask_sess]
                        
                        sess_overlaps = {}
                        for overlap_label in overlap_counts.keys():
                            count = np.sum(first_sess_view == overlap_label)
                            if count > 0:
                                sess_overlaps[overlap_label] = count
                        
                        if sess_overlaps:
                            overlap_evolution[sess_idx] = sess_overlaps
                
                # Determine if confluent (merging) or contact (just touching)
                lesion_type = 'contact'  # Default to contact
                
                # Check if overlap increases by more than 4 voxels between any consecutive sessions
                session_indices = sorted(overlap_evolution.keys())
                for i in range(len(session_indices) - 1):
                    curr_sess = session_indices[i]
                    next_sess = session_indices[i + 1]
                    
                    # Calculate total overlap for each session
                    curr_total = sum(overlap_evolution[curr_sess].values())
                    next_total = sum(overlap_evolution[next_sess].values())
                    
                    if next_total - curr_total > 4:
                        lesion_type = 'confluent'
                        break
                
                confluent_lesions[int(lesion_id)] = {
                    'sources': sorted(overlap_counts.keys()),
                    'overlaps': overlap_counts,
                    'type': lesion_type,
                    'overlap_evolution': overlap_evolution
                }
    
    return confluent_lesions


# -------------------------------------------------------------------------
# FUNCTION: Create confluent lesion maps
# -------------------------------------------------------------------------
def create_confluent_lesion_maps(relabeled_data, confluent_lesions):
    """
    Create separate maps for confluent lesions with grouped labels.
    
    This function creates new label maps where all lesions that merged into
    a confluent lesion are grouped together with the confluent lesion's new ID.
    The original maps remain unchanged.
    
    Parameters
    ----------
    relabeled_data : list of np.ndarray
        List of 3D label arrays (same shape), one per session.
    confluent_lesions : dict
        Dictionary mapping lesion IDs to their source lesion IDs.
    
    Returns
    -------
    confluent_maps : list of np.ndarray
        List of label arrays showing confluent lesions with grouped labels.
        In these maps, all source lesions are relabeled to the confluent ID + 1000.
    """
    if not confluent_lesions:
        return None
    
    confluent_maps = [data.copy() for data in relabeled_data]
    
    for lesion_id, info in confluent_lesions.items():
        new_label = lesion_id + 1000
        source_ids = info['sources']
        
        # Relabel the confluent lesion and all its sources to the new label
        for session_idx in range(len(confluent_maps)):
            # Relabel the confluent lesion itself
            mask = (confluent_maps[session_idx] == lesion_id)
            confluent_maps[session_idx][mask] = new_label
            
            # Relabel all source lesions
            for source_id in source_ids:
                source_mask = (confluent_maps[session_idx] == source_id)
                confluent_maps[session_idx][source_mask] = new_label
    
    return confluent_maps


# -------------------------------------------------------------------------
# FUNCTION: Fill intermediate missing lesions
# -------------------------------------------------------------------------
def fill_intermediate_missing_lesions(relabeled_data):
    """
    Fill missing lesions in intermediate sessions by copying from the 
    previous session.
    
    If a lesion with label L exists in session N and session N+2 but is 
    missing ONLY in session N+1, this function copies the lesion from 
    session N to session N+1. This corrects segmentation errors where a 
    lesion was accidentally not segmented in an intermediate timepoint.
    
    IMPORTANT: This only fills single-session gaps. If a lesion is missing 
    for 2 or more consecutive sessions, it is NOT filled (assumed to be 
    genuinely absent, not a segmentation error).
    
    Parameters
    ----------
    relabeled_data : list of np.ndarray
        List of 3D label arrays (same shape), one per session,
        in chronological order.
    
    Returns
    -------
    filled_data : list of np.ndarray
        List of label arrays with intermediate missing lesions filled.
    """
    filled_data = [data.copy() for data in relabeled_data]
    n_sessions = len(relabeled_data)
    
    if n_sessions < 3:
        # Need at least 3 sessions to detect intermediate missing
        return filled_data
    
    # Find all unique lesion labels across all sessions
    all_labels = set()
    for data in relabeled_data:
        all_labels.update(np.unique(data[data > 0]))
    
    # For each lesion, check if it's missing in intermediate sessions
    for lesion_id in sorted(all_labels):
        # Track which sessions have this lesion
        sessions_with_lesion = []
        lesion_masks = {}
        
        for session_idx in range(n_sessions):
            if lesion_id in relabeled_data[session_idx]:
                sessions_with_lesion.append(session_idx)
                lesion_masks[session_idx] = (relabeled_data[session_idx] == lesion_id)
        
        if len(sessions_with_lesion) < 2:
            # Lesion appears in less than 2 sessions, no intermediate to fill
            continue
        
        # Check for gaps in the sequence
        for i in range(len(sessions_with_lesion) - 1):
            current_session = sessions_with_lesion[i]
            next_session = sessions_with_lesion[i + 1]
            
            # Only fill if there is EXACTLY ONE missing session between current and next
            # If gap is 2 or more sessions, do NOT fill (not a segmentation error)
            if next_session - current_session == 2:
                # There is exactly one missing session
                missing_session = current_session + 1
                
                # Use the mask from the previous session (current_session)
                source_mask = lesion_masks[current_session]
                
                # Only fill where there's no existing lesion
                fill_region = source_mask & (filled_data[missing_session] == 0)
                filled_data[missing_session][fill_region] = lesion_id
    
    return filled_data


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_images)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)
    assert_headers_compatible(parser, args.in_images)

    # Load input images
    imgs = [nib.load(filename) for filename in args.in_images]
    data = [img.get_fdata() for img in imgs]
    # Since all images are co-registered, use the affine from the first image
    affine = imgs[0].affine

    # If --all is specified, enable all processing options
    if args.all:
        args.add_pre_lesion_labels = True
        args.detect_confluent = True
        args.fill_intermediate_missing = True

    # Apply processing based on requested options
    if args.add_pre_lesion_labels:
        print("Adding pre-lesion labels...")
        data = add_pre_lesion_labels(data)

    confluent_lesions = None
    confluent_maps = None
    if args.detect_confluent:
        print("Detecting confluent lesions using backwards-mapping...")
        confluent_lesions = detect_confluent_lesions(data, 
                                                     args.min_confluence_overlap)
        if confluent_lesions:
            # Separate by type
            confluent_count = sum(1 for info in confluent_lesions.values() if info['type'] == 'confluent')
            contact_count = sum(1 for info in confluent_lesions.values() if info['type'] == 'contact')
            
            print(f"Found {len(confluent_lesions)} lesion(s) with multiple sources:")
            print(f"  - {confluent_count} confluent (merging)")
            print(f"  - {contact_count} contact (touching)")
            print()
            
            for lesion_id, info in confluent_lesions.items():
                sources = info['sources']
                overlaps = info['overlaps']
                lesion_type = info['type'].upper()
                overlap_str = ', '.join([f"{src} ({overlaps[src]} voxels)" for src in sources])
                
                print(f"  Lesion {lesion_id} [{lesion_type}] - sources: {overlap_str}")
                print(f"    -> Confluent label: {lesion_id + 1000}")
                
                # Show overlap evolution
                overlap_evolution = info['overlap_evolution']
                print("    -> Overlap evolution:")
                for sess_idx in sorted(overlap_evolution.keys()):
                    sess_overlaps = overlap_evolution[sess_idx]
                    total = sum(sess_overlaps.values())
                    detail = ', '.join([f"{src}({count})" for src, count in sorted(sess_overlaps.items())])
                    print(f"       Session {sess_idx}: {total} voxels total ({detail})")
                print()
            
            print("Creating separate confluent lesion maps...")
            confluent_maps = create_confluent_lesion_maps(data, confluent_lesions)
        else:
            print("No confluent lesions detected.")

    if args.fill_intermediate_missing:
        print("Filling intermediate missing lesions...")
        data = fill_intermediate_missing_lesions(data)

    if args.validate_lesion_distance:
        print(f"Validating lesion distance (min: {args.min_distance_mm} mm)...")
        invalid_lesions = validate_lesion_distance(data, affine, 
                                                   args.min_distance_mm)
        if invalid_lesions:
            print(f"Found {len(invalid_lesions)} lesion(s) violating minimum distance:")
            for lesion_id, info in invalid_lesions.items():
                print(f"  Lesion {lesion_id} (first appearance: session {info['first_session']})")
                print(f"    Violation voxels: {info['violation_voxels']}")
                print(f"    Minimum distance: {info['min_distance_mm']:.2f} mm "
                      f"(required: {args.min_distance_mm} mm)")
        else:
            print("All lesions satisfy the minimum distance requirement.")

    # Save output images
    print(f"Saving {len(imgs)} processed image(s) to {args.out_dir}...")
    for i, img in enumerate(imgs):
        output_path = os.path.join(args.out_dir, os.path.basename(args.in_images[i]))
        nib.save(nib.Nifti1Image(data[i], img.affine), output_path)
    
    # Save confluent lesion maps if they were created
    if confluent_maps is not None:
        print(f"Saving {len(imgs)} confluent lesion map(s) to {args.out_dir}...")
        for i, img in enumerate(imgs):
            basename = os.path.basename(args.in_images[i])
            name, ext = os.path.splitext(basename)
            if ext == '.gz':
                name = os.path.splitext(name)[0]  # Remove .nii from .nii.gz
                ext = '.nii.gz'
            confluent_output = os.path.join(args.out_dir, f"{name}_confluent{ext}")
            nib.save(nib.Nifti1Image(confluent_maps[i], img.affine), confluent_output)
    
    # Generate lesion report CSV
    print("Generating lesion report CSV...")
    csv_path = generate_lesion_report(data, args.out_dir, affine, confluent_lesions)
    print(f"Lesion report saved to: {csv_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()
