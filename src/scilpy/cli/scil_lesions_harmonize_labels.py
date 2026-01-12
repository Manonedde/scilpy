#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script harmonizes labels across a set of lesion files represented in
NIfTI format. It ensures that labels are consistent across multiple input
images by matching labels between images based on spatial proximity and
overlap criteria.

The script works iteratively, so the multiple inputs should be in chronological
order (and changing the order affects the output). All images should be
co-registered.

To obtain labels from binary mask use scil_labels_from_mask.

WARNING: this script requires all files to have all lesions segmented.
If your data only show new lesions at each timepoints (common in manual
segmentation), use the option --incremental_lesions to merge past timepoints.
    T1 = T1, T2 = T1 + T2, T3 = T1 + T2 + T3

If a lesion exists in session N and session N+2 but is missing in session N+1
(segmentation error), use --fill_intermediate_missing to copy the lesion from
the previous session.

Confluent Lesions:
When --detect_confluent is enabled, the script identifies lesions that result
from the merger of multiple distinct lesions from the first session using
backwards-mapping. Confluent lesions are relabeled with their ID + 1000 across
all sessions to distinguish them from non-confluent lesions. This allows
researchers to treat merged lesions as a single entity throughout longitudinal
analysis while maintaining their distinct identity.
"""

import argparse
import os

import nibabel as nib
import numpy as np

from scilpy.image.labels import (get_data_as_labels, harmonize_labels,
                                 get_labels_from_mask)
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             assert_headers_compatible)

EPILOG = """
Reference:
    [1] Köhler, Caroline, et al. "Exploring individual multiple sclerosis
    lesion volume change over time: development of an algorithm for the
    analyses of longitudinal quantitative MRI measures."
    NeuroImage: Clinical 21 (2019): 101623.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_images', nargs='+',
                   help='Input file name, in nifti format.')
    p.add_argument('out_dir',
                   help='Output directory.')
    p.add_argument('--max_adjacency', type=float, default=5.0,
                   help='Maximum adjacency distance between lesions for '
                   'them to be considered as the potential match '
                   '[%(default)s].')
    p.add_argument('--min_voxel_overlap', type=int, default=1,
                   help='Minimum number of overlapping voxels between '
                   'lesions for them to be considered as the potential '
                   'match [%(default)s].')

    p.add_argument('--incremental_lesions', action='store_true',
                   help='If lesions files only show new lesions at each '
                        'timepoint, this will merge past timepoints.')
    p.add_argument('--add_pre_lesion_labels', action='store_true',
                   help='Add pre-lesion labels for each new-lesion to all '
                        'sessions before it appears for the first time. '
                        'Pre-lesion labels use negative values of the lesion ID.')
    p.add_argument('--fill_intermediate_missing', action='store_true',
                   help='Fill missing lesions in intermediate sessions. If a '
                        'lesion exists in session N and N+2 but is missing in '
                        'N+1, copy it from session N (previous session). This '
                        'corrects segmentation errors where lesions were '
                        'accidentally not segmented in intermediate timepoints.')
    p.add_argument('--detect_confluent', action='store_true',
                   help='Detect and relabel confluent lesions using backwards-mapping. '
                        'A lesion is classified as confluent if it overlaps with '
                        'multiple distinct lesions from the first session. Confluent '
                        'lesions are relabeled with their ID + 1000.')
    p.add_argument('--min_confluence_overlap', type=int, default=1,
                   help='Minimum number of overlapping voxels to consider a confluence '
                        '[%(default)s].')
    p.add_argument('--debug_mode', action='store_true',
                   help='Add a fake voxel to the corner to ensure consistent '
                        'colors in MI-Brain.')

    add_overwrite_arg(p)
    return p


# -------------------------------------------------------------------------
# NEW FUNCTION: Final harmonization across all sessions
# -------------------------------------------------------------------------
def final_label_harmonization(relabeled_data):
    """
    Ensure that all sessions have consistent labels for the same lesions.

    If a lesion label appears in all but one session and that one session
    disagrees, we correct it to match the consensus.

    Parameters
    ----------
    relabeled_data : list of np.ndarray
        List of 3D label arrays (same shape), one per session.

    Returns
    -------
    corrected_data : list of np.ndarray
        List of harmonized label arrays.
    """
    corrected_data = [data.copy() for data in relabeled_data]
    n_sessions = len(relabeled_data)

    # Combine all lesions to find regions with any lesion across sessions
    union_mask = np.zeros_like(relabeled_data[0], dtype=bool)
    for d in relabeled_data:
        union_mask |= d > 0

    # Iterate over all lesion voxels
    lesion_voxels = np.argwhere(union_mask)

    for voxel in lesion_voxels:
        labels = [data[tuple(voxel)] for data in relabeled_data]
        nonzero_labels = [lbl for lbl in labels if lbl > 0]

        if len(nonzero_labels) <= 1:
            continue  # Only one label, nothing to harmonize

        # Check if majority agrees and one differs
        unique, counts = np.unique(nonzero_labels, return_counts=True)
        if len(unique) > 1:
            # Find the majority label
            majority_label = unique[np.argmax(counts)]

            # If all but one share the same label
            if np.max(counts) >= len(nonzero_labels) - 1:
                for i, lbl in enumerate(labels):
                    if lbl > 0 and lbl != majority_label:
                        corrected_data[i][tuple(voxel)] = majority_label

    return corrected_data


# -------------------------------------------------------------------------
# NEW FUNCTION: Add pre-lesion labels
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
# NEW FUNCTION: Detect confluent lesions
# -------------------------------------------------------------------------
def detect_confluent_lesions(relabeled_data, min_overlap=2):
    """
    Detect confluent lesions using backwards-mapping.
    
    A lesion is considered confluent if it's a single blob in a later session
    that overlaps with multiple distinct lesions from the first session.
    Uses backwards-mapping: for each lesion in session N, check if its footprint
    overlaps with multiple lesions in session 1.
    
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
        Dictionary mapping lesion IDs to the list of original lesion IDs they merged from.
        Format: {lesion_id: [source_id1, source_id2, ...]}
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
            
            # If this lesion overlaps with multiple lesions from first session, it's confluent
            if len(overlap_counts) > 1:
                confluent_lesions[int(lesion_id)] = sorted(overlap_counts.keys())
    
    return confluent_lesions


# -------------------------------------------------------------------------
# NEW FUNCTION: Relabel confluent lesions
# -------------------------------------------------------------------------
def relabel_confluent_lesions(relabeled_data, confluent_lesions):
    """
    Relabel confluent lesions by adding 1000 to their ID across all sessions.
    
    Parameters
    ----------
    relabeled_data : list of np.ndarray
        List of 3D label arrays (same shape), one per session.
    confluent_lesions : dict
        Dictionary mapping lesion IDs to their source lesion IDs.
    
    Returns
    -------
    relabeled_data : list of np.ndarray
        List of label arrays with confluent lesions relabeled.
    """
    if not confluent_lesions:
        return relabeled_data
    
    relabeled_data_out = [data.copy() for data in relabeled_data]
    
    for lesion_id in confluent_lesions.keys():
        new_label = lesion_id + 1000
        
        # Relabel this lesion in all sessions where it appears
        for session_idx in range(len(relabeled_data_out)):
            mask = (relabeled_data_out[session_idx] == lesion_id)
            relabeled_data_out[session_idx][mask] = new_label
    
    return relabeled_data_out


# -------------------------------------------------------------------------
# NEW FUNCTION: Fill intermediate missing lesions
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

    imgs = [nib.load(filename) for filename in args.in_images]
    original_data = [get_data_as_labels(img) for img in imgs]

    masks = []
    if args.incremental_lesions:
        for i, data in enumerate(original_data):
            mask = np.zeros_like(data)
            mask[data > 0] = 1
            masks.append(mask)
            if i > 0:
                new_data = np.sum(masks, axis=0)
                new_data[new_data > 0] = 1
            else:
                new_data = mask
            original_data[i] = get_labels_from_mask(new_data)

    # Initial harmonization (spatial/overlap based)
    relabeled_data = harmonize_labels(original_data,
                                      args.min_voxel_overlap,
                                      max_adjacency=args.max_adjacency)

    # ---------------------------------------------------------------------
    # Final consistency harmonization across all sessions
    # ---------------------------------------------------------------------
    relabeled_data = final_label_harmonization(relabeled_data)

    # ---------------------------------------------------------------------
    # Detect and relabel confluent lesions if requested
    # ---------------------------------------------------------------------
    if args.detect_confluent:
        print("Detecting confluent lesions using backwards-mapping...")
        confluent_lesions = detect_confluent_lesions(relabeled_data, args.min_confluence_overlap)
        if confluent_lesions:
            print(f"Found {len(confluent_lesions)} confluent lesion(s):")
            for lesion_id, sources in confluent_lesions.items():
                print(f"  Lesion {lesion_id} merged from lesions: {sources}")
                print(f"    -> Relabeling to {lesion_id + 1000}")
            relabeled_data = relabel_confluent_lesions(relabeled_data, confluent_lesions)
        else:
            print("No confluent lesions detected.")

    # ---------------------------------------------------------------------
    # Fill intermediate missing lesions if requested
    # ---------------------------------------------------------------------
    if args.fill_intermediate_missing:
        relabeled_data = fill_intermediate_missing_lesions(relabeled_data)

    # ---------------------------------------------------------------------
    # Add pre-lesion labels if requested
    # ---------------------------------------------------------------------
    if args.add_pre_lesion_labels:
        relabeled_data = add_pre_lesion_labels(relabeled_data)

    max_label = np.max(relabeled_data) + 1
    for i, img in enumerate(imgs):
        if args.debug_mode:
            relabeled_data[i][0, 0, 0] = max_label  # To force identical color
        nib.save(nib.Nifti1Image(relabeled_data[i], img.affine),
                 os.path.join(args.out_dir, os.path.basename(args.in_images[i])))


if __name__ == "__main__":
    main()


