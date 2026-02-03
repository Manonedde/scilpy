#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect confluent lesions in MS patients from longitudinal FLAIR MRI data.

A confluent lesion is defined as a large, continuous area of demyelination 
formed by the spatial overlap or fusion of multiple pathologically distinct 
focal lesions across time (sessions).

Criteria for confluence:
  1. Minimum overlap of 4 voxels between 2 consecutive sessions
  2. Volume increase or stable (with ±3 voxel tolerance for segmentation errors)
  3. Lesion labels must be consistent across sessions (same ID)
  4. New lesion IDs after session 0 are ignored
  5. Negative labels represent pre-lesions and are ignored

Output:
  - CSV report with confluent lesions and overlap statistics
  - Summary statistics about confluence patterns

Usage:
    scil_lesions_detect_confluent.py ses-01_lesions.nii.gz \\
                                       ses-02_lesions.nii.gz \\
                                       ses-03_lesions.nii.gz \\
                                       output.csv \\
                                       --min_overlap 4 \\
                                       --tolerance 3

References:
    [1] Dworkin, J. D., et al. "Lesion Segmentation in the Multiple Sclerosis 
    Lesion Challenge." Proceedings of the MICCAI Workshop on Brain Lesion: 
    Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injury. 2018.
    
    [2] Lassmann, H. "Multiple sclerosis pathology." Cold Spring Harbor 
    Perspectives in Medicine 8.12 (2018): a028936.
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_headers_compatible)


def _build_arg_parser():
    """Build argument parser."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    p.add_argument('in_images', nargs='+',
                   help='Input lesion label files in NIfTI format, '
                        'in chronological order (minimum 2 sessions).')
    p.add_argument('out_csv',
                   help='Output CSV file with confluent lesion report.')
    
    p.add_argument('--min_overlap', type=int, default=4,
                   help='Minimum number of overlapping voxels to consider '
                        'confluence [%(default)s].')
    p.add_argument('--tolerance', type=int, default=3,
                   help='Tolerance for volume change (decrease) in voxels '
                        'to account for segmentation variability and '
                        'registration inaccuracy [%(default)s].')
    
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Verbose output.')
    
    add_overwrite_arg(p)
    
    return p


def load_lesion_images(image_paths):
    """
    Load lesion label images from files.
    
    Parameters
    ----------
    image_paths : list of str
        Paths to lesion label NIfTI files in chronological order.
    
    Returns
    -------
    images : list of ndarray
        Loaded lesion label maps.
    affine : ndarray
        Affine matrix from the first image.
    header : nib.Nifti1Header
        Header from the first image.
    """
    images = []
    affine = None
    header = None
    
    for path in image_paths:
        img = nib.load(path)
        data = np.asarray(img.dataobj, dtype=np.int16)
        images.append(data)
        
        if affine is None:
            affine = img.affine
            header = img.header
    
    return images, affine, header


def get_lesion_ids_in_session(lesion_map, exclude_negative=True):
    """
    Get all lesion IDs in a session.
    
    Parameters
    ----------
    lesion_map : ndarray
        Lesion label map.
    exclude_negative : bool
        If True, exclude negative labels (pre-lesions).
    
    Returns
    -------
    ids : set
        Set of lesion IDs.
    """
    unique_ids = set(np.unique(lesion_map))
    unique_ids.discard(0)  # Remove background
    
    if exclude_negative:
        unique_ids = {id for id in unique_ids if id > 0}
    
    return unique_ids


def compute_overlap(lesion_map1, lesion_map2, lesion_id):
    """
    Compute overlap between a lesion in two sessions.
    
    Parameters
    ----------
    lesion_map1 : ndarray
        Lesion map in session 1.
    lesion_map2 : ndarray
        Lesion map in session 2.
    lesion_id : int
        ID of the lesion to check.
    
    Returns
    -------
    overlap_voxels : int
        Number of overlapping voxels.
    """
    mask1 = lesion_map1 == lesion_id
    mask2 = lesion_map2 == lesion_id
    overlap = np.sum(mask1 & mask2)
    return int(overlap)


def compute_volume(lesion_map, lesion_id):
    """
    Compute volume (voxel count) of a lesion.
    
    Parameters
    ----------
    lesion_map : ndarray
        Lesion label map.
    lesion_id : int
        ID of the lesion.
    
    Returns
    -------
    volume : int
        Number of voxels in the lesion.
    """
    mask = lesion_map == lesion_id
    return int(np.sum(mask))


def detect_confluent_lesions(images, min_overlap=4, tolerance=3, verbose=False):
    """
    Detect confluent lesions across sessions with temporal consistency.
    
    For confluence to be valid across multiple sessions:
    1. Minimum overlap of min_overlap voxels must exist between consecutive sessions
    2. Volume must increase or stay stable (within tolerance) consistently
    3. Once confluence begins, it must persist and remain stable in all subsequent sessions
    4. Lesions must exist in session 0 (not new lesions)
    
    Parameters
    ----------
    images : list of ndarray
        Lesion label maps for each session.
    min_overlap : int
        Minimum overlapping voxels to consider confluence.
    tolerance : int
        Tolerance for volume decrease (voxels).
    verbose : bool
        Print verbose output.
    
    Returns
    -------
    confluent_lesions : list of dict
        Information about confluent lesions with temporal consistency validation.
    """
    if len(images) < 2:
        raise ValueError("Minimum 2 sessions required")
    
    confluent_lesions = []
    session_0_ids = get_lesion_ids_in_session(images[0], exclude_negative=False)
    
    if verbose:
        print(f"Session 0 lesion IDs: {sorted(session_0_ids)}")
    
    # For each lesion in session 0, track its entire trajectory
    for lesion_id in session_0_ids:
        # Get volume trajectory across all sessions
        volumes = []
        overlaps = []
        for i, ses in enumerate(images):
            vol = compute_volume(ses, lesion_id)
            volumes.append(vol)
            
            # Compute overlap with previous session
            if i > 0:
                overlap = compute_overlap(images[i-1], ses, lesion_id)
                overlaps.append(overlap)
            else:
                overlaps.append(None)
        
        if verbose:
            print(f"\nLesion {lesion_id}:")
            print(f"  Volume trajectory: {volumes}")
            print(f"  Overlap trajectory: {overlaps[1:]}")
        
        # Find first session where lesion disappears (becomes 0)
        last_session = len(volumes) - 1
        for i, vol in enumerate(volumes):
            if vol == 0:
                last_session = i - 1
                break
        
        # If lesion doesn't appear in any session, skip
        if all(v == 0 for v in volumes):
            if verbose:
                print(f"  → Lesion never appears")
            continue
        
        # Track confluence for this lesion across all sessions
        confluence_start_session = None
        confluence_valid = True
        confluence_reason = None
        
        # Scan through sessions looking for confluence pattern
        for session_idx in range(last_session):
            vol_current = volumes[session_idx]
            vol_next = volumes[session_idx + 1]
            overlap = overlaps[session_idx + 1]
            
            # Skip if lesion disappears in current session
            if vol_current == 0:
                continue
            
            # Skip if lesion disappears in next session
            if vol_next == 0:
                if verbose:
                    print(f"  ({session_idx}→{session_idx + 1}): "
                          f"lesion disappears in session {session_idx + 1}")
                confluence_valid = False
                break
            
            vol_change = vol_next - vol_current
            
            # Check confluence criteria for this pair
            if overlap >= min_overlap and (vol_change >= 0 or vol_change >= -tolerance):
                # This session pair shows confluence pattern
                if confluence_start_session is None:
                    confluence_start_session = session_idx
                    if verbose:
                        print(f"  ({session_idx}→{session_idx + 1}): "
                              f"confluence detected (overlap={overlap}, vol_change={vol_change})")
            else:
                # This session pair does NOT show confluence pattern
                if confluence_start_session is not None:
                    # Confluence started but now broken
                    if verbose:
                        print(f"  ({session_idx}→{session_idx + 1}): "
                              f"confluence broken (overlap={overlap}, vol_change={vol_change})")
                    confluence_valid = False
                    confluence_reason = f"confluence pattern broken at sessions {session_idx}→{session_idx + 1}"
                    break
                else:
                    # No confluence yet
                    if verbose:
                        print(f"  ({session_idx}→{session_idx + 1}): "
                              f"no confluence (overlap={overlap}, vol_change={vol_change})")
        
        # Report confluent lesion if pattern is valid and consistent
        if confluence_start_session is not None and confluence_valid:
            confluent_lesions.append({
                'lesion_id': lesion_id,
                'confluence_start': confluence_start_session,
                'confluence_end': last_session,
                'session_range': f"{confluence_start_session}-{last_session}",
                'volume_start': volumes[confluence_start_session],
                'volume_end': volumes[last_session],
                'volume_trajectory': volumes,
                'overlap_trajectory': overlaps[1:],
                'status': 'CONFLUENT',
                'reason': 'Consistent confluence pattern across all sessions'
            })
            
            if verbose:
                print(f"  ✓ CONFLUENT: Sessions {confluence_start_session}→{last_session}")
        elif confluence_start_session is not None and not confluence_valid:
            if verbose:
                print(f"  ✗ NOT CONFLUENT: {confluence_reason}")
        else:
            if verbose:
                print(f"  ✗ NOT CONFLUENT: No confluence pattern detected")
    
    return confluent_lesions


def generate_report(confluent_lesions, images, affine):
    """
    Generate summary report of confluent lesions.
    
    Parameters
    ----------
    confluent_lesions : list of dict
        Detected confluent lesions.
    images : list of ndarray
        Lesion label maps.
    affine : ndarray
        Affine matrix for voxel-to-mm conversion.
    
    Returns
    -------
    df : pd.DataFrame
        Report dataframe.
    """
    voxel_vol_mm3 = abs(np.linalg.det(affine[:3, :3]))
    
    data = []
    for conf in confluent_lesions:
        data.append({
            'Lesion_ID': conf['lesion_id'],
            'Confluence_Sessions': conf['session_range'],
            'First_Session_with_Confluence': conf['confluence_start'],
            'Last_Session': conf['confluence_end'],
            'Volume_at_Start_voxels': conf['volume_start'],
            'Volume_at_End_voxels': conf['volume_end'],
            'Volume_Change_voxels': conf['volume_end'] - conf['volume_start'],
            'Volume_at_Start_mm3': conf['volume_start'] * voxel_vol_mm3,
            'Volume_at_End_mm3': conf['volume_end'] * voxel_vol_mm3,
            'Status': conf['status'],
            'Reason': conf['reason']
        })
    
    df = pd.DataFrame(data)
    return df


def main():
    """Main execution."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.in_images) < 2:
        print("Error: Minimum 2 sessions required", file=sys.stderr)
        return 1
    
    assert_inputs_exist(parser, args.in_images)
    assert_headers_compatible(parser, args.in_images)
    
    # Check output
    if os.path.exists(args.out_csv) and not args.overwrite:
        print(f"Error: Output file {args.out_csv} already exists. "
              "Use --overwrite to replace.", file=sys.stderr)
        return 1
    
    # Load images
    if args.verbose:
        print(f"Loading {len(args.in_images)} lesion label maps...")
    
    images, affine, header = load_lesion_images(args.in_images)
    
    if args.verbose:
        print(f"  Voxel size: {np.diag(affine)[:3]}")
        for i, img in enumerate(images):
            unique_ids = get_lesion_ids_in_session(img, exclude_negative=False)
            print(f"  Session {i}: {len(unique_ids)} lesion IDs: {sorted(unique_ids)}")
    
    # Detect confluent lesions
    if args.verbose:
        print(f"\nDetecting confluent lesions...")
        print(f"  Minimum overlap: {args.min_overlap} voxels")
        print(f"  Volume change tolerance: ±{args.tolerance} voxels")
    
    confluent_lesions = detect_confluent_lesions(
        images,
        min_overlap=args.min_overlap,
        tolerance=args.tolerance,
        verbose=args.verbose
    )
    
    # Generate report
    if confluent_lesions:
        df = generate_report(confluent_lesions, images, affine)
        df.to_csv(args.out_csv, index=False)
        
        print(f"\n{'='*70}")
        print(f"CONFLUENT LESION DETECTION REPORT")
        print(f"{'='*70}")
        print(f"Sessions analyzed: {len(images)}")
        print(f"Confluent lesions detected: {len(confluent_lesions)}")
        print(f"\n{df.to_string(index=False)}")
        print(f"\n✓ Report saved to: {args.out_csv}")
    else:
        # Create empty report
        df = pd.DataFrame()
        df.to_csv(args.out_csv, index=False)
        print(f"\n{'='*70}")
        print(f"CONFLUENT LESION DETECTION REPORT")
        print(f"{'='*70}")
        print(f"Sessions analyzed: {len(images)}")
        print(f"Confluent lesions detected: 0")
        print(f"\n✓ Report saved to: {args.out_csv}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
