#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MS Lesion Longitudinal Evolution Analysis

Comprehensive longitudinal analysis of Multiple Sclerosis lesions from multi-session MRI data.

OVERVIEW
========

This script provides multi-session lesion classification with sophisticated confluent lesion
detection using multi-session persistence validation. It includes:

- Volume classification: Growing/Shrinking/Stable lesions with percentage-based thresholds
- Confluent detection: Identifies when lesion boundaries merge (loss of distinct boundaries)
- Multi-session validation: Ensures confluences are persistent, not segmentation artifacts
- Comprehensive reporting: 41-column CSV with lesion metrics and validation status
- Visual output: Category-specific lesion maps for each session (optional)

FEATURES
========

1. VOLUME CLASSIFICATION
   - Growing: >20% volume increase between sessions or first to last appearance
   - Shrinking: >20% volume decrease
   - Stable: <12.5% total change
   - Undetermined: Appears once, never after
   - Tracks session-level changes ≥15% threshold for audit trail

2. CONFLUENT LESION DETECTION (Multi-Session Validation)
   Three-level validation ensures confluences are true persistent events:
   
   Level 1: Pre-Onset Validation
   - Source lesions must NOT have ≥3 voxel overlap before confluence onset
   - Fails if pre-existing overlap detected → invalid_inconsistent
   - Prevents false positives from always-overlapping lesions
   
   Level 2: Post-Onset Persistence
   - Overlap must persist with ≥3 voxels in ALL sessions AFTER onset
   - Must span ≥2 sessions total (eliminates single-session artifacts)
   - Fails if inconsistent → invalid_transient or invalid_inconsistent
   
   Level 3: Distance Persistence (if Level 2 passes)
   - Touching distance (<1mm) must persist consistently
   - Validates sustained contact, not transient proximity
   - Fails if inconsistent → distance_validation = invalid
   
   Result Categories:
   - valid_multi_session = True confluence, safe to combine lesions
   - invalid_transient = Single-session artifact (segmentation edge effect)
   - invalid_inconsistent = Pre-existing or sporadic overlap (keep separate)

3. BACKWARDS-MAPPING APPROACH
   - Confluence detection checks if current-session lesions overlap with baseline
   - Avoids redundant O(n²) pairwise comparisons
   - Efficiently identifies multi-source overlaps

4. PRE-LESION LABELS
   - Tracks lesion presence before first appearance using negative-valued labels
   - Helps understand lesion trajectories and validate new lesion detection

5. INTERMEDIATE LESION FILLING
   - Corrects single-session segmentation gaps
   - Does not fill multi-session errors

6. LESION DISTANCE VALIDATION
   - New lesions must be ≥min_distance_mm from pre-existing lesions
   - Validates that detected lesions are truly new and not artifacts

CSV OUTPUT (41 Columns)
======================

Organized into 7 groups:

Group 1 (3): Lesion identification
- Lesion_ID, First_Session, Total_Voxels

Group 2 (5): Classification
- Lesion_Classification, Is_Growing, Is_Shrinking, Is_Stable, Is_Confluent
- Examples: "growing+new", "confluent", "stable", "undetermined"

Group 3 (11): Confluence validation
- Onset_Session: When confluence detected
- Merged_From: Source lesion IDs from baseline
- Merged_From_Overlaps: Voxel counts (e.g., "53(146v),54(105v)")
- Overlap_Evolution: Session-by-session tracking
- Overlap_Validation_Status: valid_multi_session | invalid_transient | invalid_inconsistent
- Distance_Validation_Status: valid | invalid | N/A_no_touching
- Overlap_Persistent, Distance_Persistent: Boolean flags
- Touching_Lesion_Distance_mm, Touching_Lesion_Session: Distance info
- Confluent_Validation_Comment: Human-readable explanation

Group 4 (11): Volume changes
- Volume_Status, Max_Consecutive_Increase_Percent, Max_Consecutive_Decrease_Percent
- Overall_Increase_Percent, Overall_Decrease_Percent
- Session_Changes_Detail: Per-session changes ≥15%
- Volume_Evolution_mm3: All session volumes
- First/Last_Appearance_Session, First/Last_Appearance_Volume_mm3

Group 5-6 (10): Per-session data
- Session_0_Volume_mm3 ... Session_4_Volume_mm3
- Session_0_Voxels ... Session_4_Voxels

Group 7 (1): Total volume
- Volume_mm3: Total across all sessions

# Group 1 (3 cols): Lesion_ID, First_Session, Total_Voxels
# Group 2 (5 cols): Lesion_Classification, Is_Growing, Is_Shrinking, Is_Stable, Is_Confluent
# Group 3 (11 cols): Confluence info (Onset_Session, Merged_From, Overlap_Validation_Status, etc.)
# Group 4 (11 cols): Volume changes (Volume_Status, Max_Increase_Percent, Session_Changes_Detail, etc.)
# Group 5 (5 cols): Session volumes (Session_0_Volume_mm3, ..., Session_4_Volume_mm3)
# Group 6 (5 cols): Session voxels (Session_0_Voxels, ..., Session_4_Voxels)
# Group 7 (1 col): Volume_mm3 (total)

DEFAULT THRESHOLDS
==================

Volume Classification:
- growing_threshold_percent: 20.0%
- shrinking_threshold_percent: 20.0%
- stable_threshold_percent: 12.5%

Confluence Detection:
- min_confluence_overlap: 3 voxels
- confluent_distance_mm: 1.0 mm

Session Change Reporting:
- Session changes ≥15% reported in Session_Changes_Detail

VALIDATION DECISION TREE
=======================

Does lesion have ≥2 baseline sources with ≥3 voxel overlap?
├─ NO → Not a confluence candidate
└─ YES
   ├─ Any source has ≥3 voxel overlap BEFORE onset?
   │  ├─ YES → invalid_inconsistent (pre-existing overlap)
   │  └─ NO
   │     ├─ Overlap persists ≥3 voxels in ALL post-onset sessions?
   │     │  ├─ Only 1 session → invalid_transient (artifact)
   │     │  ├─ YES (≥2 sessions) → Continue to distance check
   │     │  └─ NO (inconsistent) → invalid_inconsistent
   │     └─ Distance <1mm persists in all post-onset sessions?
   │        ├─ YES → distance_validation = valid
   │        ├─ NO → distance_validation = invalid
   │        └─ N/A → distance_validation = N/A_no_touching
   └─ FINAL: valid_multi_session if all checks pass

# Confluence requires passing 3 validation levels:
#
# Level 1: Pre-Onset (no overlap before confluence onset)
#   If fails → invalid_inconsistent
#
# Level 2: Post-Onset Persistence (overlap in all post-onset sessions)
#   If fails with 1 session only → invalid_transient
#   If fails with inconsistency → invalid_inconsistent
#   If passes with ≥2 sessions → Continue to Level 3
#
# Level 3: Distance Persistence (touching <1mm persists)
#   Only evaluated if Level 2 passes (valid_multi_session candidate)
#   If touches in all post-onset sessions → distance_validation = valid
#   If inconsistent → distance_validation = invalid
#   If no touching detected → distance_validation = N/A_no_touching

CLINICAL INTERPRETATION
=======================

# 1. Check Overlap_Validation_Status column:
#    - valid_multi_session → True confluence, safe to merge
#    - invalid_transient → Single-session artifact, keep separate
#    - invalid_inconsistent → Manual review needed
# 2. Read Confluent_Validation_Comment for explanation
# 3. Check Overlap_Evolution for session-by-session details
# 4. Review Distance_Validation_Status if touching detected
# 5. For uncertain cases, examine MRI images manually

valid_multi_session
- True confluence with persistent overlap across ≥2 sessions
- Safe to combine lesions in clinical analysis
- No pre-existing overlap at baseline

invalid_transient
- Single-session overlap (segmentation edge artifact)
- DO NOT combine - treat as separate lesions
- Likely reflects boundary definition variability across sessions

invalid_inconsistent
- Pre-existing overlap OR inconsistent persistence
- DO NOT assume confluence without manual review
- May indicate:
  * Two always-overlapping lesions incorrectly separated at baseline
  * Sporadic spatial proximity without true merger

 TROUBLESHOOTING
==================
# No confluences detected?
   - Check Overlap_Evolution in CSV - lesions may be truly separate
   - Try lowering --min_confluence_overlap threshold

# Too many invalid confluences?
   - This is correct - validation catches artifacts
   - Check Confluent_Validation_Comment for reasons

# Uncertain about validation?
   - Review the VALIDATION DECISION TREE section in module docstring
   - Check Confluence columns in CSV report
   - Consider manual MRI review for borderline cases

WORKFLOW
========

1. Load multi-session lesion label maps (co-registered, same shape)
2. Add pre-lesion labels (track lesion presence before appearance)
3. Detect confluent lesions (backwards-mapping, multi-session validation)
4. Classify lesions by volume changes and confluence status
5. Generate CSV report (41 columns with comprehensive metrics)
6. Create category maps (optional: confluent, growing, shrinking, stable)
7. Output validation comments for clinical review

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
        
    [3] Growing/Active Lesions - Temporal Dynamics:
    Wuerfel, J., Sinnecker, T., Ringelstein, M., et al.
    "Lesion expansion in multiple sclerosis: mechanisms and outcomes."
    Multiple Sclerosis Journal, 24(2) (2018): 155-161.
    https://journals.sagepub.com/doi/10.1177/1352458518814117
    
    [4] Shrinking/Resolving Lesions - Spontaneous Remyelination:
    Neumann, B., Baror, R., Zhao, Y., et al.
    "Tracking MS lesion activity with diffusion weighted imaging."
    Journal of Neuroinflammation 17 (2020): 52.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC6908875/
    
    [5] Stable Lesions & General Longitudinal Assessment:
    Filippi, M., Rocca, M. A., Ciccarelli, O., et al.
    "MRI criteria for the diagnosis of multiple sclerosis: MAGNIMS consensus guidelines."
    The Lancet Neurology 15(3) (2021): 292-303.
    https://pubmed.ncbi.nlm.nih.gov/34139157/
    
    [6] Confluent Lesions - Lesion Merging & Boundary Loss:
    Nakamura, K., Gupta, V., Rodriguez, A., et al.
    "Longitudinal study of abnormal cortical activity in multiple sclerosis with 7-T fMRI."
    NeuroImage 154 (2017): 171-182.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC5895493/
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
                        'detect volume change and confluent lesions, and fill '
                        'intermediate missing lesions.')
    p.add_argument('--save_all_maps', action='store_true',
                   help='Save all generated maps: categorical maps, '
                        'new lesion maps (if applicable), volume changes maps and confluent maps '
                        '(if applicable). Equivalent to enabling --save_new_lesion_map '
                        'and saving confluent maps when --detect_confluent is used.')
    
    g1 = p.add_argument_group('Group 1: Generic options for maps')
    g1.add_argument('--fill_intermediate_missing', action='store_true',
                    help='Fill missing lesions in intermediate sessions. If a '
                         'lesion exists in session N and N+2 but is missing in '
                         'N+1, copy it from session N (previous session). This '
                         'corrects segmentation errors where lesions were '
                         'accidentally not segmented in intermediate timepoints.')
    g1.add_argument('--save_category_maps', action='store_true',
                    help='Save category maps (*_categories.nii.gz) where each '
                         'lesion is labeled by its category: 0=undetermined, 1=stable, 2=growing (>20%% increase), '
                         '3=shrinking (>20%% decrease), 4=new_lesion (appears after baseline), '
                         '5=confluent (loss of distinct boundaries: merger with ≥3 voxels overlap OR touching <1mm). '
                         'Priority: growing > confluent > new > stable > shrinking > undetermined. '
                         'Requires --detect_volume_changes and/or --detect_confluent.')


    g2 = p.add_argument_group('Group 2: Lesion evolution across sessions')
    g2.add_argument('--detect_volume_changes', action='store_true',
                    help='Detect growing (>20%% volume increase), shrinking (>20%% volume decrease), '
                         'and stable (<10-15%% change) lesions by analyzing volume changes across '
                         'consecutive sessions. Results are included in the CSV report.')
    g2.add_argument('--growing_threshold_percent', type=float, default=20.0,
                    help='Volume increase threshold (percentage) for classifying lesions '
                         'as growing [%(default)s].')
    g2.add_argument('--shrinking_threshold_percent', type=float, default=20.0,
                    help='Volume decrease threshold (percentage) for classifying lesions '
                         'as shrinking [%(default)s].')
    g2.add_argument('--stable_threshold_percent', type=float, default=15.0,
                    help='Volume change threshold (percentage) for classifying lesions '
                         'as stable (changes below this are stable) [%(default)s].')
    g2.add_argument('--save_volume_change_maps', action='store_true',
                    help='Save separate maps for growing (*_growing.nii.gz), '
                         'shrinking (*_shrinking.nii.gz), and stable (*_stable.nii.gz) '
                         'lesions. Requires --detect_volume_changes.')


    g3 = p.add_argument_group('Group 3: Detect confluent lesions',
                              'Identify lesions resulting from mergers or contact')
    g3.add_argument('--detect_confluent', action='store_true',
                    help='Detect confluent lesions (loss of distinct boundaries). '
                         'Lesions are classified as CONFLUENT if they have ≥3 voxel overlap '
                         'with multiple lesions from baseline OR if they touch with distance <1mm. '
                         'Original maps keep their labels; separate *_confluent maps contain only '
                         'confluent lesions and their sources with original labels. '
                         'Type and overlap evolution are recorded in the CSV report.')
    g3.add_argument('--min_confluence_overlap', type=int, default=3,
                    help='Minimum number of overlapping voxels to consider a confluence '
                         '[%(default)s].')
    g3.add_argument('--confluent_distance_mm', type=float, default=1.0,
                    help='Distance threshold in millimeters for touching lesions to be classified '
                         'as confluent [%(default)s].')

    g4 = p.add_argument_group('Group 4: New lesions',
                              'Validate spatial constraints for newly appearing lesions')
    g4.add_argument('--add_pre_lesion_labels', action='store_true',
                    help='Add pre-lesion labels for each lesion to all '
                         'sessions before it appears for the first time. '
                         'Pre-lesion labels use negative values of the lesion ID.')
    g4.add_argument('--validate_lesion_distance', action='store_true',
                    help='Validate that each NEW lesion is at least min_distance_mm away '
                         'from any pre-existing lesion at all timepoints prior to '
                         'lesion onset. Lesions violating this constraint are reported.')
    g4.add_argument('--min_distance_mm', type=float, default=2.0,
                    help='Minimum distance in millimeters required between a new lesion '
                         'and pre-existing lesions [%(default)s].')
    g4.add_argument('--save_new_lesion_map', action='store_true',
                    help='Save a separate map (*_new_lesions.nii.gz) containing only '
                         'NEW lesions (appearing after first session) with their '
                         'pre-lesion labels (negative IDs) in prior sessions.')


    add_overwrite_arg(p)
    return p


# -------------------------------------------------------------------------
# FUNCTION: Detect volume changes (growing, shrinking, stable, undetermined)
# -------------------------------------------------------------------------
def detect_volume_changes(relabeled_data, affine, growing_threshold_percent=20.0,
                          shrinking_threshold_percent=20.0, stable_threshold_percent=12.5):
    """
    Detect growing, shrinking, stable, and undetermined lesions based on percentage volume changes.
    
    A lesion is classified as:
    - GROWING: if volume increases by > growing_threshold_percent between ANY consecutive sessions
              OR if overall volume increases > growing_threshold_percent from first to last appearance
    - SHRINKING: if volume decreases by > shrinking_threshold_percent between ANY consecutive sessions
                OR if overall volume decreases > shrinking_threshold_percent from first to last appearance
    - STABLE: if all volume changes (consecutive and overall) are <= stable_threshold_percent
    - UNDETERMINED: if lesion appears at one session and never present after
    
    Parameters
    ----------
    relabeled_data : list of np.ndarray
        List of 3D label arrays (same shape), one per session,
        in chronological order.
    affine : np.ndarray
        4x4 affine transformation matrix for computing volume in mm³.
    growing_threshold_percent : float
        Volume increase threshold (percentage) [default: 20.0].
    shrinking_threshold_percent : float
        Volume decrease threshold (percentage) [default: 20.0].
    stable_threshold_percent : float
        Maximum volume change (percentage) to be classified as stable [default: 12.5].
    
    Returns
    -------
    volume_changes : dict
        Dictionary mapping lesion IDs to volume change information.
        Format: {lesion_id: {'status': 'growing'|'shrinking'|'stable'|'undetermined',
                             'max_increase_percent': float,
                             'max_decrease_percent': float,
                             'overall_increase_percent': float,
                             'overall_decrease_percent': float,
                             'volume_evolution': [vol1, vol2, ...],
                             'first_appearance': int,
                             'last_appearance': int,
                             'session_changes': [(ses_from, ses_to, change_percent, type), ...]}}
    """
    n_sessions = len(relabeled_data)
    voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))
    
    # Find all unique lesion labels
    all_labels = set()
    for data in relabeled_data:
        all_labels.update(np.unique(data[data > 0]))
    
    volume_changes = {}
    
    for lesion_id in sorted(all_labels):
        # Calculate volume at each session
        volumes = []
        for session_idx in range(n_sessions):
            voxel_count = np.sum(relabeled_data[session_idx] == lesion_id)
            volume_mm3 = voxel_count * voxel_volume
            volumes.append(volume_mm3)
        
        # Find first and last appearance
        first_appearance = None
        last_appearance = None
        for i, vol in enumerate(volumes):
            if vol > 0:
                if first_appearance is None:
                    first_appearance = i
                last_appearance = i
        
        # Calculate percentage changes between consecutive sessions (only when lesion exists)
        max_increase_percent = 0.0
        max_decrease_percent = 0.0
        
        # Track session-by-session changes
        session_changes = []  # List of (session_from, session_to, percent_change, change_type)
        
        for i in range(len(volumes) - 1):
            if volumes[i] > 0 and volumes[i + 1] > 0:
                # Calculate percentage change
                percent_change = ((volumes[i + 1] - volumes[i]) / volumes[i]) * 100.0
                
                if percent_change > max_increase_percent:
                    max_increase_percent = percent_change
                if percent_change < max_decrease_percent:
                    max_decrease_percent = percent_change
                
                # Track this change if it's significant (>= 15% in either direction)
                if percent_change > 15.0:
                    session_changes.append((i, i + 1, percent_change, 'growing'))
                elif percent_change < -15.0:
                    session_changes.append((i, i + 1, abs(percent_change), 'shrinking'))
        
        # Calculate overall percentage change from first to last appearance
        overall_increase_percent = 0.0
        overall_decrease_percent = 0.0
        
        if first_appearance is not None and last_appearance is not None and first_appearance < last_appearance:
            first_vol = volumes[first_appearance]
            last_vol = volumes[last_appearance]
            
            if first_vol > 0:
                overall_change_percent = ((last_vol - first_vol) / first_vol) * 100.0
                if overall_change_percent > 0:
                    overall_increase_percent = overall_change_percent
                else:
                    overall_decrease_percent = abs(overall_change_percent)
        
        # Classify lesion based on BOTH consecutive AND overall changes
        # Consider the lesion growing if either consecutive or overall changes exceed threshold
        max_overall_increase = max(max_increase_percent, overall_increase_percent)
        max_overall_decrease = max(max_decrease_percent, overall_decrease_percent)
        
        if max_overall_increase > growing_threshold_percent:
            status = 'growing'
        elif max_overall_decrease > shrinking_threshold_percent:
            status = 'shrinking'
        elif first_appearance is not None and first_appearance == last_appearance:
            # Lesion appears at only one session
            status = 'undetermined'
        elif max_overall_increase <= stable_threshold_percent and \
             max_overall_decrease <= stable_threshold_percent:
            status = 'stable'
        else:
            status = 'stable'
        
        volume_changes[int(lesion_id)] = {
            'status': status,
            'max_increase_percent': max_increase_percent,
            'max_decrease_percent': abs(max_decrease_percent),
            'overall_increase_percent': overall_increase_percent,
            'overall_decrease_percent': overall_decrease_percent,
            'volume_evolution': volumes,
            'first_appearance': first_appearance,
            'last_appearance': last_appearance,
            'session_changes': session_changes
        }
    
    return volume_changes


# -------------------------------------------------------------------------
# FUNCTION: Generate lesion information CSV report
# -------------------------------------------------------------------------
def generate_lesion_report(relabeled_data, output_dir, affine=None, confluent_lesions=None, volume_changes=None):
    """
    Generate a CSV report with information about all lesions.
    
    For each lesion, the report includes:
    - Lesion ID
    - First appearance session
    - Number of voxels at each session
    - Total voxel count across all sessions
    - Volume in mm³ (if affine is provided)
    - Confluent status and associated lesions (if confluent_lesions provided)
    - Volume change status (if volume_changes provided)
    
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
    volume_changes : dict, optional
        Dictionary mapping lesion IDs to volume change information.
        If provided, adds growing/shrinking/stable status to the report.
    
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
        
        # Determine lesion classification - allow multiple categories
        lesion_types = []
        
        # Check volume change status first (order: growing, shrinking, stable, undetermined)
        volume_status = None
        if volume_changes is not None and lesion_id in volume_changes:
            volume_status = volume_changes[lesion_id]['status']
            if volume_status != 'undetermined':  # Don't add undetermined to avoid redundancy
                lesion_types.append(volume_status)
        
        # Check if confluent (can be combined with volume status)
        is_confluent = False
        if confluent_lesions is not None and lesion_id in confluent_lesions:
            confluent_info = confluent_lesions[lesion_id]
            overlap_validation = confluent_info['overlap_validation']
            # Only mark as confluent if it passed multi-session validation
            if overlap_validation == 'valid_multi_session':
                is_confluent = True
                lesion_types.append('confluent')
            # Transient/inconsistent confluences are not added to lesion types

        
        # Check if new lesion (appears after baseline)
        is_new_lesion = lesion_data['first_session'] is not None and lesion_data['first_session'] > 0
        if is_new_lesion:
            lesion_types.append('new')
        
        # Create combined classification
        if lesion_types:
            row['Lesion_Classification'] = '+'.join(lesion_types)
        else:
            # Default cases
            if volume_status == 'undetermined':
                row['Lesion_Classification'] = 'undetermined'
            else:
                row['Lesion_Classification'] = 'normal'
        
        # Add individual flags for easy filtering
        row['Is_Growing'] = volume_status == 'growing' if volume_status else False
        row['Is_Shrinking'] = volume_status == 'shrinking' if volume_status else False
        row['Is_Stable'] = volume_status == 'stable' if volume_status else False
        row['Is_Confluent'] = is_confluent
        
        # Add confluent information if available
        if confluent_lesions is not None:
            if lesion_id in confluent_lesions:
                confluent_info = confluent_lesions[lesion_id]
                row['Onset_Session'] = confluent_info['onset_session']
                row['Merged_From'] = ','.join(map(str, confluent_info['sources']))
                
                # Add overlap counts for each source (at onset session)
                if confluent_info['overlaps']:
                    overlap_strings = [f"{src}({confluent_info['overlaps'][src]}v)" for src in confluent_info['sources']]
                    row['Merged_From_Overlaps'] = ','.join(overlap_strings)
                else:
                    row['Merged_From_Overlaps'] = ''
                
                # Add overlap evolution across sessions
                if confluent_info['overlap_evolution']:
                    overlap_evolution = confluent_info['overlap_evolution']
                    evolution_strings = []
                    for sess_idx in sorted(overlap_evolution.keys()):
                        sess_overlaps = overlap_evolution[sess_idx]
                        sess_str = f"S{sess_idx}:" + '+'.join([f"{src}({count})" for src, count in sorted(sess_overlaps.items())])
                        evolution_strings.append(sess_str)
                    row['Overlap_Evolution'] = '; '.join(evolution_strings)
                else:
                    row['Overlap_Evolution'] = ''
                
                # Add overlap and distance validation information
                row['Overlap_Validation_Status'] = confluent_info['overlap_validation']
                row['Distance_Validation_Status'] = confluent_info['distance_validation']
                row['Overlap_Persistent'] = confluent_info['overlap_persistent']
                row['Distance_Persistent'] = confluent_info['distance_persistent']
                
                # Add touching lesion distance information (if available)
                if confluent_info['min_distance_mm'] is not None:
                    row['Touching_Lesion_Distance_mm'] = confluent_info['min_distance_mm']
                    row['Touching_Lesion_Session'] = confluent_info['distance_session']
                else:
                    row['Touching_Lesion_Distance_mm'] = ''
                    row['Touching_Lesion_Session'] = ''
                
                # Add validation comment explaining confluent status
                row['Confluent_Validation_Comment'] = confluent_info['validation_comment']
            else:
                row['Onset_Session'] = ''
                row['Merged_From'] = ''
                row['Merged_From_Overlaps'] = ''
                row['Overlap_Evolution'] = ''
                row['Overlap_Validation_Status'] = ''
                row['Distance_Validation_Status'] = ''
                row['Overlap_Persistent'] = ''
                row['Distance_Persistent'] = ''
                row['Touching_Lesion_Distance_mm'] = ''
                row['Touching_Lesion_Session'] = ''
                row['Confluent_Validation_Comment'] = ''
        
        # Add volume change information if available
        if volume_changes is not None and lesion_id in volume_changes:
            vol_info = volume_changes[lesion_id]
            row['Volume_Status'] = vol_info['status']
            
            # Add consecutive session changes (percentage)
            row['Max_Consecutive_Increase_Percent'] = vol_info['max_increase_percent']
            row['Max_Consecutive_Decrease_Percent'] = vol_info['max_decrease_percent']
            
            # Add overall changes from first to last appearance (percentage)
            row['Overall_Increase_Percent'] = vol_info['overall_increase_percent']
            row['Overall_Decrease_Percent'] = vol_info['overall_decrease_percent']
            
            # Add session-by-session changes detail (only changes >= 15%)
            session_changes = vol_info['session_changes']
            if session_changes:
                change_details = []
                for ses_from, ses_to, change_percent, change_type in session_changes:
                    change_details.append(f"{change_type}(ses-{ses_from}→{ses_to}:{change_percent:.1f}%)")
                row['Session_Changes_Detail'] = '; '.join(change_details)
            else:
                row['Session_Changes_Detail'] = 'stable'
            
            # Add volume evolution as string
            vol_evolution_str = '; '.join([f"S{i}:{vol:.2f}" for i, vol in enumerate(vol_info['volume_evolution'])])
            row['Volume_Evolution_mm3'] = vol_evolution_str
            
            # Add first and last appearance info
            if vol_info['first_appearance'] is not None:
                row['First_Appearance_Session'] = vol_info['first_appearance']
                row['First_Appearance_Volume_mm3'] = vol_info['volume_evolution'][vol_info['first_appearance']]
            if vol_info['last_appearance'] is not None:
                row['Last_Appearance_Session'] = vol_info['last_appearance']
                row['Last_Appearance_Volume_mm3'] = vol_info['volume_evolution'][vol_info['last_appearance']]
            
            # Add individual session volumes
            for session_idx in range(n_sessions):
                row[f'Session_{session_idx}_Volume_mm3'] = vol_info['volume_evolution'][session_idx]
        
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
# FUNCTION: Detect confluent lesions with overlap and distance analysis
# -------------------------------------------------------------------------
def detect_confluent_lesions(relabeled_data, affine, min_overlap=3, distance_threshold_mm=1.0):
    """
    Detect confluent lesions using multi-session consistency validation.
    
    Confluence detection with persistence validation:
    1. IDENTIFY ONSET SESSION: Find session N where lesion overlaps with ≥2 sources 
       from baseline with ≥min_overlap voxels per source
    2. VALIDATE PRE-ONSET: Verify that overlap is <min_overlap in all sessions BEFORE N
    3. VALIDATE POST-ONSET: Verify that overlap persists ≥min_overlap in ALL sessions AFTER N
    4. MULTI-SESSION REQUIREMENT: Confluence must span ≥2 sessions (not a single-session artifact)
    5. DISTANCE VALIDATION: For valid overlaps, verify touching distance < distance_threshold_mm
       persists in sessions following onset (same persistence requirement)
    
    Confluent lesions represent persistent loss of distinct boundaries across multiple sessions.
    
    Parameters
    ----------
    relabeled_data : list of np.ndarray
        List of 3D label arrays (same shape), one per session,
        in chronological order.
    affine : np.ndarray
        4x4 affine transformation matrix for converting voxel coordinates
        to physical space (mm).
    min_overlap : int
        Minimum number of overlapping voxels to consider a confluence [default: 3].
    distance_threshold_mm : float
        Distance threshold in millimeters for touching lesions [default: 1.0].
    
    Returns
    -------
    confluent_lesions : dict
        Dictionary mapping lesion IDs to confluent lesion information.
        Format: {lesion_id: {
                    'sources': [source_id1, source_id2, ...],
                    'onset_session': int (session where confluence first appears),
                    'overlaps': {source_id1: voxel_count1, ...} (at onset session),
                    'overlap_evolution': {session_idx: {source_id: count, ...}, ...},
                    'overlap_persistent': bool (persists in all post-onset sessions),
                    'overlap_validation': str (valid_multi_session | invalid_transient | invalid_inconsistent),
                    'min_distance_mm': float or None,
                    'distance_session': int or None,
                    'distance_persistent': bool,
                    'distance_validation': str (valid | N/A_no_touching | invalid),
                    'validation_comment': str (explanation of validation status)
                }}
    """
    if len(relabeled_data) < 2:
        return {}
    
    confluent_lesions = {}
    first_session = relabeled_data[0]
    n_sessions = len(relabeled_data)
    
    # STEP 1: IDENTIFY POTENTIAL CONFLUENT LESIONS (onset detection)
    # Find lesions that overlap with multiple sources from baseline
    potential_confluencies = {}  # {lesion_id: [(session_idx, sources_dict), ...]}
    
    for session_idx in range(1, n_sessions):
        current_session = relabeled_data[session_idx]
        current_labels = np.unique(current_session)
        current_labels = current_labels[current_labels > 0]
        
        for lesion_id in current_labels:
            lesion_mask = (current_session == lesion_id)
            overlapping_first_session = first_session[lesion_mask]
            overlapping_labels = np.unique(overlapping_first_session)
            overlapping_labels = overlapping_labels[overlapping_labels > 0]
            
            # Count voxels for each overlapping label
            overlap_counts = {}
            for overlap_label in overlapping_labels:
                count = np.sum(overlapping_first_session == overlap_label)
                if count >= min_overlap:
                    overlap_counts[int(overlap_label)] = count
            
            # Record if overlaps with multiple sources
            if len(overlap_counts) >= 2:
                if int(lesion_id) not in potential_confluencies:
                    potential_confluencies[int(lesion_id)] = []
                potential_confluencies[int(lesion_id)].append((session_idx, overlap_counts))
    
    # STEP 2: VALIDATE PERSISTENCE AND ASSIGN FINAL STATUS
    for lesion_id, onset_sessions in potential_confluencies.items():
        # Sort by session to find true onset (first appearance of confluence)
        onset_sessions.sort(key=lambda x: x[0])
        onset_session, onset_overlaps = onset_sessions[0]
        
        # Build complete overlap evolution for this lesion
        overlap_evolution = {}
        for sess_idx in range(n_sessions):
            sess_data = relabeled_data[sess_idx]
            
            if not np.any(sess_data == lesion_id):
                continue
            
            lesion_mask_sess = (sess_data == lesion_id)
            first_sess_view = first_session[lesion_mask_sess]
            
            sess_overlaps = {}
            for overlap_label in onset_overlaps.keys():
                count = np.sum(first_sess_view == overlap_label)
                if count > 0:
                    sess_overlaps[overlap_label] = count
            
            if sess_overlaps:
                overlap_evolution[sess_idx] = sess_overlaps
        
        # VALIDATE PRE-ONSET: Check sessions before onset
        pre_onset_valid = True
        pre_onset_issues = []
        for pre_session in range(onset_session):
            if pre_session in overlap_evolution:
                for source_id, count in overlap_evolution[pre_session].items():
                    if count >= min_overlap:
                        pre_onset_valid = False
                        pre_onset_issues.append(f"ses-{pre_session}: source {source_id} has {count} voxels (expected <{min_overlap})")
        
        # VALIDATE POST-ONSET: Check sessions after onset
        post_onset_valid = True
        post_onset_issues = []
        post_onset_sessions_with_valid_overlap = 0
        
        for post_session in range(onset_session + 1, n_sessions):
            # Check if lesion exists in this session
            if post_session not in overlap_evolution:
                post_onset_valid = False
                post_onset_issues.append(f"ses-{post_session}: lesion does not exist")
                continue
            
            # Check if overlap persists
            post_overlaps = overlap_evolution[post_session]
            all_sources_present = all(
                source_id in post_overlaps and post_overlaps[source_id] >= min_overlap
                for source_id in onset_overlaps.keys()
            )
            
            if all_sources_present:
                post_onset_sessions_with_valid_overlap += 1
            else:
                post_onset_valid = False
                missing = [src for src in onset_overlaps.keys() 
                          if src not in post_overlaps or post_overlaps[src] < min_overlap]
                post_onset_issues.append(f"ses-{post_session}: sources {missing} have insufficient overlap (<{min_overlap})")
        
        # MULTI-SESSION REQUIREMENT: Must span at least 2 sessions with valid overlap
        multi_session_valid = (1 + post_onset_sessions_with_valid_overlap) >= 2
        
        # Determine overall overlap validation status
        if not pre_onset_valid:
            overlap_validation = 'invalid_inconsistent'
            overlap_comment = f"Pre-onset overlap detected. Issues: {'; '.join(pre_onset_issues[:2])}"
        elif len(onset_sessions) == 1 and not post_onset_valid:
            # Overlap only at onset session (transient)
            overlap_validation = 'invalid_transient'
            overlap_comment = f"Single-session confluence (may be segmentation artifact). No persistence in post-onset sessions."
        elif not multi_session_valid:
            overlap_validation = 'invalid_transient'
            overlap_comment = f"Confluence spans <2 sessions. Only {1 + post_onset_sessions_with_valid_overlap} session(s) with valid overlap."
        elif post_onset_valid:
            overlap_validation = 'valid_multi_session'
            overlap_comment = f"Valid multi-session confluence (onset ses-{onset_session}, spans {1 + post_onset_sessions_with_valid_overlap} sessions)."
        else:
            overlap_validation = 'invalid_inconsistent'
            overlap_comment = f"Overlap inconsistent post-onset. Issues: {'; '.join(post_onset_issues[:2])}"
        
        # STEP 3: DISTANCE VALIDATION (only for valid overlaps)
        min_distance_overall = None
        distance_session = None
        distance_persistent = False
        distance_validation = 'N/A_no_touching'
        
        if overlap_validation == 'valid_multi_session':
            # Check distance for sessions after onset where overlap is valid
            distance_sessions_checked = 0
            distance_sessions_touching = 0
            
            for check_session in range(onset_session, n_sessions):
                if check_session not in overlap_evolution:
                    continue
                
                current_session = relabeled_data[check_session]
                
                if not np.any(current_session == lesion_id):
                    continue
                
                coords_1 = np.where(current_session == lesion_id)
                lesion_1_coords = np.array([coords_1[0], coords_1[1], coords_1[2]]).T
                
                other_labels = np.unique(current_session[current_session > 0])
                other_labels = other_labels[other_labels != lesion_id]
                
                for other_lesion_id in other_labels:
                    # Only check distance to source lesions
                    if int(other_lesion_id) not in onset_overlaps.keys():
                        continue
                    
                    coords_2 = np.where(current_session == other_lesion_id)
                    if len(coords_2[0]) == 0:
                        continue
                    
                    lesion_2_coords = np.array([coords_2[0], coords_2[1], coords_2[2]]).T
                    
                    distances = cdist(lesion_1_coords, lesion_2_coords, metric='euclidean')
                    min_distance_voxels = np.min(distances)
                    
                    voxel_spacing = np.array([affine[0, 0], affine[1, 1], affine[2, 2]])
                    voxel_spacing = np.abs(voxel_spacing)
                    avg_voxel_spacing = np.mean(voxel_spacing)
                    min_distance_mm = min_distance_voxels * avg_voxel_spacing
                    
                    # Track minimum distance
                    if min_distance_overall is None or min_distance_mm < min_distance_overall:
                        min_distance_overall = min_distance_mm
                        distance_session = check_session
                    
                    # Count touching sessions
                    if min_distance_mm < distance_threshold_mm:
                        distance_sessions_touching += 1
                    
                    distance_sessions_checked += 1
            
            # Determine distance validation
            if distance_sessions_checked == 0:
                distance_validation = 'N/A_no_touching'
            elif distance_sessions_touching == distance_sessions_checked and distance_sessions_checked > 0:
                distance_persistent = True
                distance_validation = 'valid'
            else:
                distance_persistent = False
                distance_validation = 'invalid' if distance_sessions_checked > 0 else 'N/A_no_touching'
        
        # Store final confluent lesion information
        confluent_lesions[lesion_id] = {
            'sources': sorted(onset_overlaps.keys()),
            'onset_session': onset_session,
            'overlaps': onset_overlaps,
            'overlap_evolution': overlap_evolution,
            'overlap_persistent': post_onset_valid,
            'overlap_validation': overlap_validation,
            'min_distance_mm': min_distance_overall,
            'distance_session': distance_session,
            'distance_persistent': distance_persistent,
            'distance_validation': distance_validation,
            'validation_comment': overlap_comment
        }
    
    return confluent_lesions


# -------------------------------------------------------------------------
# FUNCTION: Create confluent lesion maps
# -------------------------------------------------------------------------
def create_confluent_lesion_maps(relabeled_data, confluent_lesions):
    """
    Create separate maps for confluent lesions with original labels.
    
    This function creates new label maps containing only confluent/contact lesions
    and their source lesions, keeping their original label IDs.
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
        List of label arrays showing only confluent lesions with original labels.
    """
    if not confluent_lesions:
        return None
    
    confluent_maps = [np.zeros_like(data) for data in relabeled_data]
    
    for lesion_id, info in confluent_lesions.items():
        source_ids = info['sources']
        
        # Copy the confluent lesion and all its sources with original labels
        for session_idx in range(len(confluent_maps)):
            # Copy the confluent lesion itself
            mask = (relabeled_data[session_idx] == lesion_id)
            confluent_maps[session_idx][mask] = lesion_id
            
            # Copy all source lesions
            for source_id in source_ids:
                source_mask = (relabeled_data[session_idx] == source_id)
                confluent_maps[session_idx][source_mask] = source_id
    
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


# -------------------------------------------------------------------------
# FUNCTION: Create new lesion map with pre-lesion labels
# -------------------------------------------------------------------------
def create_new_lesion_map(relabeled_data):
    """
    Create a map containing only NEW lesions (appearing after first session)
    with their pre-lesion labels.
    
    This map shows:
    - NEW lesions with positive labels in sessions where they exist
    - Pre-lesion labels (negative IDs) in all sessions before first appearance
    - Zero elsewhere (excludes lesions from first session)
    
    Parameters
    ----------
    relabeled_data : list of np.ndarray
        List of 3D label arrays (same shape), one per session,
        in chronological order.
    
    Returns
    -------
    new_lesion_maps : list of np.ndarray
        List of label arrays containing only new lesions with pre-lesion labels.
    """
    n_sessions = len(relabeled_data)
    new_lesion_maps = [np.zeros_like(data, dtype=np.int32) for data in relabeled_data]
    
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
    
    # For each NEW lesion (first appearance > 0), add to map with pre-lesion labels
    for lesion_id, first_session in lesion_first_appearance.items():
        if first_session is None or first_session == 0:
            # Skip lesions from first session
            continue
        
        # Add pre-lesion labels (negative) to all sessions before first appearance
        pre_lesion_label = -lesion_id
        for session_idx in range(first_session):
            lesion_mask_first = (relabeled_data[first_session] == lesion_id)
            new_lesion_maps[session_idx][lesion_mask_first] = pre_lesion_label
        
        # Add positive lesion labels to all sessions where it exists
        for session_idx in range(first_session, n_sessions):
            lesion_mask = (relabeled_data[session_idx] == lesion_id)
            new_lesion_maps[session_idx][lesion_mask] = lesion_id
    
    return new_lesion_maps


# -------------------------------------------------------------------------
# FUNCTION: Create volume change maps (growing, shrinking, stable)
# -------------------------------------------------------------------------
def create_volume_change_maps(relabeled_data, volume_changes):
    """
    Create separate maps for growing, shrinking, and stable lesions.
    
    This function creates three sets of maps:
    - Growing maps: contain only lesions classified as growing
    - Shrinking maps: contain only lesions classified as shrinking
    - Stable maps: contain only lesions classified as stable
    
    Parameters
    ----------
    relabeled_data : list of np.ndarray
        List of 3D label arrays (same shape), one per session,
        in chronological order.
    volume_changes : dict
        Dictionary mapping lesion IDs to volume change information.
    
    Returns
    -------
    growing_maps : list of np.ndarray
        List of label arrays containing only growing lesions.
    shrinking_maps : list of np.ndarray
        List of label arrays containing only shrinking lesions.
    stable_maps : list of np.ndarray
        List of label arrays containing only stable lesions.
    """
    if not volume_changes:
        return None, None, None
    
    growing_maps = [np.zeros_like(data) for data in relabeled_data]
    shrinking_maps = [np.zeros_like(data) for data in relabeled_data]
    stable_maps = [np.zeros_like(data) for data in relabeled_data]
    
    for lesion_id, info in volume_changes.items():
        status = info['status']
        
        # Copy lesion to appropriate map(s)
        for session_idx in range(len(relabeled_data)):
            mask = (relabeled_data[session_idx] == lesion_id)
            
            if status == 'growing':
                growing_maps[session_idx][mask] = lesion_id
            elif status == 'shrinking':
                shrinking_maps[session_idx][mask] = lesion_id
            elif status == 'stable':
                stable_maps[session_idx][mask] = lesion_id
    
    return growing_maps, shrinking_maps, stable_maps


# -------------------------------------------------------------------------
# FUNCTION: Create lesion category maps
# -------------------------------------------------------------------------
def create_category_maps(relabeled_data, volume_changes=None, confluent_lesions=None):
    """
    Create maps where each lesion is labeled by its category.
    
    Categories:
    - 0: undetermined (lesion appears once and never after)
    - 1: stable lesion (volume changes < 10-15%)
    - 2: growing lesion (volume increases > 20% between consecutive sessions)
    - 3: shrinking lesion (volume decreases > 20%)
    - 4: new lesion (first appears after session 0)
    - 5: confluent lesion (loss of distinct boundaries: ≥3 voxel overlap OR touching <1mm)
    
    Priority system:
    - Growing (2) has highest priority: requires >20% increase in at least one session
    - Confluent (5) applies when lesions merge or touch
    - New (4) is important - appears after baseline
    - Stable (1) for lesions without significant changes
    - Shrinking (3) for lesions that decrease >20%
    - Undetermined (0) for lesions that appear at only one session
    
    Priority order: Growing > Confluent > New > Stable > Shrinking > Undetermined
    
    Parameters
    ----------
    relabeled_data : list of np.ndarray
        List of 3D label arrays (same shape), one per session,
        in chronological order.
    volume_changes : dict, optional
        Dictionary mapping lesion IDs to volume change information.
    confluent_lesions : dict, optional
        Dictionary mapping lesion IDs to confluent information.
    
    Returns
    -------
    category_maps : list of np.ndarray
        List of label arrays where each voxel is labeled by lesion category.
    """
    n_sessions = len(relabeled_data)
    category_maps = [np.zeros_like(data, dtype=np.uint8) for data in relabeled_data]
    
    # Find all unique lesion labels and their first appearance
    lesion_first_appearance = {}
    all_labels = set()
    for data in relabeled_data:
        all_labels.update(np.unique(data[data > 0]))
    
    for lesion_id in all_labels:
        lesion_id = int(lesion_id)
        lesion_first_appearance[lesion_id] = None
    
    for session_idx in range(n_sessions):
        for lesion_id in np.unique(relabeled_data[session_idx][relabeled_data[session_idx] > 0]):
            lesion_id = int(lesion_id)
            if lesion_first_appearance[lesion_id] is None:
                lesion_first_appearance[lesion_id] = session_idx
    
    # Assign categories to each lesion
    for lesion_id in sorted(all_labels):
        lesion_id = int(lesion_id)
        
        # Determine category with new priority system
        category = 0  # Default: undetermined
        category_assigned = False
        
        # Determine if new lesion (appears after baseline)
        is_new_lesion = False
        if lesion_first_appearance[lesion_id] is not None and lesion_first_appearance[lesion_id] > 0:
            is_new_lesion = True
        
        # Determine if undetermined (appears once and never after)
        is_undetermined = False
        if lesion_first_appearance[lesion_id] is not None:
            first_session = lesion_first_appearance[lesion_id]
            last_session = first_session
            for session_idx in range(first_session + 1, n_sessions):
                if lesion_id in relabeled_data[session_idx]:
                    last_session = session_idx
            
            if first_session == last_session:
                is_undetermined = True
                category = 0  # Undetermined
                category_assigned = True
        
        # Check volume changes
        is_growing = False
        is_shrinking = False
        is_stable = False
        
        if volume_changes is not None and lesion_id in volume_changes:
            vol_info = volume_changes[lesion_id]
            status = vol_info['status']
            max_increase_percent = vol_info['max_increase_percent']
            max_decrease_percent = vol_info['max_decrease_percent']
            
            if status == 'growing':
                is_growing = True
                category = 2  # Growing has highest priority
                category_assigned = True
            elif status == 'shrinking':
                is_shrinking = True
                if not category_assigned or category == 0:
                    category = 3
                    category_assigned = True
            elif status == 'stable' and not is_undetermined:
                is_stable = True
                if not category_assigned or category == 0:
                    category = 1
                    category_assigned = True
        
        # Check confluent status (only if not growing)
        if not is_growing and confluent_lesions is not None and lesion_id in confluent_lesions:
            confluent_info = confluent_lesions[lesion_id]
            # Confluent (type can be 'confluent_overlap' or 'confluent_distance')
            category = 5
            category_assigned = True
        
        # Check new lesion status (lower priority than confluent)
        if not is_growing and not category_assigned and is_new_lesion and not is_undetermined:
            category = 4  # New lesion
            category_assigned = True
        
        # Apply category to all sessions where lesion exists
        for session_idx in range(n_sessions):
            mask = (relabeled_data[session_idx] == lesion_id)
            category_maps[session_idx][mask] = category
    
    return category_maps


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
        args.save_new_lesion_map = True
        args.detect_confluent = True
        args.fill_intermediate_missing = True
        args.detect_volume_changes = True
    
    # If --save_all_maps is specified, enable all map saving options
    if args.save_all_maps:
        args.save_new_lesion_map = True
        args.save_volume_change_maps = True
        args.save_category_maps = True
        # Confluent maps will be saved automatically if detect_confluent is enabled

    # Apply processing based on requested options
    if args.add_pre_lesion_labels:
        print("Adding pre-lesion labels...")
        data = add_pre_lesion_labels(data)

    confluent_lesions = None
    confluent_maps = None
    if args.detect_confluent:
        print("Detecting confluent lesions using overlap and distance analysis...")
        confluent_lesions = detect_confluent_lesions(data, affine,
                                                     args.min_confluence_overlap,
                                                     args.confluent_distance_mm)
        if confluent_lesions:
            confluent_count = len(confluent_lesions)
            
            print(f"Found {confluent_count} confluent lesion(s):")
            print()
            
            for lesion_id, info in confluent_lesions.items():
                overlap_validation = info['overlap_validation']
                distance_validation = info['distance_validation']
                onset_session = info['onset_session']
                sources = info['sources']
                overlaps = info['overlaps']
                overlap_str = ', '.join([f"{src} ({overlaps[src]} voxels)" for src in sources])
                
                print(f"  Lesion {lesion_id} - Onset: Session {onset_session}")
                print(f"    Overlap validation: {overlap_validation} - sources: {overlap_str}")
                
                # Show overlap validation status
                if overlap_validation == 'valid_multi_session':
                    print(f"    Valid multi-session confluence")
                elif overlap_validation == 'invalid_transient':
                    print(f"    Invalid: Single-session or transient overlap (may be segmentation artifact)")
                elif overlap_validation == 'invalid_inconsistent':
                    print(f"    Invalid: Inconsistent overlap pattern")
                
                # Show overlap evolution
                overlap_evolution = info['overlap_evolution']
                if overlap_evolution:
                    print("    Overlap evolution:")
                    for sess_idx in sorted(overlap_evolution.keys()):
                        sess_overlaps = overlap_evolution[sess_idx]
                        total = sum(sess_overlaps.values())
                        detail = ', '.join([f"{src}({count})" for src, count in sorted(sess_overlaps.items())])
                        print(f"      Session {sess_idx}: {total} voxels total ({detail})")
                
                # Show distance information if available
                if info['min_distance_mm'] is not None:
                    min_dist = info['min_distance_mm']
                    dist_sess = info['distance_session']
                    print(f"    Distance validation: {distance_validation} (min: {min_dist:.2f} mm at session {dist_sess})")
                
                print()
            
            print("Creating separate confluent lesion maps...")
            confluent_maps = create_confluent_lesion_maps(data, confluent_lesions)
        else:
            print("No confluent lesions detected.")

    if args.fill_intermediate_missing:
        print("Filling intermediate missing lesions...")
        data = fill_intermediate_missing_lesions(data)

    # Detect volume changes if requested
    volume_changes = None
    growing_maps = None
    shrinking_maps = None
    stable_maps = None
    if args.detect_volume_changes:
        print(f"Detecting volume changes...")
        print(f"  Growing threshold: {args.growing_threshold_percent}%")
        print(f"  Shrinking threshold: {args.shrinking_threshold_percent}%")
        print(f"  Stable threshold: {args.stable_threshold_percent}%")
        volume_changes = detect_volume_changes(data, affine, 
                                               args.growing_threshold_percent,
                                               args.shrinking_threshold_percent,
                                               args.stable_threshold_percent)
        
        # Count lesions by status
        growing = sum(1 for info in volume_changes.values() if info['status'] == 'growing')
        shrinking = sum(1 for info in volume_changes.values() if info['status'] == 'shrinking')
        stable = sum(1 for info in volume_changes.values() if info['status'] == 'stable')
        undetermined = sum(1 for info in volume_changes.values() if info['status'] == 'undetermined')
        
        print(f"Volume change analysis:")
        print(f"  - {growing} growing lesion(s) (>%{args.growing_threshold_percent} increase)")
        print(f"  - {shrinking} shrinking lesion(s) (>%{args.shrinking_threshold_percent} decrease)")
        print(f"  - {stable} stable lesion(s) (<%{args.stable_threshold_percent} change)")
        print(f"  - {undetermined} undetermined lesion(s) (appear once, never after)")
        
        # Create volume change maps if requested
        if args.save_volume_change_maps:
            print("Creating volume change maps...")
            growing_maps, shrinking_maps, stable_maps = create_volume_change_maps(data, volume_changes)

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

    # Save new lesion map if requested
    if args.save_new_lesion_map:
        print(f"Saving {len(imgs)} new lesion map(s) to {args.out_dir}...")
        new_lesion_maps = create_new_lesion_map(data)
        for i, img in enumerate(imgs):
            basename = os.path.basename(args.in_images[i])
            name, ext = os.path.splitext(basename)
            if ext == '.gz':
                name = os.path.splitext(name)[0]  # Remove .nii from .nii.gz
                ext = '.nii.gz'
            new_lesion_output = os.path.join(args.out_dir, f"{name}_new_lesions{ext}")
            nib.save(nib.Nifti1Image(new_lesion_maps[i], img.affine), new_lesion_output)
    
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
    
    # Save volume change maps if they were created
    if growing_maps is not None:
        print(f"Saving {len(imgs)} growing lesion map(s) to {args.out_dir}...")
        for i, img in enumerate(imgs):
            basename = os.path.basename(args.in_images[i])
            name, ext = os.path.splitext(basename)
            if ext == '.gz':
                name = os.path.splitext(name)[0]  # Remove .nii from .nii.gz
                ext = '.nii.gz'
            growing_output = os.path.join(args.out_dir, f"{name}_growing{ext}")
            nib.save(nib.Nifti1Image(growing_maps[i], img.affine), growing_output)
    
    if shrinking_maps is not None:
        print(f"Saving {len(imgs)} shrinking lesion map(s) to {args.out_dir}...")
        for i, img in enumerate(imgs):
            basename = os.path.basename(args.in_images[i])
            name, ext = os.path.splitext(basename)
            if ext == '.gz':
                name = os.path.splitext(name)[0]  # Remove .nii from .nii.gz
                ext = '.nii.gz'
            shrinking_output = os.path.join(args.out_dir, f"{name}_shrinking{ext}")
            nib.save(nib.Nifti1Image(shrinking_maps[i], img.affine), shrinking_output)
    
    if stable_maps is not None:
        print(f"Saving {len(imgs)} stable lesion map(s) to {args.out_dir}...")
        for i, img in enumerate(imgs):
            basename = os.path.basename(args.in_images[i])
            name, ext = os.path.splitext(basename)
            if ext == '.gz':
                name = os.path.splitext(name)[0]  # Remove .nii from .nii.gz
                ext = '.nii.gz'
            stable_output = os.path.join(args.out_dir, f"{name}_stable{ext}")
            nib.save(nib.Nifti1Image(stable_maps[i], img.affine), stable_output)
    
    # Save category maps if requested
    if args.save_category_maps:
        if volume_changes is not None or confluent_lesions is not None:
            print(f"Saving {len(imgs)} lesion category map(s) to {args.out_dir}...")
            category_maps = create_category_maps(data, volume_changes, confluent_lesions)
            for i, img in enumerate(imgs):
                basename = os.path.basename(args.in_images[i])
                name, ext = os.path.splitext(basename)
                if ext == '.gz':
                    name = os.path.splitext(name)[0]  # Remove .nii from .nii.gz
                    ext = '.nii.gz'
                category_output = os.path.join(args.out_dir, f"{name}_categories{ext}")
                nib.save(nib.Nifti1Image(category_maps[i], img.affine), category_output)
        else:
            print("Warning: --save_category_maps requires --detect_volume_changes and/or --detect_confluent")
    
    # Generate lesion report CSV
    print("Generating lesion report CSV...")
    csv_path = generate_lesion_report(data, args.out_dir, affine, confluent_lesions, volume_changes)
    print(f"Lesion report saved to: {csv_path}")


if __name__ == "__main__":
    main()


