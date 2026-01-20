#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script creates contralateral (mirror) ROIs for each lesion ROI,
with validation constraints for use as control tissue in NAWM.

Method:
For each lesion ROI, the corresponding contralateral NAWM (Normal Appearing
White Matter) ROI is determined by reflecting the lesion ROI across the
midline of the brain. Contralateral ROIs are adjusted to ensure that only
portions that are classified as NAWM are considered as spatially matched 
contralateral control tissue. Any adjustments made to contralateral ROIs are 
also applied to the original ROI via reflection across the midline.

Output:
- contralateral_roi.nii.gz: Mirrored ROIs
- roi_adjusted.nii.gz: Adjusted original ROIs
- contralateral_roi_report.csv: Summary of contralateral ROI creation with
  NAWM violation warnings

All images should be co-registered and have the same shape.

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
    [1] Sormani, Maria P., et al. "Magnetic Resonance Imaging as a Potential
    Measure of Lesion Burden and Brain Atrophy in Multiple Sclerosis."
    Nature Reviews Neurology 17.8 (2021): 465-481.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC8453433/
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('roi_labels',
                   help='ROI label image in NIfTI format containing all lesion/ROI labels. '
                        'Each positive integer represents a distinct ROI/lesion.')
    p.add_argument('nawm_mask',
                   help='Normal Appearing White Matter (NAWM) mask in NIfTI format. '
                        'Voxels with value > 0 are considered valid NAWM.')
    p.add_argument('out_dir',
                   help='Output directory for contralateral ROIs and reports.')
    
    p.add_argument('--min_distance_mm', type=float, default=2.0,
                   help='Minimum distance in millimeters between ROI/contralateral '
                        'and NAWM boundary, and between different ROIs [%(default)s].')
    
    p.add_argument('--save_adjusted_roi', action='store_true',
                   help='Save the adjusted original ROI (roi_adjusted.nii.gz). '
                        'By default, only the contralateral ROI is saved.')
    
    p.add_argument('--force_creation', action='store_true',
                   help='Force creation of contralateral ROIs even if they violate '
                        'distance constraints. Violations will be noted in the report.')

    add_overwrite_arg(p)
    return p


# -------------------------------------------------------------------------
# FUNCTION: Reflect ROI across midline
# -------------------------------------------------------------------------
def reflect_roi_across_midline(roi_data):
    """
    Reflect an ROI across the midline (sagittal plane) of the brain.
    
    The reflection is done by flipping the image along the X axis (left-right).
    
    Parameters
    ----------
    roi_data : np.ndarray
        3D binary/labeled ROI array.
    
    Returns
    -------
    reflected_roi : np.ndarray
        3D reflected ROI array (flipped in X dimension).
    """
    # Flip the ROI in the first dimension (X/left-right)
    reflected_roi = np.flip(roi_data, axis=0)
    
    return reflected_roi


# -------------------------------------------------------------------------
# FUNCTION: Validate contralateral ROI
# -------------------------------------------------------------------------
def validate_contralateral_roi(contralateral_roi, nawm_mask, affine, 
                               min_distance_mm=2.0, force_creation=False):
    """
    Validate and adjust contralateral ROI based on NAWM classification and
    distance constraints.
    
    The contralateral ROI is adjusted to include only voxels that are:
    1. Within the NAWM mask
    2. At least min_distance_mm away from NAWM boundary
    
    Parameters
    ----------
    contralateral_roi : np.ndarray
        3D binary contralateral ROI array.
    nawm_mask : np.ndarray
        3D binary NAWM mask array.
    affine : np.ndarray
        4x4 affine transformation matrix.
    min_distance_mm : float
        Minimum distance in millimeters from NAWM boundary.
    force_creation : bool
        If True, create ROI even if distance constraints are violated.
    
    Returns
    -------
    adjusted_roi : np.ndarray
        Adjusted contralateral ROI.
    nawm_violation : bool
        Whether the original ROI contains any non-NAWM voxels.
    nawm_violation_voxels : int
        Number of non-NAWM voxels.
    distance_violation : bool
        Whether voxels too close to NAWM boundary exist.
    distance_violation_voxels : int
        Number of voxels too close to NAWM boundary.
    """
    adjusted_roi = contralateral_roi.copy()
    
    # Check for non-NAWM voxels
    non_nawm_mask = (adjusted_roi > 0) & (nawm_mask == 0)
    nawm_violation = np.sum(non_nawm_mask) > 0
    nawm_violation_voxels = int(np.sum(non_nawm_mask))
    
    # Constraint 1: Must be within NAWM (if not forcing creation)
    if not force_creation:
        adjusted_roi = adjusted_roi & (nawm_mask > 0)
    
    # Constraint 2: Must be at least min_distance_mm from NAWM boundary
    roi_voxels = np.argwhere(adjusted_roi > 0)
    distance_violation = False
    distance_violation_voxels = 0
    
    if len(roi_voxels) > 0:
        # Find non-NAWM voxels and compute distance from ROI to them
        non_nawm_voxels = np.argwhere(nawm_mask == 0)
        
        if len(non_nawm_voxels) > 0:
            # Convert to physical coordinates
            roi_coords_mm = []
            for voxel in roi_voxels:
                voxel_homogeneous = np.append(voxel, 1)
                coord_mm = affine @ voxel_homogeneous
                roi_coords_mm.append(coord_mm[:3])
            roi_coords_mm = np.array(roi_coords_mm)
            
            non_nawm_coords_mm = []
            for voxel in non_nawm_voxels:
                voxel_homogeneous = np.append(voxel, 1)
                coord_mm = affine @ voxel_homogeneous
                non_nawm_coords_mm.append(coord_mm[:3])
            non_nawm_coords_mm = np.array(non_nawm_coords_mm)
            
            # Compute distances
            distances = cdist(roi_coords_mm, non_nawm_coords_mm)
            min_distances = np.min(distances, axis=1)
            
            # Check violations
            valid_mask = min_distances >= min_distance_mm
            distance_violation = np.sum(~valid_mask) > 0
            distance_violation_voxels = int(np.sum(~valid_mask))
            
            # Update ROI with only valid voxels (if not forcing)
            if not force_creation:
                for i, (voxel, is_valid) in enumerate(zip(roi_voxels, valid_mask)):
                    if not is_valid:
                        adjusted_roi[tuple(voxel)] = 0
    
    return adjusted_roi, nawm_violation, nawm_violation_voxels, distance_violation, distance_violation_voxels


# -------------------------------------------------------------------------
# FUNCTION: Check inter-ROI distances
# -------------------------------------------------------------------------
def check_inter_roi_distance(roi_labels, affine, min_distance_mm=2.0):
    """
    Check distances between different ROIs.
    
    Parameters
    ----------
    roi_labels : np.ndarray
        3D array with distinct labels for each ROI.
    affine : np.ndarray
        4x4 affine transformation matrix.
    min_distance_mm : float
        Minimum distance in millimeters.
    
    Returns
    -------
    violations : dict
        Dictionary mapping ROI_ID tuples to minimum distance found.
    """
    unique_rois = np.unique(roi_labels[roi_labels > 0])
    violations = {}
    
    roi_coords = {}
    # Get coordinates for each ROI
    for roi_id in unique_rois:
        roi_id = int(roi_id)
        roi_voxels = np.argwhere(roi_labels == roi_id)
        coords_mm = []
        for voxel in roi_voxels:
            voxel_homogeneous = np.append(voxel, 1)
            coord_mm = affine @ voxel_homogeneous
            coords_mm.append(coord_mm[:3])
        roi_coords[roi_id] = np.array(coords_mm)
    
    # Check pairwise distances
    roi_ids = sorted(unique_rois)
    for i, roi_id1 in enumerate(roi_ids):
        for roi_id2 in roi_ids[i+1:]:
            roi_id1 = int(roi_id1)
            roi_id2 = int(roi_id2)
            
            if len(roi_coords[roi_id1]) > 0 and len(roi_coords[roi_id2]) > 0:
                distances = cdist(roi_coords[roi_id1], roi_coords[roi_id2])
                min_dist = np.min(distances)
                
                if min_dist < min_distance_mm:
                    violations[(roi_id1, roi_id2)] = min_dist
    
    return violations


# -------------------------------------------------------------------------
# FUNCTION: Create contralateral ROIs
# -------------------------------------------------------------------------
def create_contralateral_rois(roi_labels, nawm_mask, affine, 
                              min_distance_mm=2.0, force_creation=False):
    """
    Create contralateral ROIs with distance and NAWM validation constraints.
    
    Parameters
    ----------
    roi_labels : np.ndarray
        3D array with distinct labels for each ROI/lesion.
    nawm_mask : np.ndarray
        NAWM mask array.
    affine : np.ndarray
        4x4 affine transformation matrix.
    min_distance_mm : float
        Minimum distance in millimeters.
    force_creation : bool
        If True, create ROIs even if distance constraints are violated.
    
    Returns
    -------
    contralateral_rois : np.ndarray
        Contralateral ROI labels.
    adjusted_roi_labels : np.ndarray
        Adjusted ROI labels (reflected from contralateral).
    stats : list of dict
        Statistics for each ROI.
    inter_roi_violations : dict
        Dictionary of inter-ROI distance violations.
    """
    unique_rois = np.unique(roi_labels[roi_labels > 0])
    
    contralateral_labels = np.zeros_like(roi_labels)
    adjusted_roi_labels_out = np.zeros_like(roi_labels)
    stats = []
    
    for roi_id in sorted(unique_rois):
        roi_id = int(roi_id)
        roi_mask = (roi_labels == roi_id)
        
        # Step 1: Reflect ROI across midline
        reflected_roi = reflect_roi_across_midline(roi_mask)
        
        # Step 2: Validate contralateral ROI
        adjusted_contra, nawm_viol, nawm_viol_voxels, dist_viol, dist_viol_voxels = validate_contralateral_roi(
            reflected_roi, nawm_mask, affine, min_distance_mm, force_creation)
        
        # Step 3: Reflect adjusted contralateral back to adjust original ROI
        adjusted_original = reflect_roi_across_midline(adjusted_contra)
        
        # Store in output maps
        contralateral_labels[adjusted_contra > 0] = roi_id
        adjusted_roi_labels_out[adjusted_original > 0] = roi_id
        
        # Store statistics
        original_voxels = np.sum(roi_mask > 0)
        adjusted_original_voxels = np.sum(adjusted_original > 0)
        adjusted_contra_voxels = np.sum(adjusted_contra > 0)
        
        violation_notes = []
        if nawm_viol:
            violation_notes.append('NAWM constraint violated')
        if dist_viol:
            violation_notes.append('Distance from NAWM boundary violated')
        
        stats.append({
            'ROI_ID': roi_id,
            'Original_Voxels': int(original_voxels),
            'Adjusted_Original_Voxels': int(adjusted_original_voxels),
            'Contralateral_Voxels': int(adjusted_contra_voxels),
            'NAWM_Violation': nawm_viol,
            'NAWM_Violation_Voxels': int(nawm_viol_voxels),
            'Distance_Violation': dist_viol,
            'Distance_Violation_Voxels': int(dist_viol_voxels),
            'Retention_Rate': float(adjusted_contra_voxels / original_voxels) if original_voxels > 0 else 0.0,
            'Note': '; '.join(violation_notes) if violation_notes else 'OK'
        })
    
    # Check inter-ROI distances
    inter_roi_violations = check_inter_roi_distance(contralateral_labels, affine, min_distance_mm)
    
    return contralateral_labels, adjusted_roi_labels_out, stats, inter_roi_violations


# -------------------------------------------------------------------------
# FUNCTION: Generate contralateral ROI report
# -------------------------------------------------------------------------
def generate_contralateral_report(stats, inter_roi_violations, output_dir, affine=None):
    """
    Generate a CSV report with contralateral ROI creation statistics,
    NAWM violations, and distance constraint violations.
    
    Parameters
    ----------
    stats : list of dict
        Statistics from ROI creation.
    inter_roi_violations : dict
        Inter-ROI distance violations.
    output_dir : str
        Directory where the CSV file will be saved.
    affine : np.ndarray, optional
        4x4 affine transformation matrix for volume calculation.
    
    Returns
    -------
    csv_path : str
        Path to the generated CSV file.
    """
    rows = []
    for stat in stats:
        row = {
            'ROI_ID': stat['ROI_ID'],
            'Original_Voxels': stat['Original_Voxels'],
            'Adjusted_Original_Voxels': stat['Adjusted_Original_Voxels'],
            'Contralateral_Voxels': stat['Contralateral_Voxels'],
            'NAWM_Violation': stat['NAWM_Violation'],
            'NAWM_Violation_Voxels': stat['NAWM_Violation_Voxels'],
            'Distance_Violation': stat['Distance_Violation'],
            'Distance_Violation_Voxels': stat['Distance_Violation_Voxels'],
            'Retention_Rate': stat['Retention_Rate'],
            'Note': stat['Note']
        }
        
        # Add volume in mm³ if affine is provided
        if affine is not None:
            voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))
            row['Contralateral_Volume_mm3'] = stat['Contralateral_Voxels'] * voxel_volume
            row['Adjusted_Original_Volume_mm3'] = stat['Adjusted_Original_Voxels'] * voxel_volume
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'contralateral_roi_report.csv')
    df.to_csv(csv_path, index=False)
    
    # Print inter-ROI violations if any
    if inter_roi_violations:
        print('\nInter-ROI distance violations (< min_distance_mm):')
        for (roi_id1, roi_id2), min_dist in inter_roi_violations.items():
            print(f'  ROI {roi_id1} - ROI {roi_id2}: {min_dist:.2f} mm')
    
    return csv_path


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.roi_labels, args.nawm_mask])
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)
    assert_headers_compatible(parser, [args.roi_labels, args.nawm_mask])

    # Load ROI labels
    print("Loading ROI label image...")
    roi_img = nib.load(args.roi_labels)
    roi_data = roi_img.get_fdata()
    affine = roi_img.affine

    # Load NAWM mask
    print("Loading NAWM mask...")
    nawm_img = nib.load(args.nawm_mask)
    nawm_data = nawm_img.get_fdata()

    # Create contralateral ROIs
    print("Creating contralateral ROIs with distance and NAWM validation...")
    if args.force_creation:
        print(f"  (Force creation enabled - violations will be noted in report)")
    print(f"  Min distance threshold: {args.min_distance_mm} mm")
    
    contralateral_rois, adjusted_roi_labels, stats, inter_roi_violations = create_contralateral_rois(
        roi_data, nawm_data, affine, args.min_distance_mm, args.force_creation)

    # Save contralateral ROI image
    print("Saving contralateral ROI image...")
    output_path = os.path.join(args.out_dir, 'contralateral_roi.nii.gz')
    nib.save(nib.Nifti1Image(contralateral_rois, affine), output_path)

    # Save adjusted ROI image (optional)
    if args.save_adjusted_roi:
        print("Saving adjusted ROI image...")
        output_path = os.path.join(args.out_dir, 'roi_adjusted.nii.gz')
        nib.save(nib.Nifti1Image(adjusted_roi_labels, affine), output_path)

    # Generate report
    print("Generating contralateral ROI report...")
    csv_path = generate_contralateral_report(stats, inter_roi_violations, args.out_dir, affine)
    print(f"Report saved to: {csv_path}")
    
    # Print summary of violations if any
    nawm_violations = [s for s in stats if s['NAWM_Violation']]
    dist_violations = [s for s in stats if s['Distance_Violation']]
    
    if nawm_violations:
        print(f"\nWarning: {len(nawm_violations)} ROI(s) have NAWM violations:")
        for stat in nawm_violations:
            print(f"  ROI {stat['ROI_ID']}: {stat['NAWM_Violation_Voxels']} non-NAWM voxels")
    
    if dist_violations:
        print(f"\nWarning: {len(dist_violations)} ROI(s) violate distance constraint:")
        for stat in dist_violations:
            print(f"  ROI {stat['ROI_ID']}: {stat['Distance_Violation_Voxels']} voxels too close to NAWM boundary")

    print("Done!")


if __name__ == "__main__":
    main()
