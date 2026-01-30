#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify lesions based on their spatial relationship with tissue masks.

This script takes a labeled lesion image (where each lesion has a unique
integer label) and tissue masks (white matter, grey matter, and CSF) to
classify each lesion into one of the following categories:

1. Periventricular: Lesion that overlaps with the CSF mask (at least 3 voxels)
2. Juxtacortical/Cortical: Lesion that overlaps with grey matter (3 voxels min)
   or that is in white matter but in contact with grey matter (expand lesion by 1
   voxel to identify overlap with GM)
3. White Matter: Lesion that overlaps 100% with white matter
4. Confluent: Lesion with > 60% WM, > 5% CSF and > 5% GM overlap

The script outputs:
- A labeled image where each lesion is relabeled according to its class
- A CSV report with detailed information about each lesion's classification

Classification priority (when a lesion meets multiple criteria):
1. Confluent (if meets all confluent criteria)
2. Periventricular (if overlaps with CSF)
3. Juxtacortical/Cortical (if overlaps with GM or is adjacent to GM)
4. White Matter (default if only overlaps with WM)

Usage example:
    scil_lesions_classify.py lesions_labels.nii.gz \\
                             wm_mask.nii.gz \\
                             gm_mask.nii.gz \\
                             csf_mask.nii.gz \\
                             out_classified.nii.gz \\
                             out_report.csv \\
                             --min_overlap 3

Label mapping for output image:
    1 = White Matter
    2 = Juxtacortical/Cortical
    3 = Periventricular
    4 = Confluent

References:
    [1] Filippi, M., et al. "Diagnosis of progressive multiple sclerosis from
    the imaging perspective: a review." JAMA Neurology 78.3 (2021): 351-364.
    
    [2] Thompson, A. J., et al. "Diagnosis of multiple sclerosis: 2017 revisions
    of the McDonald criteria." The Lancet Neurology 17.2 (2018): 162-173.
"""

import argparse
import logging

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from scilpy.image.labels import get_data_as_labels
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_headers_compatible,
                             add_verbose_arg)


# Classification labels
CLASS_WM = 1
CLASS_JUXTACORTICAL = 2
CLASS_PERIVENTRICULAR = 3
CLASS_CONFLUENT = 4

CLASS_NAMES = {
    CLASS_WM: 'White Matter',
    CLASS_JUXTACORTICAL: 'Juxtacortical/Cortical',
    CLASS_PERIVENTRICULAR: 'Periventricular',
    CLASS_CONFLUENT: 'Confluent'
}


def _build_arg_parser():
    """Build argument parser."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    p.add_argument('in_lesions',
                   help='Input lesion label file in NIfTI format.')
    p.add_argument('in_wm_mask',
                   help='White matter binary mask in NIfTI format.')
    p.add_argument('in_gm_mask',
                   help='Grey matter binary mask in NIfTI format.')
    p.add_argument('in_csf_mask',
                   help='CSF binary mask in NIfTI format.')
    p.add_argument('out_classified',
                   help='Output classified lesion map in NIfTI format.')
    p.add_argument('out_csv',
                   help='Output CSV report with lesion classifications.')
    
    p.add_argument('--min_overlap', type=int, default=3,
                   help='Minimum number of overlapping voxels for '
                        'periventricular and juxtacortical classification '
                        '[%(default)s].')
    p.add_argument('--confluent_wm_threshold', type=float, default=0.60,
                   help='Minimum white matter overlap percentage for '
                        'confluent classification [%(default)s].')
    p.add_argument('--confluent_csf_threshold', type=float, default=0.05,
                   help='Minimum CSF overlap percentage for '
                        'confluent classification [%(default)s].')
    p.add_argument('--confluent_gm_threshold', type=float, default=0.05,
                   help='Minimum grey matter overlap percentage for '
                        'confluent classification [%(default)s].')
    
    add_verbose_arg(p)
    add_overwrite_arg(p)
    
    return p


def classify_lesion(lesion_mask, wm_mask, gm_mask, csf_mask,
                   min_overlap=3, wm_thresh=0.60, csf_thresh=0.05,
                   gm_thresh=0.05):
    """
    Classify a single lesion based on tissue overlap.
    
    Parameters
    ----------
    lesion_mask : ndarray
        Binary mask of the lesion.
    wm_mask : ndarray
        Binary white matter mask.
    gm_mask : ndarray
        Binary grey matter mask.
    csf_mask : ndarray
        Binary CSF mask.
    min_overlap : int
        Minimum number of voxels for overlap criteria.
    wm_thresh : float
        WM percentage threshold for confluent lesions.
    csf_thresh : float
        CSF percentage threshold for confluent lesions.
    gm_thresh : float
        GM percentage threshold for confluent lesions.
    
    Returns
    -------
    classification : int
        Classification label (1-4).
    overlap_stats : dict
        Dictionary with overlap statistics.
    """
    lesion_voxels = np.sum(lesion_mask)
    
    # Compute overlaps
    wm_overlap = np.sum(lesion_mask & wm_mask)
    gm_overlap = np.sum(lesion_mask & gm_mask)
    csf_overlap = np.sum(lesion_mask & csf_mask)
    
    # Dilate lesion by 1 voxel to check for adjacency to grey matter
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    lesion_dilated = ndimage.binary_dilation(lesion_mask, structure=struct,
                                             iterations=1)
    lesion_dilated_gm_overlap = np.sum(lesion_dilated & gm_mask)
    
    # Compute percentages
    wm_pct = wm_overlap / lesion_voxels if lesion_voxels > 0 else 0
    gm_pct = gm_overlap / lesion_voxels if lesion_voxels > 0 else 0
    csf_pct = csf_overlap / lesion_voxels if lesion_voxels > 0 else 0
    
    overlap_stats = {
        'lesion_volume_voxels': lesion_voxels,
        'wm_overlap_voxels': wm_overlap,
        'gm_overlap_voxels': gm_overlap,
        'csf_overlap_voxels': csf_overlap,
        'wm_percentage': wm_pct * 100,
        'gm_percentage': gm_pct * 100,
        'csf_percentage': csf_pct * 100
    }
    
    # Classification logic (priority order)
    
    # 1. Check for confluent lesion
    if (wm_pct > wm_thresh and csf_pct > csf_thresh and gm_pct > gm_thresh):
        classification = CLASS_CONFLUENT
    
    # 2. Check for periventricular (overlaps with CSF)
    elif csf_overlap >= min_overlap:
        classification = CLASS_PERIVENTRICULAR
    
    # 3. Check for juxtacortical/cortical
    # Either directly overlaps with GM or is adjacent to GM (dilated lesion overlaps with GM)
    elif (gm_overlap >= min_overlap) or (lesion_dilated_gm_overlap >= min_overlap):
        classification = CLASS_JUXTACORTICAL
    
    # 4. Default to white matter lesion
    else:
        classification = CLASS_WM
    
    return classification, overlap_stats


def main():
    """Main function."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    
    # Validate inputs and outputs
    assert_inputs_exist(parser, [args.in_lesions, args.in_wm_mask,
                                 args.in_gm_mask, args.in_csf_mask])
    assert_outputs_exist(parser, args, [args.out_classified, args.out_csv])
    assert_headers_compatible(parser, [args.in_lesions, args.in_wm_mask,
                                       args.in_gm_mask, args.in_csf_mask])
    
    # Load images
    logging.info("Loading lesion labels...")
    lesion_img = nib.load(args.in_lesions)
    lesion_data = get_data_as_labels(lesion_img)
    
    logging.info("Loading tissue masks...")
    wm_img = nib.load(args.in_wm_mask)
    wm_mask = get_data_as_mask(wm_img)
    
    gm_img = nib.load(args.in_gm_mask)
    gm_mask = get_data_as_mask(gm_img)
    
    csf_img = nib.load(args.in_csf_mask)
    csf_mask = get_data_as_mask(csf_img)
    
    # Get unique lesion labels (excluding background 0)
    lesion_ids = np.unique(lesion_data)
    lesion_ids = lesion_ids[lesion_ids != 0]
    
    logging.info(f"Found {len(lesion_ids)} lesions to classify.")
    
    # Initialize output
    classified_data = np.zeros_like(lesion_data, dtype=np.uint8)
    
    # Store results for CSV
    results = []
    
    # Classify each lesion
    for lesion_id in lesion_ids:
        logging.debug(f"Classifying lesion {lesion_id}...")
        
        # Extract lesion mask
        lesion_mask = (lesion_data == lesion_id)
        
        # Classify lesion
        classification, overlap_stats = classify_lesion(
            lesion_mask, wm_mask, gm_mask, csf_mask,
            min_overlap=args.min_overlap,
            wm_thresh=args.confluent_wm_threshold,
            csf_thresh=args.confluent_csf_threshold,
            gm_thresh=args.confluent_gm_threshold
        )
        
        # Store classification in output image
        classified_data[lesion_mask] = classification
        
        # Store results for CSV
        result = {
            'lesion_id': int(lesion_id),
            'classification': CLASS_NAMES[classification],
            'classification_code': classification,
            **overlap_stats
        }
        results.append(result)
        
        logging.info(f"Lesion {lesion_id}: {CLASS_NAMES[classification]}")
    
    # Save classified image
    logging.info(f"Saving classified lesion map to {args.out_classified}...")
    classified_img = nib.Nifti1Image(classified_data, lesion_img.affine,
                                     lesion_img.header)
    nib.save(classified_img, args.out_classified)
    
    # Create and save CSV report
    logging.info(f"Saving classification report to {args.out_csv}...")
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    columns_order = ['lesion_id', 'classification', 'classification_code',
                     'lesion_volume_voxels', 'wm_overlap_voxels',
                     'gm_overlap_voxels', 'csf_overlap_voxels',
                     'wm_percentage', 'gm_percentage', 'csf_percentage']
    df = df[columns_order]
    
    df.to_csv(args.out_csv, index=False, float_format='%.2f')
    
    # Print summary statistics
    logging.info("\n" + "="*60)
    logging.info("Classification Summary:")
    logging.info("="*60)
    for class_code, class_name in CLASS_NAMES.items():
        count = np.sum(df['classification_code'] == class_code)
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        logging.info(f"{class_name}: {count} lesions ({percentage:.1f}%)")
    logging.info("="*60)
    
    logging.info("Classification complete!")


if __name__ == "__main__":
    main()
