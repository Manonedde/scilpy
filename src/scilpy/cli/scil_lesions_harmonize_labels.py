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

-----------------------------------------------------------------------------
Manon : Add final harmonization step to ensure consistent labels across all
sessions. For exemple if a lesion is labeled 1 in session 1 and 2, but 3 in session 3,
it will be relabeled to 1 in session 3.
Manon : Add generation of a CSV report tracking lesion changes across sessions.
-----------------------------------------------------------------------------
"""

import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd

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
    p.add_argument('--debug_mode', action='store_true',
                   help='Add a fake voxel to the corner to ensure consistent '
                        'colors in MI-Brain.')

    add_overwrite_arg(p)
    return p


# -------------------------------------------------------------------------
# Manon - ADD NEW FUNCTION: Final harmonization across all sessions
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
# Manon - ADD NEW FUNCTION: Generate harmonization report
# -------------------------------------------------------------------------
def generate_harmonization_report(original_data, harmonized_data, output_dir, affine):
    """
    Generate a CSV report tracking lesion changes across sessions.
    
    Parameters
    ----------
    original_data : list of np.ndarray
        Original label arrays before harmonization.
    harmonized_data : list of np.ndarray
        Label arrays after harmonization.
    output_dir : str
        Output directory for the CSV report.
    affine : np.ndarray
        4x4 affine transformation matrix for volume calculation.
    """
    n_sessions = len(original_data)
    voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))
    
    # Collect all unique lesion IDs across all sessions
    all_lesion_ids = set()
    for data in harmonized_data:
        all_lesion_ids.update(np.unique(data[data > 0]))
    
    rows = []
    
    for lesion_id in sorted(all_lesion_ids):
        lesion_id = int(lesion_id)
        row = {'Lesion_ID': lesion_id}
        
        # Track presence and voxel count for each session
        for session_idx in range(n_sessions):
            original_voxels = np.sum(original_data[session_idx] == lesion_id)
            harmonized_voxels = np.sum(harmonized_data[session_idx] == lesion_id)
            
            row[f'Session_{session_idx}_Original_Voxels'] = int(original_voxels)
            row[f'Session_{session_idx}_Harmonized_Voxels'] = int(harmonized_voxels)
            row[f'Session_{session_idx}_Volume_mm3'] = float(harmonized_voxels * voxel_volume)
            
            # Check if label changed
            if original_voxels != harmonized_voxels:
                row[f'Session_{session_idx}_Changed'] = True
            else:
                row[f'Session_{session_idx}_Changed'] = False
        
        # First and last appearance
        first_appearance = None
        last_appearance = None
        for session_idx in range(n_sessions):
            if np.sum(harmonized_data[session_idx] == lesion_id) > 0:
                if first_appearance is None:
                    first_appearance = session_idx
                last_appearance = session_idx
        
        row['First_Session'] = first_appearance
        row['Last_Session'] = last_appearance
        row['Total_Voxels'] = int(np.sum([np.sum(harmonized_data[i] == lesion_id) 
                                           for i in range(n_sessions)]))
        row['Total_Volume_mm3'] = float(row['Total_Voxels'] * voxel_volume)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'lesion_harmonization_report.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Harmonization report saved to: {csv_path}")
    
    return csv_path


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

    # Final consistency harmonization across all sessions
    relabeled_data = final_label_harmonization(relabeled_data)

    max_label = np.max(relabeled_data) + 1
    for i, img in enumerate(imgs):
        if args.debug_mode:
            relabeled_data[i][0, 0, 0] = max_label  # To force identical color
        nib.save(nib.Nifti1Image(relabeled_data[i], img.affine),
                 os.path.join(args.out_dir, os.path.basename(args.in_images[i])))
    
    # Generate harmonization report
    generate_harmonization_report(original_data, relabeled_data, args.out_dir, imgs[0].affine)


if __name__ == "__main__":
    main()


