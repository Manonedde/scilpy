#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The NAWM (Normal Appearing White Matter) is the white matter that is
neighboring a lesion. It is used to compute metrics in the white matter
surrounding lesions.

This script generates NAWM that is valid across multiple longitudinal sessions.
Given lesion label maps from all sessions and a white matter mask, it ensures
that the generated NAWM:
1. Is within the white matter (WM mask)
2. Does NOT contain any lesions from ANY session
3. Maintains a minimum distance (default 2 mm) from lesions in ALL sessions
4. Is consistent across all sessions for longitudinal analysis

The output is a 3D binary mask of valid NAWM voxels that satisfy all constraints.

WARNING: Voxels must be isotropic.
"""

import argparse

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

from scilpy.image.labels import get_data_as_labels
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_lesion_maps', nargs='+',
                   help='Lesion label maps from all sessions (.nii.gz).\n'
                        'All images must be co-registered and have the same shape.\n'
                        'Sessions should be in chronological order.')
    p.add_argument('wm_mask',
                   help='White matter mask (.nii.gz). NAWM will be computed within this mask.')
    p.add_argument('out_image',
                   help='Output NAWM mask file (.nii.gz).')

    p.add_argument('--min_distance_mm', type=float, default=2.0,
                   help='Minimum distance in millimeters between NAWM and any '
                        'lesions across all sessions [%(default)s].')

    add_overwrite_arg(p)

    return p


def combine_lesion_maps(lesion_maps):
    """
    Combine lesion maps from all sessions into a single union map.
    
    Parameters
    ----------
    lesion_maps : list of np.ndarray
        List of 3D lesion label arrays from all sessions.
    
    Returns
    -------
    combined_lesions : np.ndarray
        3D binary array where 1 indicates any lesion from any session.
    """
    combined = np.zeros_like(lesion_maps[0], dtype=bool)
    for lesion_map in lesion_maps:
        combined |= (lesion_map > 0)
    
    return combined.astype(np.uint8)


def apply_distance_constraint(nawm_rings, combined_lesions, affine, min_distance_mm):
    """
    Remove NAWM voxels that are too close to any lesion in any session.
    
    Parameters
    ----------
    nawm_rings : np.ndarray
        3D NAWM ring array.
    combined_lesions : np.ndarray
        3D binary array of all lesions from all sessions.
    affine : np.ndarray
        4x4 affine transformation matrix.
    min_distance_mm : float
        Minimum distance in millimeters from lesions.
    
    Returns
    -------
    constrained_nawm : np.ndarray
        NAWM rings with distance constraint applied.
    """
    constrained_nawm = nawm_rings.copy()
    
    # Get voxel size from affine
    voxel_size = np.abs(np.diag(affine)[:3])
    
    if not np.allclose(voxel_size, np.mean(voxel_size)):
        raise ValueError('Voxels must be isotropic.')
    
    voxel_size_mm = np.mean(voxel_size)
    
    # Compute distance transform from lesions
    # distance_transform_edt returns distance in voxels
    distance_from_lesions = distance_transform_edt(combined_lesions == 0)
    
    # Convert voxel distance to mm
    distance_from_lesions_mm = distance_from_lesions * voxel_size_mm
    
    # Remove NAWM voxels too close to lesions
    too_close = (constrained_nawm > 0) & (distance_from_lesions_mm < min_distance_mm)
    constrained_nawm[too_close] = 0
    
    return constrained_nawm


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.min_distance_mm < 0:
        parser.error('The minimum distance must be >= 0.')

    assert_inputs_exist(parser, args.in_lesion_maps, args.wm_mask)
    assert_outputs_exist(parser, args, args.out_image)

    # Load all lesion maps
    lesion_imgs = [nib.load(img_path) for img_path in args.in_lesion_maps]
    lesion_maps = [get_data_as_labels(img) for img in lesion_imgs]
    
    # Check that all images have same shape and affine
    ref_shape = lesion_maps[0].shape
    ref_affine = lesion_imgs[0].affine
    voxel_size = lesion_imgs[0].header.get_zooms()
    
    for i, lesion_map in enumerate(lesion_maps[1:], 1):
        if lesion_map.shape != ref_shape:
            raise ValueError(f'Image {args.in_lesion_maps[i]} has different shape '
                           f'({lesion_map.shape}) than first image ({ref_shape}).')
        if not np.allclose(lesion_imgs[i].affine, ref_affine):
            raise ValueError(f'Image {args.in_lesion_maps[i]} has different affine '
                           'than first image.')

    if not np.allclose(voxel_size, np.mean(voxel_size)):
        raise ValueError('Voxels must be isotropic.')

    # Combine all lesion maps from all sessions
    combined_lesions = combine_lesion_maps(lesion_maps)
    
    # Load white matter mask
    wm_img = nib.load(args.wm_mask)
    wm_data = get_data_as_mask(wm_img)
    
    # Check WM mask has same shape and affine
    if wm_data.shape != ref_shape:
        raise ValueError(f'WM mask has different shape ({wm_data.shape}) '
                        f'than lesion maps ({ref_shape}).')
    if not np.allclose(wm_img.affine, ref_affine):
        raise ValueError('WM mask has different affine than lesion maps.')
    
    # Create NAWM: white matter that is NOT lesion
    nawm = wm_data.copy()
    nawm[combined_lesions > 0] = 0
    
    # Apply distance constraint based on all sessions
    if args.min_distance_mm > 0:
        nawm = apply_distance_constraint(nawm, combined_lesions, 
                                        lesion_imgs[0].affine, args.min_distance_mm)
    
    nib.save(nib.Nifti1Image(nawm, lesion_imgs[0].affine), args.out_image)


if __name__ == "__main__":
    main()
