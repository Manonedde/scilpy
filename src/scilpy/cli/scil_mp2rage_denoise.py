#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to denoise the UNIT1 MP2RAGE background image.

Usage example : 

    scil_mp2rage_denoise.py *UNIT1*.nii.gz *inv-2*.nii.gz --output *_UNIDEN.nii.gz    

"""

import argparse
import os
import ants

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg)

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog='')
    p.add_argument('in_uni',
                   help='Path of the input UNIT1 image.')
    p.add_argument('in_inv2',
                   help='Path of the input inversion recovery image 2.')
    p.add_argument('--output',
                   help='Path of the output file.'
                   '\nIf not provided, the output will be saved in the current directory '
                   'with the same name as the input UNIT1 image using UNIDEN as custom suffix.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.basename(args.in_uni).replace('_UNIT1', '_UNIDEN')
        
    unit1 = ants.image_read(args.in_uni)
    inv2 = ants.image_read(args.in_inv2)
    
    # If inv2 is 4D, take the second 3D volume
    if inv2.dimension == 4:
        inv2 = ants.slice_image(inv2, axis=3, idx=1)
        print("Your input is 4D. Removing the fourth dimension.")

    inv2_n4 = ants.abp_n4(inv2)
    inv2_n4_norm = ants.iMath(inv2_n4, "Normalize")
    try:
        unit1_inv2 = unit1 * inv2_n4_norm
    except ValueError:
        inv2_n4_norm = ants.resample_image_to_target(inv2_n4_norm, unit1)
        unit1_inv2 = unit1 * inv2_n4_norm
    unit1_inv2.to_filename(f"{args.output}")


if __name__ == "__main__":
    main()
