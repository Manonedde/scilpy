#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create an animated GIF showing all slices of a volume along the
specified axis. Each image is normalized using the same intensity window,
computed from the 1st/99th percentile of the non-zero voxels; unless
--imin/--imax are provided.

If the input image is 4D, a single volume is used to build the GIF (the
first volume by default; use --volume to select another one). Use
--across_volumes to instead build the GIF by animating across volumes at a
single fixed slice (0 to the last volume, by default).

Use --mask if the input is a binary (0/1) mask: this fixes the intensity
window to 0/1 instead of using the 1st/99th percentile of non-zero voxels
(which would otherwise be 1/1 for a binary image).

--------------------------------------------------------------------------------
Usage examples:
>>> scil_viz_volume_gif.py fa.nii.gz fa_axial_gray.gif
>>> scil_viz_volume_gif.py fa.nii.gz fa_axial_jet.gif --cmap jet
>>> scil_viz_volume_gif.py fa.nii.gz fa_sagittal.gif --axis sagittal --fps 20 --loop 2
>>> scil_viz_volume_gif.py fa.nii.gz fa_axial_crop.gif --min_slice 20 --max_slice 110
>>> scil_viz_volume_gif.py dwi.nii.gz dwi_b0.gif --volume 0
>>> scil_viz_volume_gif.py dwi.nii.gz dwi_sagittal_volumes.gif --axis sagittal \\
        --across_volumes --slice 60
>>> scil_viz_volume_gif.py brain_mask.nii.gz mask.gif --mask --cmap Reds
--------------------------------------------------------------------------------
"""

import argparse
import logging

import imageio.v2 as imageio
import matplotlib
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.utils.spatial import get_axis_index
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_image',
                   help='Input volume (.nii.gz).')
    p.add_argument('out_gif',
                   help='Output animated GIF filename (.gif).')

    p.add_argument('--axis', choices=['sagittal', 'coronal', 'axial'],
                   default='axial',
                   help='Axis along which to slice. [%(default)s]')

    g4d = p.add_argument_group('4D image options')
    g4d.add_argument('--volume', type=int, default=0,
                     help='If the input image is 4D, index of the volume '
                          'to use\nto build the GIF. Ignored for 3D images '
                          'and when\n--across_volumes is set. '
                          '[%(default)s]')
    g4d.add_argument('--across_volumes', action='store_true',
                     help='If the input image is 4D, build the GIF by '
                          'animating\nacross volumes at a single fixed '
                          'slice, instead of\nanimating across slices of a '
                          'single volume.')
    g4d.add_argument('--slice', type=int,
                     help='Slice index to use when --across_volumes is '
                          'set. Defaults\nto the middle slice along '
                          '--axis.')
    g4d.add_argument('--min_volume', type=int,
                     help='Minimum volume index to include when '
                          '--across_volumes is\nset (inclusive). Defaults '
                          'to 0.')
    g4d.add_argument('--max_volume', type=int,
                     help='Maximum volume index to include when '
                          '--across_volumes is\nset (inclusive). Defaults '
                          'to the last volume.')

    gmask = p.add_argument_group('Mask options')
    gmask.add_argument('--mask', action='store_true',
                       help='Treat the input as a binary (0/1) mask: fix '
                            'the intensity\nwindow to 0/1 instead of using '
                            'the 1st/99th percentile,\nand validate that '
                            'the image only contains 0s and 1s.')

    p.add_argument('--cmap', default='gray',
                   help='Matplotlib colormap used to render each slice.\n'
                        '[%(default)s]')
    p.add_argument('--fps', type=int, default=15,
                   help='Frames per second. [%(default)s]')
    p.add_argument('--loop', type=int, default=0,
                   help='Number of times to loop. [%(default)s] (means loop indefinitely)')
    p.add_argument('--min_slice', type=int,
                   help='Minimum slice index to include (inclusive). '
                        'Defaults to 0.')
    p.add_argument('--max_slice', type=int,
                   help='Maximum slice index to include (inclusive). '
                        'Defaults to the last slice.')
    p.add_argument('--imin', type=float,
                   help='Minimum intensity for normalization. If not set, '
                        'the 1st percentile\nof the non-zero voxels is '
                        'used.')
    p.add_argument('--imax', type=float,
                   help='Maximum intensity for normalization. If not set, '
                        'the 99th percentile\nof the non-zero voxels is '
                        'used.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image)
    assert_outputs_exist(parser, args, args.out_gif)

    if args.across_volumes and (args.min_slice is not None or
                                args.max_slice is not None):
        parser.error('--min_slice/--max_slice cannot be used together '
                     'with --across_volumes.')

    # Load image and get data
    vol_img = nib.load(args.in_image)
    data = vol_img.get_fdata()
    axis_index = get_axis_index(args.axis, vol_img.affine)

    if args.across_volumes:
        if data.ndim != 4:
            parser.error('--across_volumes can only be used with a 4D '
                         'input image.')
        n_volumes = data.shape[3]
        n_slices = data.shape[axis_index]

        # Set slice index, defaulting to the middle slice, within bounds
        slice_idx = args.slice if args.slice is not None else n_slices // 2
        if not 0 <= slice_idx <= n_slices - 1:
            parser.error('--slice must be between 0 and {} for the {} '
                         'axis.'.format(n_slices - 1, args.axis))

        # Set min/max volume indices, ensuring they are within bounds
        min_volume = args.min_volume if args.min_volume is not None else 0
        max_volume = args.max_volume if args.max_volume is not None \
            else n_volumes - 1
        if not 0 <= min_volume <= max_volume <= n_volumes - 1:
            parser.error('--min_volume/--max_volume must satisfy 0 <= '
                         'min_volume <= max_volume <= {}.'.format(
                             n_volumes - 1))

        logging.info('Rendering volumes {} to {} at slice {} ({} axis).'
                     .format(min_volume, max_volume, slice_idx, args.axis))

        # Reduce to the frames of interest: one slice per selected volume
        data = np.take(data[..., min_volume:max_volume + 1], slice_idx,
                       axis=axis_index)
        axis_index = None
    else:
        # Handle 4D images by selecting a single volume
        if data.ndim == 4:
            if not 0 <= args.volume <= data.shape[3] - 1:
                parser.error('--volume must be between 0 and {} for this '
                             '4D image.'.format(data.shape[3] - 1))
            logging.info('Input image is 4D ({} volumes), using volume {}.'
                         .format(data.shape[3], args.volume))
            data = data[..., args.volume]

    if args.mask:
        if args.imin is not None or args.imax is not None:
            parser.error('--imin/--imax cannot be used together with '
                         '--mask (the intensity window is fixed to 0/1).')
        unique_vals = np.unique(data)
        if not np.all(np.isin(unique_vals, [0, 1])):
            parser.error('--mask expects a binary (0/1) image, but found '
                         'values: {}.'.format(unique_vals))
        imin, imax = 0, 1
    else:
        # set default imin/imax to 1st/99th percentile of non-zero voxels if not provided
        imin = args.imin if args.imin is not None \
            else np.percentile(data[data > 0], 1)
        imax = args.imax if args.imax is not None \
            else np.percentile(data[data > 0], 99)

    # Set colormap
    colormap = matplotlib.colormaps[args.cmap]

    if args.across_volumes:
        # One frame per volume, at the fixed slice selected above
        frame_indices = range(data.shape[-1])
        frame_axis = data.ndim - 1
    else:
        # One frame per slice along the requested axis
        n_slices = data.shape[axis_index]
        min_slice = args.min_slice if args.min_slice is not None else 0
        max_slice = args.max_slice if args.max_slice is not None \
            else n_slices - 1
        if not 0 <= min_slice <= max_slice <= n_slices - 1:
            parser.error('--min_slice/--max_slice must satisfy 0 <= '
                         'min_slice <= max_slice <= {} for the {} axis.'
                         .format(n_slices - 1, args.axis))
        logging.info('Rendering slices {} to {} ({} axis).'.format(
            min_slice, max_slice, args.axis))
        frame_indices = range(min_slice, max_slice + 1)
        frame_axis = axis_index

    # Create frames and save as GIF
    frames = []
    for i in frame_indices:
        curr_slice = np.rot90(np.take(data, i, axis=frame_axis))
        background_mask = curr_slice <= 0
        normed = np.clip((curr_slice - imin) / (imax - imin), 0, 1)
        frame = (colormap(normed) * 255).astype(np.uint8)
        frame[background_mask] = [0, 0, 0, 255]
        frames.append(frame)
    # Save frames as gif
    imageio.mimsave(args.out_gif, frames, fps=args.fps, loop=args.loop)


if __name__ == '__main__':
    main()
