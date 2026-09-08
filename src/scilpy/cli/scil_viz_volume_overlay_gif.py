#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create an animated GIF showing all slices of a background volume along the
specified axis, with a second volume overlaid at a fixed opacity.

Both background and overlay are normalized independently using the 1st/99th
percentile of their non-zero voxels, unless --bg_imin/--bg_imax or
--overlay_imin/--overlay_imax are provided. Overlay voxels <= 0 are left
fully transparent (background shows through unmodified); voxels > 0 are
blended with the background at --alpha opacity.

If an input image is 4D, a single volume is used to build the GIF (the
first volume by default; use --bg_volume/--overlay_volume to select
another one).

Use --mask if the overlay is a binary (0/1) mask: this fixes the overlay
intensity window to 0/1 instead of using the 1st/99th percentile (which
would otherwise be 1/1 for a binary image).

Use --bg_rgb/--overlay_rgb if the corresponding image is a true-color RGB
map (e.g. a tensor/FA-RGB directionality map), i.e. a 4D image whose last
dimension holds 3 (or 4, RGB/RGBA) color channels rather than separate
volumes. The channels are rendered directly instead of through a colormap,
and --bg_cmap/--overlay_cmap, --bg_imin/--bg_imax, --overlay_imin/
--overlay_imax and --bg_volume/--overlay_volume are ignored for that image.

-----------------------------------------------------------------------------
Usage examples:
>>> scil_viz_volume_overlay_gif.py t1.nii.gz mask.nii.gz overlay.gif --mask
>>> scil_viz_volume_overlay_gif.py t1.nii.gz fa.nii.gz overlay.gif \\
        --overlay_cmap jet --alpha 0.5
>>> scil_viz_volume_overlay_gif.py t1.nii.gz stat_map.nii.gz overlay.gif \\
        --axis sagittal --alpha 0.4 --overlay_imin 2.3 --overlay_imax 8
>>> scil_viz_volume_overlay_gif.py rgb_map.nii.gz t1.nii.gz overlay.gif \\
        --bg_rgb --alpha 0.4
-----------------------------------------------------------------------------
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

    p.add_argument('in_background',
                   help='Background volume (.nii.gz), shown underneath.')
    p.add_argument('in_overlay',
                   help='Overlay volume (.nii.gz), blended on top of the '
                        'background.')
    p.add_argument('out_gif',
                   help='Output animated GIF filename (.gif).')

    p.add_argument('--axis', choices=['sagittal', 'coronal', 'axial'],
                   default='axial',
                   help='Axis along which to slice. [%(default)s]')
    p.add_argument('--alpha', type=float, default=0.5,
                   help='Opacity of the overlay, between 0 (invisible) '
                        'and 1\n(fully opaque). [%(default)s]')

    g4d = p.add_argument_group('4D image options')
    g4d.add_argument('--bg_volume', type=int, default=0,
                     help='If the background image is 4D, index of the '
                          'volume to use. \nIgnored for 3D images. '
                          '[%(default)s]')
    g4d.add_argument('--overlay_volume', type=int, default=0,
                     help='If the overlay image is 4D, index of the '
                          'volume\nto use. \nIgnored for 3D images. '
                          '[%(default)s]')

    gmask = p.add_argument_group('Mask options')
    gmask.add_argument('--mask', action='store_true',
                       help='Treat the overlay as a binary (0/1) mask: \nfix '
                            'its intensity window to 0/1 instead of using '
                            'the 1st/99th percentile, \nand validate that '
                            'it only contains 0s and 1s.')

    grgb = p.add_argument_group('RGB image options')
    grgb.add_argument('--bg_rgb', action='store_true',
                      help='Consider the background as a true-color RGB '
                           'image\n(4D, 3 or 4 channels in the last '
                           'dimension) and\nrender its channels directly '
                           'instead of through --bg_cmap.')
    grgb.add_argument('--overlay_rgb', action='store_true',
                      help='Consider the overlay as a true-color RGB image\n'
                           '(4D, 3 or 4 channels in the last dimension) '
                           'and\nrender its channels directly instead of '
                           'through --overlay_cmap. \nCannot be combined '
                           'with --mask.')

    p.add_argument('--bg_cmap', default='gray',
                   help='Matplotlib colormap used to render the '
                        'background.\n[%(default)s]')
    p.add_argument('--overlay_cmap', default='gray',
                   help='Matplotlib colormap used to render the overlay.\n'
                        '[%(default)s]')
    p.add_argument('--fps', type=int, default=15,
                   help='Frames per second. [%(default)s]')
    p.add_argument('--loop', type=int, default=0,
                   help='Number of times to loop. [%(default)s] (means '
                        'loop indefinitely)')
    p.add_argument('--min_slice', type=int,
                   help='Minimum slice index to include (inclusive). '
                        'Defaults to 0.')
    p.add_argument('--max_slice', type=int,
                   help='Maximum slice index to include (inclusive). '
                        'Defaults to the last slice.')
    p.add_argument('--bg_imin', type=float,
                   help='Minimum intensity for background normalization. '
                        '\nIf not set, the 1st percentile of its non-zero '
                        'voxels is used.')
    p.add_argument('--bg_imax', type=float,
                   help='Maximum intensity for background normalization. '
                        '\nIf not set, the 99th percentile of its non-zero '
                        'voxels is used.')
    p.add_argument('--overlay_imin', type=float,
                   help='Minimum intensity for overlay normalization. \nIf '
                        'not set, the 1st percentile of its non-zero '
                        'voxels is used.')
    p.add_argument('--overlay_imax', type=float,
                   help='Maximum intensity for overlay normalization. \nIf '
                        'not set, the 99th percentile of its non-zero '
                        'voxels is used.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _load_volume(parser, path, label, volume_idx, volume_flag, is_rgb,
                 rgb_flag):
    img = nib.load(path)
    data = img.get_fdata()

    if is_rgb:
        if data.ndim != 4 or data.shape[3] not in (3, 4):
            parser.error('{} expects a 4D image with 3 or 4 channels in '
                         'the last dimension (RGB[A]) for the {} image, '
                         'got shape {} for {}.'.format(
                             rgb_flag, label, data.shape, path))
        data = data[..., :3]
        if data.max() <= 1.0:
            data = data * 255.0
        data = np.clip(data, 0, 255)
        logging.info('{} image is RGB, shape {}.'.format(
            label, data.shape))
    elif data.ndim == 4:
        if not 0 <= volume_idx <= data.shape[3] - 1:
            parser.error('{} must be between 0 and {} for the {} image.'
                         .format(volume_flag, data.shape[3] - 1, label))
        logging.info('{} image is 4D ({} volumes), using volume {}.'
                     .format(label, data.shape[3], volume_idx))
        data = data[..., volume_idx]
    elif data.ndim != 3:
        parser.error('{} image must be a 3D or 4D image.'.format(label))

    return img, data


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_background, args.in_overlay])
    assert_outputs_exist(parser, args, args.out_gif)

    if not 0 <= args.alpha <= 1:
        parser.error('--alpha must be between 0 and 1.')
    if args.mask and args.overlay_rgb:
        parser.error('--mask cannot be used together with --overlay_rgb.')

    # Load background and overlay, reducing 4D images to a single volume
    # (unless --bg_rgb/--overlay_rgb is set, in which case the last
    # dimension is treated as color channels instead).
    bg_img, bg_data = _load_volume(parser, args.in_background, 'background',
                                   args.bg_volume, '--bg_volume',
                                   args.bg_rgb, '--bg_rgb')
    _, overlay_data = _load_volume(parser, args.in_overlay, 'overlay',
                                   args.overlay_volume, '--overlay_volume',
                                   args.overlay_rgb, '--overlay_rgb')

    if bg_data.shape[:3] != overlay_data.shape[:3]:
        parser.error('Background and overlay must have the same spatial '
                     'shape, got {} and {}.'.format(bg_data.shape[:3],
                                                     overlay_data.shape[:3]))

    axis_index = get_axis_index(args.axis, bg_img.affine)

    # Set background intensity window
    bg_imin = bg_imax = None
    if args.bg_rgb:
        if args.bg_imin is not None or args.bg_imax is not None:
            parser.error('--bg_imin/--bg_imax cannot be used together '
                         'with --bg_rgb.')
        if args.bg_volume != 0:
            logging.warning('--bg_volume is ignored because --bg_rgb is '
                            'set.')
        if args.bg_cmap != 'gray':
            logging.warning('--bg_cmap is ignored because --bg_rgb is '
                            'set.')
    else:
        bg_imin = args.bg_imin if args.bg_imin is not None \
            else np.percentile(bg_data[bg_data > 0], 1)
        bg_imax = args.bg_imax if args.bg_imax is not None \
            else np.percentile(bg_data[bg_data > 0], 99)

    # Set overlay intensity window
    overlay_imin = overlay_imax = None
    if args.overlay_rgb:
        if args.overlay_imin is not None or args.overlay_imax is not None:
            parser.error('--overlay_imin/--overlay_imax cannot be used '
                         'together with --overlay_rgb.')
        if args.overlay_volume != 0:
            logging.warning('--overlay_volume is ignored because '
                            '--overlay_rgb is set.')
        if args.overlay_cmap != 'hot':
            logging.warning('--overlay_cmap is ignored because '
                            '--overlay_rgb is set.')
    elif args.mask:
        if args.overlay_imin is not None or args.overlay_imax is not None:
            parser.error('--overlay_imin/--overlay_imax cannot be used '
                         'together with --mask (the overlay intensity '
                         'window is fixed to 0/1).')
        unique_vals = np.unique(overlay_data)
        if not np.all(np.isin(unique_vals, [0, 1])):
            parser.error('--mask expects a binary (0/1) overlay, but '
                         'found values: {}.'.format(unique_vals))
        overlay_imin, overlay_imax = 0, 1
    else:
        overlay_imin = args.overlay_imin if args.overlay_imin is not None \
            else np.percentile(overlay_data[overlay_data > 0], 1)
        overlay_imax = args.overlay_imax if args.overlay_imax is not None \
            else np.percentile(overlay_data[overlay_data > 0], 99)

    bg_colormap = matplotlib.colormaps[args.bg_cmap]
    overlay_colormap = matplotlib.colormaps[args.overlay_cmap]
    n_slices = bg_data.shape[axis_index]

    min_slice = args.min_slice if args.min_slice is not None else 0
    max_slice = args.max_slice if args.max_slice is not None \
        else n_slices - 1
    if not 0 <= min_slice <= max_slice <= n_slices - 1:
        parser.error('--min_slice/--max_slice must satisfy 0 <= min_slice '
                     '<= max_slice <= {} for the {} axis.'.format(
                         n_slices - 1, args.axis))

    logging.info('Rendering slices {} to {} ({} axis), overlay alpha={}.'
                 .format(min_slice, max_slice, args.axis, args.alpha))

    # Create frames by compositing the overlay on top of the background
    frames = []
    for i in range(min_slice, max_slice + 1):
        bg_slice = np.rot90(np.take(bg_data, i, axis=axis_index))
        overlay_slice = np.rot90(np.take(overlay_data, i, axis=axis_index))

        if args.bg_rgb:
            bg_rgb = bg_slice.astype(np.float32)
            bg_rgb[np.all(bg_slice <= 0, axis=-1)] = 0
        else:
            bg_normed = np.clip((bg_slice - bg_imin) / (bg_imax - bg_imin),
                                0, 1)
            bg_rgb = (bg_colormap(bg_normed)[..., :3]
                     * 255).astype(np.float32)
            bg_rgb[bg_slice <= 0] = 0

        if args.overlay_rgb:
            overlay_rgb = overlay_slice.astype(np.float32)
            overlay_signal = np.any(overlay_slice > 0, axis=-1)
        else:
            overlay_normed = np.clip(
                (overlay_slice - overlay_imin)
                / (overlay_imax - overlay_imin), 0, 1)
            overlay_rgb = (overlay_colormap(overlay_normed)[..., :3]
                          * 255).astype(np.float32)
            overlay_signal = overlay_slice > 0

        # Only blend the overlay where it has signal; elsewhere show the
        # background unmodified.
        alpha_map = np.where(overlay_signal, args.alpha, 0.0)[..., None]
        blended = bg_rgb * (1 - alpha_map) + overlay_rgb * alpha_map

        frame = np.dstack(
            [blended, np.full(blended.shape[:2], 255)]).astype(np.uint8)
        frames.append(frame)

    # Save gif
    imageio.mimsave(args.out_gif, frames, fps=args.fps, loop=args.loop)


if __name__ == '__main__':
    main()
