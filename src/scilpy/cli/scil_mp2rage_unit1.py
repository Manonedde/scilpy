#!/usr/bin/env python3
"""
Compute an MP2RAGE UNIT1 image, following Marques
et al. 2010 (NeuroImage), with the optional noise-regularization term
from O'Brien et al. 2014. Equivalent to the original MATLAB
combination script from  https://github.com/JosePMarques/MP2RAGE-related-scripts
(same formula, same integer rescale by default); 
verified numerically identical up to floating-point noise.

Three input `modes` are supported: --mag|--phase (Philips, combined 4D),
--inv1_mag|--inv1_phase|--inv2_mag|--inv2_phase (Philips, already split into
separate 3D files, e.g. BIDS inv-1/inv-2 part-mag/part-phase MP2RAGE outputs),
and --inv1|2_real|imag (Siemens). Mutually exclusive.

1) Combined 4D magnitude+phase files (e.g. Philips dcm2niix output
   where volume 0 = INV1 and volume 1 = INV2):

   scil_mp2rage_unit1_.py \
       --mag *MP2RAGE*nii.gz \
       --phase *MP2RAGE*ph.nii.gz \
       -o sub-XXX_ses-XXX_UNIT1.nii.gz

2) Separate 3D magnitude/phase files per inversion (e.g. already-split BIDS
   inv-1/inv-2 part-mag/part-phase MP2RAGE files):

   scil_mp2rage_unit1_.py \
       --inv1_mag sub-XXX_inv-1_part-mag_MP2RAGE.nii.gz \
       --inv1_phase sub-XXX_inv-1_part-phase_MP2RAGE.nii.gz \
       --inv2_mag sub-XXX_inv-2_part-mag_MP2RAGE.nii.gz \
       --inv2_phase sub-XXX_inv-2_part-phase_MP2RAGE.nii.gz \
       -o sub-XXX_ses-XXX_UNIT1.nii.gz

3) Separate 3D real/imaginary files per inversion (Siemens dcm2niix output):

   python scil_compute_unit1_mp2rage.py \
       --inv1_real *MP2RAGE*_301_real.nii.gz --inv1_imag *MP2RAGE*_301_imaginary.nii.gz \
       --inv2_real *MP2RAGE*_302_real.nii.gz --inv2_imag *MP2RAGE*_302_imaginary.nii.gz \
       -o sub-XXX_ses-XXX_UNIT1.nii.gz

To remove background noise like MATLAB's RobustCombination.m
(regularization = 10; RobustCombination(MP2RAGE, regularization)),
pass --regularization 10 (!NOT! --beta 10 --regularization is
auto-converted to the equivalent beta from a background corner of INV2,
matching RobustCombination.m exactly).

WARNING : to run denoise_mp2rage.py you need to add --rescale_int option to avoid the 
          [-0.5,0.5] rescaling of UNIT1 image (denoising do not work with it)

----------------------------------
References: 
    Marques JP, et al. (2010) MP2RAGE, a self bias-field corrected sequence for improved segmentation 
    and T1-mapping at high field. Neuroimage. 2010 Jan 15;49(2):1271-81
    https://pubmed.ncbi.nlm.nih.gov/19819338/
        https://github.com/JosePMarques/MP2RAGE-related-scripts

    O'Brien KR, et al. (2014) Robust T1-weighted structural brain imaging and morphometry at 7T 
    using MP2RAGE. PLoS One. 2014;9(6):e996, https://pmc.ncbi.nlm.nih.gov/articles/PMC4059664/
----------------------------------

"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg)
from scilpy.version import version_string


def to_radians(phase_data: np.ndarray) -> np.ndarray:
    """Rescale phase data to [-pi, pi] if it isn't already."""
    pmin, pmax = np.nanmin(phase_data), np.nanmax(phase_data)
    if pmax - pmin > 2 * np.pi + 0.1:
        return (phase_data - pmin) / (pmax - pmin) * 2 * np.pi - np.pi
    return phase_data


def compute_unit1_complex(inv1: np.ndarray, inv2: np.ndarray,
                           beta: float = 0.0) -> np.ndarray:
    """
    Combine two inversion-time complex images into an MP2RAGE UNIT1 image.

    Standard formula (Marques et al. 2010):
        UNIT1 = Re( conj(INV1) * INV2 ) / (|INV1|^2 + |INV2|^2)

    With beta > 0, the robust/regularized combination from
    O'Brien et al. 2014 is used instead, which suppresses noise
    amplification in background/low-signal voxels:
        UNI = Re( conj(INV1) * INV2 - beta ) / (|INV1|^2 + |INV2|^2 + 2*beta)

    Output range is [-0.5, 0.5].

    params : 
        inv1 : np.ndarray
            Complex-valued INV1 image (e.g. INV1 magnitude * exp(i * INV1 phase))
        inv2 : np.ndarray
            Complex-valued INV2 image (e.g. INV2 magnitude * exp(i * INV2 phase))
        beta : float
            Regularization term for robust background-noise removal (O'Brien et al. 2014). 
            0 = standard Marques formula (default).

    retruns :
        np.ndarray
            UNIT1 image, same shape as inv1/inv2
            range [-0.5, 0.5] (default)
            or rescaled to int16 if --rescale_int is used
    """
    numerator = np.real(np.conj(inv1) * inv2) - beta
    denominator = np.abs(inv1) ** 2 + np.abs(inv2) ** 2 + 2 * beta
    # deal with nondivide zero
    with np.errstate(invalid="ignore", divide="ignore"):
        unit1 = numerator / denominator
    unit1 = np.nan_to_num(unit1, nan=0.0, posinf=0.0, neginf=0.0)
    return unit1


def load_pair(mag_path, phase_path, vol_index=None):
    """
    Load a magnitude/phase NIfTI pair (Philips).
    
    params : 
        mag_path: 
            Path to the magnitude NIfTI file
        phase_path: 
            Path to the phase NIfTI file
        vol_index: 
            Index of the volume to load (if None, all volumes are loaded)
    return: 
        np.ndarray, np.ndarray, nib.Nifti1Image
            Tuple of (magnitude array, phase array in radians, reference image for affine/header)
    """
    mag_img = nib.load(mag_path)
    phase_img = nib.load(phase_path)
    mag = mag_img.get_fdata()
    phase = phase_img.get_fdata()
    if vol_index is not None:
        mag = mag[..., vol_index]
        phase = phase[..., vol_index]
    return mag, to_radians(phase), mag_img


def load_complex(real_path, imag_path):
    """Load a real/imaginary NIfTI pair.
    
    params : 
        real_path: 
            Path to the real-part NIfTI file
        imag_path: 
            Path to the imaginary-part NIfTI file
    return: 
        np.ndarray, nib.Nifti1Image
            Tuple of (complex array, reference image for affine/header)
    """
    real_img = nib.load(real_path)
    imag_img = nib.load(imag_path)
    return real_img.get_fdata() + 1j * imag_img.get_fdata(), real_img


def estimate_beta(inv2_mag: np.ndarray, regularization: float) -> float:
    """
    Not recommanded !
    Auto-estimate beta from a background corner of INV2, replicating
    RobustCombination.m (O'Brien et al. 2014 / Marques' official toolbox):

        noiselevel = regularization * mean(INV2(:, end-10:end, end-10:end))
        beta = noiselevel^2

    `regularization` is the "multiplyingFactor" passed to RobustCombination.m
    (e.g. 10 in typical usage) -- it is NOT beta itself.

    Caveat: the background corner is the last 11 voxels along axes 2 and 3
    (all of axis 1), same as the MATLAB script. This assumes that corner of
    the FOV is air/background, which depends on subject positioning and
    image orientation -- inspect the result and adjust if it looks wrong.
    If that corner is exact zero-padding (mean == 0, e.g. a masked-out FOV
    edge rather than real background noise), MATLAB falls back to
    noiselevel = regularization directly -- replicated here too.

    params : 
        inv2_mag : np.ndarray
            Magnitude of the INV2 image (abs(INV2))
        regularization : float
            Regularization factor (multiplyingFactor in Marques' RobustCombination.m)
    return :
        float
            Estimated beta value for robust combination formula 
    """
    corner_mean = np.mean(inv2_mag[:, -11:, -11:])
    noiselevel = regularization * corner_mean if corner_mean != 0 else regularization
    return noiselevel ** 2


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mag",
                      help="Mode 1: 4D magnitude file, volume 0=INV1, volume 1=INV2 "
                           "(requires --phase)")
    mode.add_argument("--inv1_mag",
                      help="Mode 2: INV1 magnitude file, separate 3D volume "
                           "(requires --inv1_phase/--inv2_mag/--inv2_phase)")
    mode.add_argument("--inv1_real",
                      help="Mode 3: INV1 real-part file "
                           "(requires --inv1_imag/--inv2_real/--inv2_imag)")

    p.add_argument("--phase",
                   help="Mode 1: 4D phase file, volume 0=INV1, volume 1=INV2")
    p.add_argument("--inv1_phase",
                   help="Mode 2: INV1 phase file, separate 3D volume")
    p.add_argument("--inv2_mag",
                   help="Mode 2: INV2 magnitude file, separate 3D volume")
    p.add_argument("--inv2_phase",
                   help="Mode 2: INV2 phase file, separate 3D volume")
    p.add_argument("--inv1_imag",
                   help="Mode 3: INV1 imaginary-part file")
    p.add_argument("--inv2_real",
                   help="Mode 3: INV2 real-part file")
    p.add_argument("--inv2_imag",
                   help="Mode 3: INV2 imaginary-part file")

    p.add_argument("-o", "--output", required=True, 
                   help="Output UNIT1 NIfTI path")

    reg_group = p.add_mutually_exclusive_group()
    reg_group.add_argument("--beta", type=float, default=0.0,
                            help="Explicit regularization term for robust background-noise "
                                 "removal (O'Brien et al. 2014). 0 = standard Marques formula "
                                 "(default, matches MP2RAGE_nii.m). This is the literal beta in "
                                 "the formula -- for RobustCombination.m-style regularization, "
                                 "use --regularization instead.")
    reg_group.add_argument("--regularization", type=float,
                            help="Auto-estimate beta from a background corner of INV2, exactly "
                                 "like MATLAB's RobustCombination.m: "
                                 "beta = (regularization * mean(background corner of INV2))^2. "
                                 "Pass the same value used as 'regularization' in MATLAB "
                                 "(e.g. 10) -- it is NOT beta itself.")

    p.add_argument("--rescale_int", action="store_true",
                   help="Rescale output from [-0.5, 0.5] to an integer range and save as "
                        "int16, to be use for AMBRA when you use denoise_mp2rage.py: "
                        "UNI*4096 + 2047 (approx. -1 to 4095, unclamped).")

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Load pair cases (Philips) and convert to complex-valued INV1/INV2
    if not args.phase:
        parser.error("--mag requires --phase.")
        mag1, phase1, ref_img = load_pair(args.mag, args.phase, vol_index=0)
        # use _ because we don't need the reference image again (same for both volumes)
        mag2, phase2, _ = load_pair(args.mag, args.phase, vol_index=1)
        # convert to complex-valued INV1/INV2
        inv1 = mag1 * np.exp(1j * phase1)
        inv2 = mag2 * np.exp(1j * phase2)
    elif args.inv1_mag:
        # Load already-split 3D magnitude/phase pairs and convert to
        # complex-valued INV1/INV2 (bids-converted version compatible)
        if not all([args.inv1_phase, args.inv2_mag, args.inv2_phase]):
            parser.error("--inv1_mag requires --inv1_phase, --inv2_mag and --inv2_phase.")
        mag1, phase1, ref_img = load_pair(args.inv1_mag, args.inv1_phase)
        mag2, phase2, _ = load_pair(args.inv2_mag, args.inv2_phase)
        inv1 = mag1 * np.exp(1j * phase1)
        inv2 = mag2 * np.exp(1j * phase2)
    else:
        # Load complex cases (Siemens) and convert to complex-valued INV1/INV2
        if not all([args.inv1_imag, args.inv2_real, args.inv2_imag]):
            parser.error("--inv1_real requires --inv1_imag, --inv2_real and --inv2_imag.")
        inv1, ref_img = load_complex(args.inv1_real, args.inv1_imag)
        inv2, _ = load_complex(args.inv2_real, args.inv2_imag)

    if args.regularization is not None:
        beta = estimate_beta(np.abs(inv2), args.regularization)
        print(f"Auto-estimated beta = {beta:.6g} from --regularization {args.regularization:g}")
        logging.info(f"NOTE ::: Check image to ensure it is background noise, not zero-padding.")
    else:
        beta = args.beta

    # compute UNIT1 image using the complex-valued formula
    unit1 = compute_unit1_complex(inv1, inv2, beta=beta)

    # Save output, optionally rescaling to int16
    if args.rescale_int:
        unit1 = np.round(unit1 * 4096 + 2047).astype(np.int16)
        out_img = nib.Nifti1Image(unit1, ref_img.affine, header=ref_img.header)
        out_img.header.set_data_dtype(np.int16)
    else:
        out_img = nib.Nifti1Image(unit1.astype(np.float32), ref_img.affine)

    nib.save(out_img, args.output)
    print(f"Saved UNIT1 image to {args.output}  (shape={unit1.shape}, "
          f"range=[{unit1.min():.3g}, {unit1.max():.3g}])")


if __name__ == "__main__":
    main()
