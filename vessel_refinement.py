import argparse
import numpy as np
from PIL import Image
from copy import deepcopy
from pathlib import Path

from skimage.transform import resize
from skimage.filters import threshold_li, threshold_otsu


# CONFIG
CLASS_TO_COLOR = {
    1: [238, 130, 238],  # violet (artery)
    2: [135, 206, 235],  # sky blue (vein)
    3: [180, 180, 0],    # dark yellow (crossing)
}

SECONDARY_TO_PRIMARY = {
    (238, 130, 238): (255, 0, 0),   # artery
    (135, 206, 235): (0, 0, 255),   # vein
    (180, 180, 0): (0, 255, 0),     # crossing
}


# UTILITIES
def assert_exists(path: Path):
    assert path.exists(), f"[ERROR] Path does not exist: {path}"


def load_image(path: Path):
    return np.array(Image.open(path))


def load_prob(path: Path):
    return np.load(path)


def resize_vessel(vessel, use_otsu=False):
    vessel = vessel.astype(np.float32)

    vessel_resized = resize(
        vessel,
        output_shape=(1024, 1024),
        order=3,
        anti_aliasing=True
    )

    if use_otsu:
        thr = threshold_otsu(vessel_resized)
    else:
        thr = threshold_li(vessel_resized)

    return vessel_resized > thr


def ensure_rgb(img):
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    return img[:, :, :3]


# PREREFINE (single sample)
def prerefine(prob_path: Path,
              vessel_path: Path,
              use_otsu: bool = False,
              remove_fp: bool = True,
              add_fns: bool = True):

    assert_exists(prob_path)
    assert_exists(vessel_path)

    prob = load_prob(prob_path)          # (4, H, W)
    vessel = load_image(vessel_path)

    vessel_mask = resize_vessel(vessel, use_otsu=use_otsu)

    # STEP 1: argmax prediction
    prob_no_bg = prob[1:]
    pred = np.argmax(prob_no_bg, axis=0) + 1  # (H, W)

    h, w = pred.shape
    seg = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in CLASS_TO_COLOR.items():
        seg[pred == cls] = color

    # STEP 2: remove FP
    if remove_fp:
        seg[~vessel_mask] = [0, 0, 0]

    # STEP 3: add FN (optional soft assignment)
    if add_fns:
        background_pixels = np.all(seg == [0, 0, 0], axis=-1)
        candidates = np.logical_and(background_pixels, vessel_mask)

        pred_classes = np.argmax(prob_no_bg, axis=0) + 1

        for cls, color in CLASS_TO_COLOR.items():
            seg[np.logical_and(candidates, pred_classes == cls)] = color

    return seg


# REFINE (single sample)
def refine(prerefined_path: Path,
           annotation_path: Path,
           vessel_path: Path,
           use_otsu: bool = False):

    assert_exists(prerefined_path)
    assert_exists(annotation_path)
    assert_exists(vessel_path)

    prerefined = load_image(prerefined_path)
    annotation = ensure_rgb(load_image(annotation_path))
    vessel = load_image(vessel_path)

    vessel_mask = resize_vessel(vessel, use_otsu=use_otsu)

    # STEP 1: mask annotation
    annotation = annotation * vessel_mask[:, :, None]

    final = deepcopy(prerefined)

    manual_mask = np.any(annotation != [0, 0, 0], axis=-1)
    final[manual_mask] = annotation[manual_mask]

    # STEP 2: convert secondary → primary
    non_manual = ~manual_mask

    for color_from, color_to in SECONDARY_TO_PRIMARY.items():
        color_from = np.array(color_from)

        mask = np.logical_and.reduce((
            np.all(prerefined == color_from, axis=-1),
            vessel_mask,
            non_manual
        ))

        final[mask] = color_to

    # sanity check
    for c in SECONDARY_TO_PRIMARY.keys():
        c = np.array(c)
        assert not np.any(
            np.logical_and(np.all(final == c, axis=-1), vessel_mask)
        ), f"Unmapped secondary color found: {c}"

    return final


# CLI
def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode", required=True)

    #prerefine
    p_pre = subparsers.add_parser("prerefine")
    p_pre.add_argument("--prob", type=str, required=True)
    p_pre.add_argument("--vessel", type=str, required=True)
    p_pre.add_argument("--out", type=str, required=True)
    p_pre.add_argument("--otsu", action="store_true", default=False)

    #refine
    p_ref = subparsers.add_parser("refine")
    p_ref.add_argument("--prerefined", type=str, required=True)
    p_ref.add_argument("--annotation", type=str, required=True)
    p_ref.add_argument("--vessel", type=str, required=True)
    p_ref.add_argument("--out", type=str, required=True)
    p_ref.add_argument("--otsu", action="store_true", default=False)

    args = parser.parse_args()

    if args.mode == "prerefine":
        result = prerefine(
            Path(args.prob),
            Path(args.vessel),
            use_otsu=args.otsu
        )

    elif args.mode == "refine":
        result = refine(
            Path(args.prerefined),
            Path(args.annotation),
            Path(args.vessel),
            use_otsu=args.otsu
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(result.astype(np.uint8), mode="RGB").save(out_path)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()