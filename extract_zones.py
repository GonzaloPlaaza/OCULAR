import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from copy import deepcopy
import pathlib
import argparse
from skimage.morphology import skeletonize
from utils.GeometricalVBMs import GeometricalVBMs
from utils.DiscSegmenter import DiscSegmenter   
from utils.junctions_utils import handle_interpoints, compute_od, extract_crossings_roi, extract_major_vessels


#Utility functions
def check_required_folders(root):
    required = ["images", "artery", "vein", "optic_disc", "crossings"]
    for r in required:
        if not (root / r).exists():
            raise FileNotFoundError(f"Missing required folder: {r}")
    print("All required folders found.")


def load_data(image_path, artery_path, vein_path, od_path, crossings_path=None):
    
    image = np.array(Image.open(image_path).convert("RGB"))
    artery = np.array(Image.open(artery_path).convert("L")) / 255
    vein = np.array(Image.open(vein_path).convert("L")) / 255
    crossings = np.array(Image.open(crossings_path).convert("L")) / 255
    optic_disc = np.array(Image.open(od_path))

    return image, artery, vein, optic_disc, crossings


def get_skeleton(vessel_mask):
    #Deepcopy vessel_mask so it does not get modified
    vessel_mask_copy = deepcopy(vessel_mask)
    skeleton = skeletonize(vessel_mask_copy).astype(np.uint8)
    return skeleton

def process_dataset(root_dir: str,
                    image_type: str = "ODC",
                    fw: int = None):
    
    root = Path(root_dir)
    check_required_folders(root)

    image_dir = root / "images"
    artery_dir = root / "artery"
    vein_dir = root / "vein" 
    od_dir = root / "optic_disc"
    crossings_dir = root / "crossings"

    (root / "bifurcations_arteries").mkdir(parents=True, exist_ok=True)
    (root / "bifurcations_veins").mkdir(parents=True, exist_ok=True)
    (root / "crossings_rois").mkdir(parents=True, exist_ok=True)
    (root / "major_vessels_arteries").mkdir(parents=True, exist_ok=True)
    (root / "major_vessels_veins").mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(image_dir.glob("*")))
    geoVBM = GeometricalVBMs()
    segmenter = DiscSegmenter()

    for img_path in image_files:
        
        name = img_path.stem
        artery_path = artery_dir / f"{name}.png"
        vein_path = vein_dir / f"{name}.png"
        od_path = od_dir / f"{name}.png"
        crossings_path = crossings_dir / f"{name}.png"

        data = {'artery': {}, 'vein': {}}

        if not artery_path.exists() or not od_path.exists():
            print(f"Skipping {name}: missing files")
            continue

        print(f"Processing {name}")

        image, artery, vein, optic_disc, crossings = load_data(img_path, artery_path, vein_path, od_path, crossings_path)
        
        data['artery']['complete'], data['vein']['complete'] = artery,  vein
        data['artery']['skeleton'], data['vein']['skeleton'] = skeletonize(data['artery']['complete']).astype(np.uint8), skeletonize(data['vein']['complete']).astype(np.uint8)

        #Compute center of optic disc for geoVBM analysis
        center, radius, _, _ = segmenter.post_processing(segmentation=optic_disc, max_roi_size=400, image_type=image_type)

        if center is None or radius is None:
            raise ValueError(f"Could not compute optic disc center/radius for {name}")

        for vessel_type in ["artery", "vein"]:

            vessel_complete = data[vessel_type]['complete']
            vessel_skeleton = data[vessel_type]['skeleton']

            #Junctions
            try:
                _, bifurcations = handle_interpoints(
                            vessel_complete=vessel_complete,
                            vessel_skeleton=vessel_skeleton,
                            crossings=crossings,
                            geoVBM=geoVBM,
                            center=center,
                            iterative_or_recursive="recursive",
                            radius=radius)        

            except:
                _, bifurcations = handle_interpoints(
                            vessel_complete=vessel_complete,
                            vessel_skeleton=vessel_skeleton,
                            crossings=crossings,
                            geoVBM=geoVBM,
                            center=center,
                            iterative_or_recursive="iterative",
                            radius=radius)                                  

            #Major vessels
            major_vessels = extract_major_vessels(vessel_mask=vessel_complete,
                                                od_mask=optic_disc,
                                                mode="opening",
                                                footprint_width=fw,
                                                n_cc=4)

            #Save
            cv2.imwrite(str(root / f"bifurcations_{vessel_type}" / f"{name}.png"), bifurcations * 255)
            cv2.imwrite(str(root / f"major_vessels_{vessel_type}" / f"{name}.png"), major_vessels * 255)

        #Crossings ROIs
        crossings_rois = extract_crossings_roi(crossings_mask=crossings,
                                                region_size=20)
        cv2.imwrite(str(root / f"crossings_rois" / f"{name}.png"), crossings_rois * 255)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--fw", type=int, default=None, help="Footprint width for major vessel extraction (define between 2-4 and observe results visually)")
    parser.add_argument("--image_type", type=str, default="ODC", choices=["ODC", "MC"], help="Type of image to determine ROI for Df computation (ODC: 0.5-2.0 OD diameters, MC: 5.0 OD diameters)")
    args = parser.parse_args()

    process_dataset(args.data_root, args.image_type, args.fw)