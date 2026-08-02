import numpy as np
from skimage.morphology import dilation, disk
from skimage.measure import label, regionprops
from skimage import morphology
from scipy.ndimage import binary_dilation
import cv2

def compute_od(optic_disc: np.ndarray):
    """
    Compute the center and radius of the optic disc using image moments.
    Args:
        optic_disc: Binary mask of the optic disc (H, W)
    Returns:
        center: (x, y) coordinates of the optic disc center
        radius: Radius of the optic disc
    """

    M = cv2.moments(optic_disc)
    if M["m00"] > 0:
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    else:
        center = (optic_disc.shape[1] // 2, optic_disc.shape[0] // 2)  # Fallback to image center

    radius = int(np.sqrt(M["m00"] / np.pi))  # Approximate radius from area

    return center, radius

def handle_interpoints(
    vessel_complete: np.ndarray,
    vessel_skeleton: np.ndarray,
    crossings: np.ndarray,
    geoVBM,
    center: tuple,
    radius: int,
    iterative_or_recursive: str = "recursive",
    return_mask: bool = True):
    """
    Extract bifurcation points (interpoints) and corresponding ROI mask.

    Returns:
        coords: (N, 2) array of bifurcation coordinates
        bifurcation_mask: binary mask (same size as input)
    """

    #Compute interpoints
    interpoints = geoVBM.compute_geomVBMs(
                blood_vessel=vessel_complete,
                skeleton=vessel_skeleton,
                xc=center[0],
                yc=center[1],
                radius=radius,
                iterative_or_recursive=iterative_or_recursive,
                interpoints_only=True,
            )

    interpoints = interpoints * (1 - crossings)

    #Coordinates
    coords = np.column_stack(np.nonzero(interpoints))  # (N, 2)

    if not return_mask:
        return coords, None

    #Build ROI mask
    bifurcation_mask = dilation(interpoints, disk(20))

    return coords, bifurcation_mask

def extract_crossings_roi(crossings_mask: np.ndarray,
                          region_size: int = 20) -> np.ndarray:
    """
    Generate circular ROIs for all crossings in the image.
    
    Args:
        artery_mask (np.ndarray): Binary artery segmentation.
        vein_mask (np.ndarray): Binary vein segmentation.
        crossings_mask (np.ndarray): Binary crossings segmentation.
        region_size (int): Radius of circular ROI around each crossing.
        subject_dir (Path): Optional folder to save crossings ROI as PNG.
        image_suffix (str): Optional suffix for saving file.
    
    Returns:
        np.ndarray: Binary image with circular ROIs around crossings.
    """
    # Label connected components in the crossings mask
    labeled_crossings = label(crossings_mask)
    crossing_rois = np.zeros_like(crossings_mask, dtype=np.uint8)

    for region in regionprops(labeled_crossings):
        # Skip very small regions (noise)
        if region.area < 3:
            continue
        
        # Centroid coordinates
        cy, cx = map(int, region.centroid)
        crossing_rois[cy, cx] = 1

    # Dilate centroid points to circular masks
    selem = disk(region_size)
    crossing_rois = binary_dilation(crossing_rois, selem).astype(np.uint8)

    return crossing_rois


def extract_major_vessels(vessel_mask: np.ndarray,
                          od_mask: np.ndarray,
                          mode: str = "opening",
                          footprint_width: int = None,
                          n_cc: int = 4) -> np.ndarray:
    """
    Extract major vessels from a vessel mask using the optic disc as reference.
    
    Args:
        vessel_mask (np.ndarray): Binary vessel mask (H, W)
        od_mask (np.ndarray): Binary optic disc mask (H, W)
        mode (str): Method to extract major vessels:
            - 'opening': morphological opening using a footprint from OD size
            - 'largest_cc': keep the largest n_cc connected components
        footprint_width (int, optional): Width of morphological footprint (only for 'opening' mode)
        n_cc (int): Number of largest connected components to keep (only for 'largest_cc' mode)
    
    Returns:
        np.ndarray: Binary mask of major vessels (H, W)
    """
    assert mode in ["opening", "largest_cc"], "Mode must be 'opening' or 'largest_cc'"

    vessel_mask = vessel_mask.astype(bool)
    od_mask = od_mask.astype(bool)

    if mode == "opening":
        labels = label(od_mask)
        props = regionprops(labels)
        if len(props) == 0:
            return np.zeros_like(vessel_mask, dtype=bool)

        major_axis = props[0].major_axis_length
        if footprint_width is None:
            footprint_width = int(np.ceil(major_axis / 100))
        footprint = morphology.rectangle(footprint_width, footprint_width)
        zone = morphology.opening(vessel_mask, footprint)
        zone = morphology.remove_small_objects(zone.astype(bool), min_size=1000)

    elif mode == "largest_cc":
        labels = label(vessel_mask)
        props = regionprops(labels)
        if len(props) == 0:
            return np.zeros_like(vessel_mask, dtype=bool)

        # Sort connected components by area descending
        props = sorted(props, key=lambda x: x.area, reverse=True)
        zone = np.zeros_like(vessel_mask, dtype=bool)
        for r in props[:n_cc]:
            zone[labels == r.label] = True

    return zone.astype(np.uint8)