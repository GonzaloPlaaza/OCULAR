import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import monai.transforms as t
import pathlib
from skimage import io
from skimage.morphology import disk
from scipy import ndimage
from pathlib import Path
from PIL import Image
from scipy.interpolate import interp1d

from utils.RetinalSegDataset import RetinalSegDataset
from utils.PadFromCSVd import PadFromCSVd
from monai.transforms import MapTransform

####### RRWNet-SPECIFIC CODE (for image enhancement and padding) #######

def crop_center(img, cropx, cropy):
    y, x = img.shape[0], img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def to_0_1(img):
    interp_fun = interp1d([img.min(), img.max()], [0.0, 1.0])
    return interp_fun(img)

class EnhanceImageTransform(MapTransform):
    def __init__(self, keys=("image", "fov"), dataset_name: str = None):
        super().__init__(keys)
        self.dataset_name = dataset_name

    def __call__(self, data):
        d = dict(data)
        img_key, fov_key = self.keys
        img = d[img_key]  # (3, H, W)
        fov = d[fov_key]    # (1 or 3, H, W)

        # Convert to numpy
        img = img.cpu().numpy()
        fov = fov.cpu().numpy()

        if self.dataset_name != "RAVIR":

            # Move channels LAST
            img = np.transpose(img, (1, 2, 0))   # (H, W, 3)
            fov = np.transpose(fov, (1, 2, 0))   # (H, W, C)

            enhanced_img, enhanced_fov = enhance_image(img, fov)

            # Move back to channel-first
            enhanced_img = np.transpose(enhanced_img, (2, 0, 1))  # (3, H, W)

            d[img_key] = torch.as_tensor(enhanced_img, dtype=torch.float32)
            d[fov_key]   = torch.as_tensor(enhanced_fov, dtype=torch.float32).unsqueeze(0)

        else:
            if img.ndim == 2:  # (H,W)
                img = np.stack([img]*3, axis=0)  # (3,H,W)
            elif img.shape[0] == 1:  # (1,H,W)
                img = np.concatenate([img]*3, axis=0)  # (3,H,W)

            d[img_key] = torch.as_tensor(img, dtype=torch.float32)

            # FOV mask stays as 1 channel
            d[fov_key] = torch.as_tensor(fov, dtype=torch.float32)

        return d

def enhance_image(img, mask, int_format=False, disk_size=5):
    """Enhance an image using the method described in the paper.
    Args:
        img (np.ndarray): Image to enhance
        mask (np.ndarray): ROI mask of the image to enhance

    Returns:
        Enhanced image
    """
    # Read image and its corresponding mask
    if isinstance(img, str) or isinstance(img, Path):
        img = io.imread(img)[..., :3]
    if isinstance(mask, str) or isinstance(mask, Path):
        mask = io.imread(mask)

    if len(img.shape) == 3:
        if img.shape[2] > 3:
            img = img[:, :, :3]
    if len(mask.shape) == 3:
        mask = np.sum(mask[:, :, :3], axis=2)

    img = img / 255
    # mask = np.where(mask > (255//2), 255, 0)
    mask = np.where(mask > 0.5, 1, 0)

    # Copy original image
    img_copy = img.copy()
    # Convert to PIL format
    zoomed_image = Image.fromarray(np.uint8(img_copy*255))
    # Enlarge image
    zoomed_image = zoomed_image.resize(
        (int(img_copy.shape[1]*1.15), int(img_copy.shape[0]*1.15)),
        Image.BICUBIC
    )
    # To numpy array type
    zoomed_image = np.array(zoomed_image)
    # Crop image to original size (zoom result)
    zoomed_image = crop_center(zoomed_image, img_copy.shape[1],
                               img_copy.shape[0])
    # Convert image from 0-255 format to 0.0-1.0 format
    zoomed_image = zoomed_image / 255.0

    # Create circular kernel for mask erosion
    kernel = disk(disk_size)

    # Erode mask
    mask = ndimage.binary_erosion(mask, kernel)
    # Convert boolean array to float array
    mask = mask * 1.0  # type: ignore

    img_copy[mask < 1.0] = 0.0

    # Create RGB mask (same mask for all channels)
    mask = np.stack((mask, mask, mask), axis=2)

    composed_image = mask.copy()
    composed_image[mask == 1.0] = img_copy[mask == 1.0]
    composed_image[mask < 1.0] = zoomed_image[mask < 1.0]

    filtered_image = ndimage.gaussian_filter(composed_image, sigma=(10, 10, 0))

    subtracted_image = composed_image - filtered_image
    subtracted_image[mask < 1.] = 0.

    enhanced_image = subtracted_image/np.std(subtracted_image + 1e-8)
    enhanced_image = to_0_1(enhanced_image)
    enhanced_image[mask < 1.] = 0.

    mask = mask[:, :, 0]

    if int_format:
        enhanced_image *= 255
        enhanced_image = enhanced_image.astype(np.uint8)
        mask *= 255
        mask = mask.astype(np.uint8)

    return enhanced_image, mask

####### END OF RRWNet-SPECIFIC CODE #######

class DebugShapesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        print({k: tuple(v.shape) for k, v in d.items() if isinstance(v, torch.Tensor)})
        return d

# index we will use to mark pixels that must be ignored by the loss
IGNORE_INDEX = 255 

def get_norm_transform(keys, norm_mode):
    """
    Build a normalization transform for the specified image keys.

    Supported modes:
        - "none"
        - "per_image"
        - "imagenet"
    """

    if norm_mode == "none":
        return t.Identityd(keys=keys)

    elif norm_mode == "per_image":
        return t.NormalizeIntensityd(
            keys=keys,
            nonzero=False,
            channel_wise=True,
        )

    elif norm_mode == "imagenet":
        return t.NormalizeIntensityd(
            keys=keys,
            subtrahend=torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
            divisor=torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1),
        )
    
    else:
        raise ValueError(f"Unknown normalization mode '{norm_mode}'")

def get_tr_vl_image_transforms(
    tg_size: tuple = (1024, 1024),
    cfp_norm_mode: str = "imagenet",
    ffa_norm_mode: str = "per_image",
    resize_mode: str = "resize",
    debug: bool = False,
    dataset_name: str = None,
    no_labels: bool = False,
    ffa: bool = False,
):
    """
    Create MONAI training and validation transforms for retinal vessel segmentation.

    Supports:
        - RGB CFP images.
        - Optional FFA modalities (ffa_a, ffa_av).
        - Training or inference pipelines.
        - Multiple resizing strategies.

    Parameters:
        tg_size (tuple):
            Target image size as (height, width).

        norm_mode (str):
            CFP normalization strategy.
            Supported values:
                - "imagenet"
                - "per_image"

        resize_mode (str):
            Image resizing strategy.
            Supported values:
                - "resize"
                - "pad"
                - "rrwnet"
                - "keep_resolution"

        debug (bool):
            Whether to insert DebugShapesd into the pipeline.

        dataset_name (str):
            Dataset name used by RRWNet enhancement.

        no_labels (bool):
            Whether labels are available.

        ffa (bool):
            Whether FFA modalities are included as additional inputs.

    Returns:
        tuple:
            (training_transform, validation_transform)
    """

    image_keys = ["image"]
    if ffa:
        image_keys += ["ffa_a", "ffa_av"]

    label_keys = []
    if not no_labels:
        label_keys = [
            "arteries",
            "major_arteries",
            "veins",
            "major_veins",
            "uncertain_vessels",
            "vessels",
            "bifurcations_arteries",
            "bifurcations_veins",
            "crossings_roi",
            "crossings",
            "od",
        ]

    keys = image_keys + label_keys

    #Build normalization transforms
    cfp_norm_tx = get_norm_transform(keys=["image"], norm_mode=cfp_norm_mode)
    ffa_a_norm_tx = get_norm_transform(keys=["ffa_a"] if ffa else ['image'], 
                                     norm_mode=ffa_norm_mode if ffa else "none")
    ffa_av_norm_tx = get_norm_transform(keys=["ffa_av"] if ffa else ['image'],
                                      norm_mode=ffa_norm_mode if ffa else "none")

    #Select resize transform
    if resize_mode == "resize":
        resize_transform = t.Resized(
            keys=keys,
            spatial_size=tg_size,
            mode=["area"] * len(image_keys) + ["nearest"] * len(label_keys),
        )

    elif resize_mode == "pad":
        resize_transform = PadFromCSVd(
            keys=keys,
            target_size=tg_size[0],
        )

    elif resize_mode == "None":
        resize_transform = t.Identityd(keys=keys)

    elif resize_mode in ("rrwnet", "keep_resolution"):
        resize_transform = t.Identityd(keys=keys)

    else:
        raise ValueError(
            "resize_mode must be one of "
            "'resize', 'pad', 'None', 'rrwnet', 'keep_resolution'"
        )

    #Common preprocessing
    common = [
        t.LoadImaged(keys=keys),
        t.EnsureChannelFirstd(keys=keys),
        t.EnsureTyped(keys=keys, dtype=torch.uint8),

        t.ScaleIntensityRanged(
            keys=image_keys,
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        t.Transposed(
            keys=keys,
            indices=(0, 2, 1),),

        resize_transform,
        DebugShapesd(keys=keys) if debug else t.Identityd(keys=keys),
    ]

    #Training augmentations
    prob = 0.5
    tr_pix = (int(0.05 * tg_size[1]), int(0.05 * tg_size[0]))

    train_aug = []
    if not no_labels:

        train_aug += [
            t.RandAffined(
                keys=keys,
                prob=prob,
                rotate_range=(np.pi / 6, np.pi / 6),
                scale_range=(0.2, 0.05),
                translate_range=tr_pix,
                mode=["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys),
                padding_mode="zeros",
            ),

            t.RandLambdad(
                keys=("image",),
                prob=prob,
                func=lambda x: t.TorchVision(
                    name="ColorJitter",
                    brightness=(0.75, 1.5),
                    contrast=(0.75, 1.5),
                    saturation=(0.75, 1.5),
                    hue=0.025,
                )(x),
            ),

            t.RandAdjustContrastd(
                keys=("image",),
                prob=prob,
                gamma=(0.75, 1.5),
            ),

            t.RandBiasFieldd(
                keys=("image",),
                prob=prob / 5,
                coeff_range=(-0.25, 0.25),
            ),

            t.RandGaussianSmoothd(
                keys=("image",),
                prob=prob / 5,
                sigma_x=(0.01, 0.5),
                sigma_y=(0.01, 0.5),
            ),

            t.RandGaussianNoised(
                keys=("image",),
                prob=prob / 5,
                mean=0.0,
                std=0.01,
            ),
        ]

    #Common postprocessing
    post = []
    if label_keys:
        post.append(
            t.CastToTyped(
                keys=label_keys,
                dtype=torch.bool,
            )
        )

    post += [
        t.Lambdad(
            keys=image_keys,
            func=lambda x: x.clamp(0.0, 1.0),
        ),
        cfp_norm_tx,
        ffa_a_norm_tx,
        ffa_av_norm_tx,
    ]

    #Compose train/validation transforms
    if no_labels:
        tr_tx = None
        vl_tx = t.Compose(common + post)

    else:
        tr_tx = t.Compose(
            common +
            train_aug +
            post)

        vl_tx = t.Compose(
            common +
            post)

    #RRWNet override.
    if resize_mode == "rrwnet":

        vl_tx = t.Compose([
            t.LoadImaged(keys=keys),
            t.EnsureChannelFirstd(keys=keys),
            t.EnsureTyped(keys=["image"], dtype=torch.float32),

            EnhanceImageTransform(
                keys=("image", "fov"),
                dataset_name=dataset_name,
            ),

            t.Transposed(keys=keys, indices=(0, 2, 1)),

            t.SpatialPadd(
                keys=["image"],
                spatial_size=(1024, 1024),
                mode="constant",
                constant_values=0.0,
            ),

            t.SpatialPadd(
                keys=[k for k in keys if k != "image"],
                spatial_size=(1024, 1024),
                mode="constant",
                constant_values=0,
            ),

            t.CastToTyped(
                keys=label_keys,
                dtype=torch.bool,
            ),

            DebugShapesd(keys=keys) if debug else t.Identityd(keys=keys),
        ])

    return tr_tx, vl_tx


def get_retinal_seg_train_val_loaders(csv_path_tr: str, 
                                      batch_size: int, 
                                      tg_size: tuple,
                                      num_workers:int=0, 
                                      resize_mode:str="pad",
                                      problem_type: str = "multi_class",
                                      crossings_is_class: bool = False,
                                      cfp_norm_mode: str = "imagenet",
                                      ffa_norm_mode: str = "per_image",
                                      ignore_od: bool = True,
                                      dataset_name: str = None,
                                      no_labels: bool = False,
                                      ffa: bool = False):

    assert resize_mode in ["pad", "resize", "keep_resolution"], "resize_mode must be either 'pad', 'resize' or 'keep_resolution'"
    
    #Define transforms
    tr_transforms, vl_transforms = get_tr_vl_image_transforms(tg_size, 
                                                              cfp_norm_mode=cfp_norm_mode,
                                                              ffa_norm_mode=ffa_norm_mode,
                                                              resize_mode=resize_mode,
                                                              dataset_name=dataset_name,
                                                              ffa=ffa,
                                                              no_labels=no_labels)
    
    #Define datasets and dataloaders
    tr_ds = RetinalSegDataset(csv_path_tr, transforms=tr_transforms, problem_type=problem_type, ignore_od=ignore_od, crossing_is_class=crossings_is_class, dataset_name=dataset_name, no_labels=no_labels, ffa=ffa)
    csv_path_vl = pathlib.Path(str(csv_path_tr).replace('/tr', '/vl'))
    if csv_path_vl.is_file():
        vl_ds = RetinalSegDataset(csv_path_vl, transforms=vl_transforms, problem_type=problem_type, ignore_od=ignore_od, crossing_is_class=crossings_is_class, dataset_name=dataset_name, no_labels=no_labels, ffa=ffa)
        ovft_ds = RetinalSegDataset(csv_path_tr, transforms=vl_transforms, problem_type=problem_type, ignore_od=ignore_od, crossing_is_class=crossings_is_class, dataset_name=dataset_name, no_labels=no_labels, ffa=ffa)

    else:
        vl_ds = RetinalSegDataset(csv_path_tr, transforms=vl_transforms, problem_type=problem_type, ignore_od=ignore_od, crossing_is_class=crossings_is_class, dataset_name=dataset_name, no_labels=no_labels, ffa=ffa)
        ovft_ds = RetinalSegDataset(csv_path_tr, transforms=vl_transforms, problem_type=problem_type, ignore_od=ignore_od, crossing_is_class=crossings_is_class, dataset_name=dataset_name, no_labels=no_labels, ffa=ffa)

    subset_size = len(vl_ds)
    subset_idxs = torch.randperm(len(ovft_ds))[:subset_size]
    ovft_ds = Subset(ovft_ds, subset_idxs)

    tr_loader = DataLoader(dataset=tr_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    vl_loader = DataLoader(dataset=vl_ds, batch_size=batch_size, num_workers=num_workers)
    ovft_loader= DataLoader(dataset=ovft_ds, batch_size=batch_size, num_workers=num_workers)

    return tr_loader, ovft_loader, vl_loader

def get_retinal_seg_test_loader(csv_path_test: str, 
                                batch_size: int, 
                                tg_size: tuple,
                                num_workers:int=0, 
                                resize_mode:str="pad",
                                problem_type: str = "multi_class",
                                crossings_is_class: bool = False,  
                                ignore_od: bool = True,
                                dataset_name: str = None,   
                                no_labels: bool = False,
                                cfp_norm_mode: str = "imagenet",
                                ffa_norm_mode: str = "per_image",
                                ffa: bool = False):

    assert resize_mode in ["pad", "resize", "rrwnet", "None"], "resize_mode must be either 'pad', 'resize' or 'rrwnet'"
    
    #Define transforms
    _, test_transforms = get_tr_vl_image_transforms(tg_size, 
                                                    resize_mode=resize_mode, 
                                                    dataset_name=dataset_name, 
                                                    no_labels=no_labels,
                                                    cfp_norm_mode=cfp_norm_mode,
                                                    ffa_norm_mode=ffa_norm_mode,
                                                    ffa=ffa)
    
    #Define dataset and dataloader
    test_ds = RetinalSegDataset(csv_path_test, 
                                transforms=test_transforms, 
                                problem_type=problem_type, 
                                ignore_od=ignore_od, 
                                crossing_is_class=crossings_is_class, 
                                dataset_name=dataset_name,
                                no_labels=no_labels,
                                ffa=ffa)
    
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, num_workers=num_workers)

    return test_loader