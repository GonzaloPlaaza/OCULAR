import os
import pathlib
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from utils.get_model import get_model

#python inference.py --input_dir data/images/ --output_dir data/segmentations/ --weights pretrained_weights/OCULARNet.pth

def get_args():
    
    parser = argparse.ArgumentParser("OCULARNet Inference")
    parser.add_argument("--input_dir", type=str, default="data/images/", help="Folder with input images",)
    parser.add_argument("--output_dir", type=str, default="data/segmentations/",help="Folder to save predictions")
    parser.add_argument("--weights", type=str, default="pretrained_weights/OCULARNet/OCULARNet.pth", 
                        help="Path to pretrained weights. If OCULARNet-nano ensemble, specify the path to the first fold weights, e.g." \
                        "pretrained_weights/OCULARNet-nano/nano_f1.pth",)
    parser.add_argument("--model_name", type=str, default="base_unet_repvgg_b3", help="Model architecture to use")
    parser.add_argument("--im_size", type=int, nargs=2,default=[1024, 1024])
    parser.add_argument("--device", type=str, default="cuda",choices=["cuda", "cpu", "mps"])
    parser.add_argument("--ensemble", action="store_true", default=False, help="Whether to use model ensembling (5-fold default)")
    parser.add_argument("--resize", action="store_true", default=False, help="Whether to resize input images to 1024x1024 (recommended for best performance)")
    return parser.parse_args()


def load_images_from_folder(folder):
    image_paths = sorted([
        p for p in pathlib.Path(folder).glob("*")
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif"]
    ])
    return image_paths

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def norm_fn(x: torch.Tensor):
    """
    Normalize using ImageNet mean/std (assuming input in [0,1])
    Args:
        x: Tensor of shape (C, H, W) with values in [0,1]
    Returns:
        Normalized tensor of shape (C, H, W)
    """

    return (x - IMAGENET_MEAN) / IMAGENET_STD

def preprocess_image(img_path: pathlib.Path, im_size: tuple, norm_fn: callable=None, resize: bool=True):
    """
    Function to preprocess input image.
    Args:
        img_path: Path to input image
        im_size: Tuple (H, W) to resize image to
        norm_fn: Optional normalization function to apply after scaling to [0,1]
        resize: bool; whether to resize input image to im_size. 

    Returns:
        Preprocessed image tensor (C, H, W)
    
    """
    img = Image.open(img_path).convert("RGB")
    #Resize (longest side already 1024 assumed, but we enforce)
    if resize:
        img = img.resize(im_size)
    img = np.array(img).astype(np.float32)

    #Scale to [0,1]
    img = np.clip(img, 0, 255) / 255.0

    #HWC → CHW
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float() #this is float32 by default

    #Final normalization
    if norm_fn is not None:
        img = norm_fn(img)

    return img

def inference_multiclass(model: list, 
                         inputs: torch.Tensor,
                         ensemble: bool,
                         device: torch.device,
                         num_classes: int=4,) -> tuple:
    """
    Args:
        model: List of models (length 1 if no ensembling, length 5 if ensembling)
        inputs: Input tensor of shape [B,C,H,W]
        ensemble: Whether to use model ensembling
        device: Device to run inference on
        num_classes: Number of output classes (default 4 for background, artery, vein, crossings)

    Returns:
        preds: Predicted class labels of shape [B,H,W]
        probs: Class probabilities of shape [B,C,H,W]
    """
    with torch.inference_mode():
        
        if ensemble:
            #TTA + model ensembling
            #model is a list of 5 models
            logits = torch.zeros((inputs.size(0), num_classes, inputs.size(2), inputs.size(3)), device=device)
            for model_fold in model:
                logits += (
                    model_fold(inputs) +
                    model_fold(inputs.flip(dims=[-1])).flip(dims=[-1]) +
                    model_fold(inputs.flip(dims=[-2])).flip(dims=[-2]) +
                    model_fold(inputs.flip(dims=[-1, -2])).flip(dims=[-1, -2])
                )
            logits /= (4.0 * len(model))  # average across TTA and folds
        
        
        else:
            #TTA only with single model
            logits = (
                model[0](inputs) +
                model[0](inputs.flip(-1)).flip(-1) +        # horizontal
                model[0](inputs.flip(-2)).flip(-2) +        # vertical
                model[0](inputs.flip(-1, -2)).flip(-1, -2)  # both
            ) / 4.0

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    return preds.cpu().numpy(), probs.cpu().numpy()


def postprocess_prediction(pred):
    """
    Convert class map → RGB mask
    """
    pred = pred.astype(np.uint8)

    pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    #artery → R
    pred_rgb[:, :, 0][pred == 1] = 255
    #vein → B
    pred_rgb[:, :, 2][pred == 2] = 255
    #crossings → G
    pred_rgb[:, :, 1][pred == 3] = 255

    return pred_rgb


def main():
    
    args = get_args()
    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available())
        else "mps" if (args.device == "mps" and torch.backends.mps.is_available())
        else "cpu"
    )

    print(f"* Using device: {device}")

    num_classes = 4  #background, artery, vein, crossings
    n_folds = 5 if args.ensemble else 1
    models = []

    #Check consistency in paths
    if args.ensemble and "_f1" not in args.weights:
        raise ValueError(
            "--ensemble expects a path ending in *_f1.pth"
        )

    for fold in range(1, n_folds + 1):
        
        if args.ensemble:
            print(f"* Processing fold {fold}/{n_folds}")
        
        model = get_model(args.model_name, num_classes=num_classes, in_c=3)
        load_path = args.weights if not args.ensemble else args.weights.replace("_f1", f"_f{fold}")
        checkpoint = torch.load(load_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        models.append(model)

        print(f"* Loaded weights from {load_path}")
 

    #Load images
    image_paths = load_images_from_folder(args.input_dir)

    #Make sure segmentation output folder exists
    weights_path = pathlib.Path(args.weights)   
    model_name = weights_path.parent.name #e.g., OCULARNet; OCULARNet-nano
    output_dir = pathlib.Path(args.output_dir) / model_name #e.g., data/segmentations/OCULARNet/
    os.makedirs(output_dir, exist_ok=True)

    print(f"* Found {len(image_paths)} images")

    #Inference loop
    with torch.no_grad():
        for img_path in tqdm(image_paths):

            #Preprocess
            img_tensor = preprocess_image(img_path, args.im_size, norm_fn=norm_fn, resize=args.resize)
            img_tensor = img_tensor.unsqueeze(0).to(device)

            #Inference
            pred, prob = inference_multiclass(models, img_tensor, args.ensemble, device, num_classes=num_classes)
            pred = pred[0]  #remove batch dimension

            #Postprocess
            pred_rgb = postprocess_prediction(pred)

            #Save
            save_path = pathlib.Path(output_dir) / img_path.name
            Image.fromarray(pred_rgb).save(save_path)

    print("Done.")


if __name__ == "__main__":
    main()