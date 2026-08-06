import numpy as np
import torch
from tqdm import trange

from utils.metric_factory import multiclass_dice, masked_multiclass_dice, masked_multiclass_cldice, bin_recall
from utils.training import update_running_losses_and_postfix_weighted


def compute_batch_losses_multiclass(losses_dict: dict, logits: torch.Tensor, labels: torch.Tensor, zones: dict) -> dict:
    batch_losses = {}
    for key in losses_dict:
        if key == "av":
            loss = losses_dict["av"](logits, labels)
        elif key == "arcades":
            arcades_zone = zones["major_arteries"] | zones["major_veins"]
            loss = losses_dict["arcades"](logits, labels, arcades_zone)
        elif key == "junctions":
            junction_zone = zones["bifurcations_arteries"] | zones["bifurcations_veins"] | zones["crossings_roi"]
            loss = losses_dict["junctions"](logits, labels, junction_zone)
        else:
            continue
        batch_losses[key] = loss.item()
    return batch_losses


def compute_batch_losses_multilabel(
    losses_dict: dict, logits: torch.Tensor, labels: torch.Tensor, zones: dict, ignore_mask: torch.Tensor = None
) -> dict:
    batch_losses = {}
    for key, loss_fn in losses_dict.items():
        if key == "av":
            if ignore_mask is not None:
                loss = loss_fn(logits, labels.float(), ignore_mask)
            else:
                loss = loss_fn(logits, labels.float())
        elif key == "arcades":
            arcade_mask = zones["major_arteries"] | zones["major_veins"]
            valid = ignore_mask & arcade_mask.unsqueeze(1) if ignore_mask is not None else arcade_mask.unsqueeze(1)
            loss = loss_fn(logits, labels.float(), valid)
        elif key == "junctions":
            junction_mask = zones["bifurcations_arteries"] | zones["bifurcations_veins"] | zones["crossings_roi"]
            valid = ignore_mask & junction_mask.unsqueeze(1) if ignore_mask is not None else junction_mask.unsqueeze(1)
            loss = loss_fn(logits, labels.float(), valid)
        else:
            continue
        batch_losses[key] = loss.item()
    return batch_losses


def validate_multiclass(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    losses_dict: dict,
    lambda_dict: dict,
    CLASS_DICT: dict,
    device: torch.device = None,
    half_precision: bool = False,
):
    model.eval()
    dscs, junction_dscs, arcades_cldscs = [], [], []
    junction_recalls = {"artery_bif": [], "vein_bif": [], "crossings": []}
    running_losses = {"count": 0}

    with torch.no_grad(), trange(len(loader)) as t:
        for batch_data in loader:
            inputs, labels, zones, _, _, _ = batch_data
            inputs, labels = inputs.to(device), labels.to(device)
            if losses_dict.get("arcades") or losses_dict.get("junctions"):
                zones = {k: v.to(device) for k, v in zones.items()}

            if not half_precision:
                logits = model(inputs)
                batch_losses = compute_batch_losses_multiclass(losses_dict, logits, labels, zones)
            else:
                with torch.cuda.amp.autocast(enabled=True):
                    logits = model(inputs)
                    batch_losses = compute_batch_losses_multiclass(losses_dict, logits, labels, zones)

            batch_size = inputs.shape[0]
            running_losses["count"] += batch_size
            update_running_losses_and_postfix_weighted(
                t=t,
                running_losses=running_losses,
                batch_losses=batch_losses,
                lambdas=lambda_dict,
                batch_size=batch_size,
                optimizer=None,
            )

            y_true = labels.cpu().numpy().astype(np.uint8)
            y_pred = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            arcades_np = (zones["major_arteries"] | zones["major_veins"]).cpu().numpy().astype(bool)

            for gt, seg, bif_a, bif_v, cross, arc in zip(
                y_true,
                y_pred,
                zones["bifurcations_arteries"].cpu().numpy().astype(bool),
                zones["bifurcations_veins"].cpu().numpy().astype(bool),
                zones["crossings_roi"].cpu().numpy().astype(bool),
                arcades_np,
            ):
                dscs.append(multiclass_dice(actual=gt, predicted=seg, n_classes=len(CLASS_DICT), exclude_background=True))

                junction_mask = bif_a | bif_v | cross
                junction_dscs.append(
                    masked_multiclass_dice(
                        actual=gt,
                        predicted=seg,
                        mask=junction_mask,
                        n_classes=len(CLASS_DICT),
                        exclude_background=True,
                    )
                )

                junction_recalls["artery_bif"].append(bin_recall((gt == 1) & bif_a, (seg == 1) & bif_a))
                junction_recalls["vein_bif"].append(bin_recall((gt == 2) & bif_v, (seg == 2) & bif_v))
                junction_recalls["crossings"].append(bin_recall((gt == 3) & cross, (seg == 3) & cross))

                arcades_cldscs.append(
                    masked_multiclass_cldice(
                        actual=gt, predicted=seg, mask=arc, n_classes=len(CLASS_DICT), ignore_index=255
                    )
                )

    dsc_per_class = np.nanmean(np.asarray(dscs, dtype=np.float32), axis=0)
    junction_dsc_per_class = np.nanmean(np.asarray(junction_dscs, dtype=np.float32), axis=0)
    arcades_cldsc_per_class = np.nanmean(np.asarray(arcades_cldscs, dtype=np.float32), axis=0)
    junction_recalls_avg = {k: np.nanmean(v) for k, v in junction_recalls.items()}

    metrics = {
        "global": dsc_per_class,
        "junctions": junction_dsc_per_class,
        "junction_recalls": junction_recalls_avg,
        "arcades_cldice": arcades_cldsc_per_class,
    }

    avg_losses = {k: running_losses[k] / running_losses["count"] for k in running_losses if k != "count"}
    avg_losses["total"] = sum(lambda_dict.get(k, 0.0) * avg_losses[k] for k in avg_losses if k in lambda_dict)
    return metrics, avg_losses


def validate_multilabel(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    losses_dict: dict,
    lambda_dict: dict,
    CLASS_DICT: dict,
    device: torch.device = None,
    threshold: float = 0.5,
):
    model.eval()
    dscs, junction_dscs, arcades_cldscs = [], [], []
    junction_recalls = {"artery_bif": [], "vein_bif": [], "crossings": []}
    running_losses = {"count": 0}

    with torch.no_grad(), trange(len(loader)) as t:
        for batch_data in loader:
            inputs, labels, zones, _, ignore_mask, _ = batch_data
            inputs, labels = inputs.to(device), labels.to(device)
            ignore_mask = ignore_mask.to(device) if ignore_mask is not None else torch.ones_like(labels[:, 0, :, :])
            if losses_dict.get("arcades") or losses_dict.get("junctions"):
                zones = {k: v.to(device) for k, v in zones.items()}

            logits = model(inputs)
            labels_art_vein = labels[:, :2, :, :]

            batch_losses = compute_batch_losses_multilabel(
                losses_dict=losses_dict,
                logits=logits,
                labels=labels_art_vein,
                zones=zones,
                ignore_mask=ignore_mask,
            )

            update_running_losses_and_postfix_weighted(
                t=t,
                running_losses=running_losses,
                batch_losses=batch_losses,
                lambdas=lambda_dict,
                batch_size=inputs.shape[0],
                optimizer=None,
                problem_type="multi_label",
            )

            probs = torch.sigmoid(logits)
            artery_pred = probs[:, 0, :, :] > threshold
            vein_pred = probs[:, 1, :, :] > threshold

            pred_labels = torch.zeros_like(labels[:, 0, :, :], dtype=torch.uint8)
            pred_labels[artery_pred & ~vein_pred] = 1
            pred_labels[~artery_pred & vein_pred] = 2
            pred_labels[artery_pred & vein_pred] = 3

            if ignore_mask.ndim == 4:
                ignore_mask = ignore_mask[:, 0]
            pred_labels[ignore_mask == 0] = 255

            gt_labels = torch.zeros_like(labels[:, 0, :, :], dtype=torch.uint8)
            art_gt = labels[:, 0, :, :] == 1
            vein_gt = labels[:, 1, :, :] == 1
            gt_labels[art_gt & ~vein_gt] = 1
            gt_labels[~art_gt & vein_gt] = 2
            gt_labels[art_gt & vein_gt] = 3
            gt_labels[ignore_mask == 0] = 255

            y_true = gt_labels.cpu().numpy()
            y_pred = pred_labels.cpu().numpy()
            for gt, seg, ign, bif_a, bif_v, cross, arc in zip(
                y_true,
                y_pred,
                ignore_mask.cpu().numpy().astype(bool),
                zones["bifurcations_arteries"].cpu().numpy().astype(bool),
                zones["bifurcations_veins"].cpu().numpy().astype(bool),
                zones["crossings_roi"].cpu().numpy().astype(bool),
                (zones["major_arteries"] | zones["major_veins"]).cpu().numpy().astype(bool),
            ):
                dscs.append(
                    masked_multiclass_dice(
                        actual=gt,
                        predicted=seg,
                        mask=ign,
                        n_classes=len(CLASS_DICT),
                        exclude_background=True,
                        ignore_index=255,
                    )
                )

                junction_mask = bif_a | bif_v | cross
                junction_dscs.append(
                    masked_multiclass_dice(
                        actual=gt,
                        predicted=seg,
                        mask=ign & junction_mask,
                        n_classes=len(CLASS_DICT),
                        exclude_background=True,
                        ignore_index=255,
                    )
                )

                junction_recalls["artery_bif"].append(bin_recall((gt == 1) & bif_a & ign, (seg == 1) & bif_a & ign))
                junction_recalls["vein_bif"].append(bin_recall((gt == 2) & bif_v & ign, (seg == 2) & bif_v & ign))
                junction_recalls["crossings"].append(bin_recall((gt == 3) & cross & ign, (seg == 3) & cross & ign))

                arcades_cldscs.append(
                    masked_multiclass_cldice(
                        actual=gt, predicted=seg, mask=ign & arc, n_classes=len(CLASS_DICT), ignore_index=255
                    )
                )

    dsc_per_class = np.nanmean(np.asarray(dscs, dtype=np.float32), axis=0)
    junction_dsc_per_class = np.nanmean(np.asarray(junction_dscs, dtype=np.float32), axis=0)
    arcades_cldsc_per_class = np.nanmean(np.asarray(arcades_cldscs, dtype=np.float32), axis=0)
    junction_recalls_avg = {k: np.nanmean(v) for k, v in junction_recalls.items()}

    metrics = {
        "global": dsc_per_class,
        "junctions": junction_dsc_per_class,
        "junction_recalls": junction_recalls_avg,
        "arcades_cldice": arcades_cldsc_per_class,
    }

    avg_losses = {k: running_losses[k] / running_losses["count"] for k in running_losses if k != "count"}
    avg_losses["total"] = sum(lambda_dict.get(k, 0.0) * avg_losses[k] for k in avg_losses if k in lambda_dict)
    return metrics, avg_losses
