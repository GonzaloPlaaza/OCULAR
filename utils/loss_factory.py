import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import sys
import argparse

####### MULTI-CLASS / BINARY CLASS LOSSES ##########
class SoftSkeletonize(torch.nn.Module):
    def __init__(self, num_iter=5):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = int(num_iter)

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        if len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)
        raise ValueError('expected 4D or 5D tensor, got {}'.format(len(img.shape)))

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        if len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        raise ValueError('expected 4D or 5D tensor, got {}'.format(len(img.shape)))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, img):
        return self.soft_skel(img)



class CompoundLoss(torch.nn.Module):
    def __init__(self, loss1, loss2=None, alpha1=1., alpha2=0.):
        super(CompoundLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, y_pred, y_true, ignore_mask=None):
        l1 = self.loss1(y_pred, y_true, ignore_mask=ignore_mask)
        if self.alpha2 == 0 or self.loss2 is None:
            return self.alpha1*l1
        l2 = self.loss2(y_pred, y_true, ignore_mask=ignore_mask)
        return self.alpha1*l1 + self.alpha2 * l2

class CELoss(torch.nn.Module):
    def __init__(self, ce_ls: float = 0., ignore_index: int = 255):
        super(CELoss, self).__init__()
        # CrossEntropy will ignore pixels with value == ignore_index
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=ce_ls, ignore_index=ignore_index)

    def forward(self, y_pred, y_true, ignore_mask=None):
        # y_pred: [B, C, H, W], y_true: [B, 1, H, W] or [B, H, W]
        if y_true.dim() == 4 and y_true.size(1) == 1:
            y = y_true.squeeze(1).long()
        else:
            y = y_true.long()
        return self.loss(y_pred, y)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-6, ignore_index=255, include_background=False, batch_dice=True):
        super(DiceLoss, self).__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)
        self.ignore_index = int(ignore_index)
        self.include_background = bool(include_background)
        self.batch_dice = bool(batch_dice)

    def forward(self, y_pred, y_true, ignore_mask=None):
        # y_pred: [B, C, H, W] logits
        # y_true: [B, 1, H, W] or [B, H, W] integer labels, ignore_index allowed
        if y_true.dim() == 4 and y_true.size(1) == 1:
            y_true = y_true.squeeze(1)
        y_true = y_true.long()

        _, c, _, _ = y_pred.shape
        probs = F.softmax(y_pred, dim=1)

        valid = (y_true != self.ignore_index)  # [B, H, W]
        y_safe = y_true.clone()
        y_safe[~valid] = 0  # placeholder for one_hot

        y_onehot = F.one_hot(y_safe, num_classes=c).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        y_onehot = y_onehot * valid_f

        if self.batch_dice:
            # Dice per class over (B,H,W)
            dims = (0, 2, 3)
            inter = torch.sum(probs * y_onehot, dims)              # [C]
            denom = torch.sum(probs, dims) + torch.sum(y_onehot, dims)  # [C]
            dice = (2.0 * inter + self.smooth) / (denom + self.smooth + self.eps)  # [C]

            if (not self.include_background) and c > 1:
                dice = dice[1:]

            return (1.0 - dice).mean()

        # Per-image Dice: dice[b, c] then mean over valid classes/images
        dims = (2, 3)
        inter = torch.sum(probs * y_onehot, dims)                   # [B, C]
        denom = torch.sum(probs, dims) + torch.sum(y_onehot, dims)  # [B, C]
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth + self.eps)  # [B, C]

        if (not self.include_background) and c > 1:
            dice = dice[:, 1:]        # [B, C-1]
            y_onehot_fg = y_onehot[:, 1:]
        else:
            y_onehot_fg = y_onehot

        # Exclude classes absent in GT for each image (in the averaged loss)
        # present[b, k] = GT has at least one pixel of class k in image b
        present = (torch.sum(y_onehot_fg, dims) > 0.0).float()  # [B, C'] where C' is C or C-1

        loss = 1.0 - dice
        loss = loss * present
        denom_present = present.sum().clamp_min(1.0)
        return loss.sum() / denom_present


class CLDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-6, ignore_index=255, skel_num_iter=5, batch_reduction=True):
        super(CLDiceLoss, self).__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)
        self.ignore_index = int(ignore_index)
        self.skel = SoftSkeletonize(num_iter=int(skel_num_iter))
        self.batch_reduction = bool(batch_reduction)  # True = reduce over (B,H,W); False = per-image then mean

    def forward(self, y_pred, y_true, ignore_mask=None):
        # y_pred: [B, C, H, W] logits
        # y_true: [B, 1, H, W] or [B, H, W] integer labels, ignore_index allowed
        _, c, _, _ = y_pred.shape
        probs = F.softmax(y_pred, dim=1)
        if c < 2:
            raise ValueError('CLDiceLoss expects C>=2 (binary should be encoded as C=2). Got C={}.'.format(c))        

        if y_true.dim() == 4 and y_true.size(1) == 1:
            y_true = y_true.squeeze(1)
        y_true = y_true.long()


        valid = (y_true != self.ignore_index)  # [B, H, W]
        y_safe = y_true.clone()
        y_safe[~valid] = 0
        y_onehot = F.one_hot(y_safe, num_classes=c).permute(0, 3, 1, 2).float()  # [B, C, H, W]

        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        y_onehot = y_onehot * valid_f

        # Always ignore background (class 0)
        probs = probs[:, 1:]
        y_onehot = y_onehot[:, 1:]

        if self.batch_reduction:
            dims = (0, 2, 3)  # reduce over (B,H,W)
            skel_pred = self.skel(probs)  # [C_eff, H, W]
            skel_true = self.skel(y_onehot)

            tprec_num = torch.sum(skel_pred * y_onehot, dims)
            tprec_den = torch.sum(skel_pred, dims)
            tsens_num = torch.sum(skel_true * probs, dims)
            tsens_den = torch.sum(skel_true, dims)

            tprec = (tprec_num + self.smooth) / (tprec_den + self.smooth + self.eps)
            tsens = (tsens_num + self.smooth) / (tsens_den + self.smooth + self.eps)

            cldice = 1.0 - (2.0 * tprec * tsens) / (tprec + tsens + self.eps)  # [C_eff]

            present = (torch.sum(y_onehot, dims) > 0.0).float()  # [C_eff]
            cldice = cldice * present
            denom_present = present.sum().clamp_min(1.0)
            return cldice.sum() / denom_present

        # Per-image variant: compute cldice[b, k], ignore absent classes per-image
        dims = (2, 3)
        skel_pred = self.skel(probs)  
        skel_true = self.skel(y_onehot)

        tprec_num = torch.sum(skel_pred * y_onehot, dims)  # [B, C_eff]
        tprec_den = torch.sum(skel_pred, dims)
        tsens_num = torch.sum(skel_true * probs, dims)
        tsens_den = torch.sum(skel_true, dims)

        tprec = (tprec_num + self.smooth) / (tprec_den + self.smooth + self.eps)
        tsens = (tsens_num + self.smooth) / (tsens_den + self.smooth + self.eps)

        cldice = 1.0 - (2.0 * tprec * tsens) / (tprec + tsens + self.eps)  # [B, C_eff]

        present = (torch.sum(y_onehot, dims) > 0.0).float()  # [B, C_eff]
        cldice = cldice * present
        denom_present = present.sum().clamp_min(1.0)
        return cldice.sum() / denom_present


class CLCELoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-6, ignore_index=255, skel_num_iter=10, batch_reduction=True):
        super(CLCELoss, self).__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)
        self.ignore_index = int(ignore_index)
        self.skel = SoftSkeletonize(num_iter=int(skel_num_iter))
        self.batch_reduction = bool(batch_reduction)

    def forward(self, y_pred, y_true, ignore_mask=None):
        # y_pred: [B, C, H, W] logits (C>=2)
        # y_true: [B, 1, H, W] or [B, H, W] integer labels, ignore_index allowed
        if y_true.dim() == 4 and y_true.size(1) == 1:
            y_true = y_true.squeeze(1)
        y_true = y_true.long()

        _, c, _, _ = y_pred.shape
        if c < 2:
            raise ValueError('CLCELoss expects C>=2 (binary should be C=2). Got C={}.'.format(c))

        valid = (y_true != self.ignore_index)  # [B, H, W]
        y_safe = y_true.clone()
        y_safe[~valid] = 0

        # Unreduced CE (per pixel), mask ignore
        l_unred = F.cross_entropy(y_pred, y_safe, reduction='none')  # [B, H, W]
        l_unred = l_unred * valid.float()
        l = l_unred.unsqueeze(1)  # [B,1,H,W] for broadcasting

        probs = F.softmax(y_pred, dim=1)  # [B, C, H, W]

        # Build one-hot GT, mask ignore
        y_onehot = F.one_hot(y_safe, num_classes=c).permute(0, 3, 1, 2).float()  # [B,C,H,W]
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        y_onehot = y_onehot * valid_f

        # Always ignore background (class 0)
        probs_fg = probs[:, 1:]       # [B, C-1, H, W]
        y_fg = y_onehot[:, 1:]        # [B, C-1, H, W]

        skel_pred = self.skel(probs_fg)  # [B, C-1, H, W]
        skel_true = self.skel(y_fg)      # [B, C-1, H, W]

        if self.batch_reduction:
            dims = (0, 2, 3)  # reduce over (B,H,W)
            tprec_num = torch.sum(l * skel_true, dims)  # [C-1]
            tprec_den = torch.sum(skel_true, dims)      # [C-1]
            tsens_num = torch.sum(l * skel_pred, dims)  # [C-1]
            tsens_den = torch.sum(skel_pred, dims)      # [C-1]

            tprec = tprec_num / (tprec_den + self.smooth + self.eps)
            tsens = tsens_num / (tsens_den + self.smooth + self.eps)

            present = (torch.sum(y_fg, dims) > 0.0).float()  # [C-1]
            loss_vec = (tprec + tsens) * present
            denom_present = present.sum().clamp_min(1.0)
            return loss_vec.sum() / denom_present

        # Per-image reduction then average over present classes/images
        dims = (2, 3)
        tprec_num = torch.sum(l * skel_true, dims)  # [B, C-1]
        tprec_den = torch.sum(skel_true, dims)
        tsens_num = torch.sum(l * skel_pred, dims)  # [B, C-1]
        tsens_den = torch.sum(skel_pred, dims)

        tprec = tprec_num / (tprec_den + self.smooth + self.eps)
        tsens = tsens_num / (tsens_den + self.smooth + self.eps)

        present = (torch.sum(y_fg, dims) > 0.0).float()  # [B, C-1]
        loss_mat = (tprec + tsens) * present
        denom_present = present.sum().clamp_min(1.0)
        return loss_mat.sum() / denom_present


######## MULTILABEL LOSSES ##########
    
class BCEWithIgnore(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, ignore_mask):
        #logits, labels: (B,2,H,W)
        #ignore_mask: (B,H,W) -> 1=keep, 0=ignore
        loss = self.bce(logits, labels.float())
        try:
            loss = loss * ignore_mask.unsqueeze(1)  # ask ignored pixels
        except Exception as e:
            loss = loss * ignore_mask  #if already broadcastable
        return loss.sum() / ignore_mask.sum()
    

class DiceLossMultiLabel(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-6, batch_dice=True):
        super().__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)
        self.batch_dice = bool(batch_dice)

    def forward(self, y_pred, y_true, ignore_mask=None):
        """
        y_pred: [B, C, H, W] logits
        y_true: [B, C, H, W] binary labels
        ignore_mask: [B, 1, H, W] binary mask (1=valid, 0=ignore)
        """
        probs = torch.sigmoid(y_pred)
        valid = torch.ones_like(y_true) if ignore_mask is None else ignore_mask
        probs = probs * valid 
        y_true = y_true.float() * valid

        dims = (0, 2, 3) if self.batch_dice else (2, 3)
        inter = torch.sum(probs * y_true, dims)
        denom = torch.sum(probs, dims) + torch.sum(y_true, dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth + self.eps)

        return 1.0 - dice.mean()

class CLDiceLossMultiLabel(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-6, skel_num_iter=10, batch_reduction=True):
        super().__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)
        self.skel = SoftSkeletonize(num_iter=int(skel_num_iter))
        self.batch_reduction = bool(batch_reduction)

    def forward(self, y_pred, y_true, ignore_mask=None):
        """
        y_pred: [B, C, H, W] logits
        y_true: [B, C, H, W] binary labels
        ignore_mask: [B, 1, H, W] binary mask
        """
        probs = torch.sigmoid(y_pred)
        valid = torch.ones_like(y_true) if ignore_mask is None else ignore_mask
        probs = probs * valid
        y_true = y_true.float() * valid

        # Reduce over batch or per-image
        dims = (0, 2, 3) if self.batch_reduction else (2, 3)
        skel_pred = self.skel(probs)
        skel_true = self.skel(y_true)

        tprec = (torch.sum(skel_pred * y_true, dims) + self.smooth) / (torch.sum(skel_pred, dims) + self.smooth + self.eps)
        tsens = (torch.sum(skel_true * probs, dims) + self.smooth) / (torch.sum(skel_true, dims) + self.smooth + self.eps)

        cldice = 1.0 - (2.0 * tprec * tsens) / (tprec + tsens + self.eps)
        return cldice.mean()
    

######## ZONE LOSSES ##########

#A. MULTI-CLASS

class ZoneMaskedLoss(nn.Module):
    """
    Class to compute a loss only within a specified zone.
    The zone is defined by a binary mask where pixels outside the zone are ignored in the loss computation.
    Inputs:
        - base_loss: a loss function (e.g., CrossEntropyLoss, DiceLoss)
        - ignore_index: label value to ignore in the loss computation
    """

    def __init__(self, base_loss: nn.Module, zone: str, ignore_index: int = 255):
        super().__init__()
        self.base_loss = base_loss
        self.zone = zone
        self.ignore_index = ignore_index

    def forward(self, logits, labels, zone_mask):
        """
        logits: [B, C, H, W]
        labels: [B, H, W] or [B, 1, H, W] (multi-class integer labels)
        zone_mask: [B, H, W] or [H, W], values {0, 255}
        """
        if zone_mask.dim() == 2:
            zone_mask = zone_mask.unsqueeze(0)

        # Ensure labels are [B, H, W]
        if labels.dim() == 4 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        labels_zone = labels.clone()
        labels_zone[zone_mask == 0] = self.ignore_index

        return self.base_loss(logits, labels_zone)

########### LOSS ONLY FOR CF-LOSS ###############
class CFStructuralLoss(nn.Module):
    """
    Structural CF loss for 4-class AV segmentation.
    Applies multi-scale and density constraints to artery and vein only.
    Compatible with loss_factory.
    """

    def __init__(self, alpha_fd=0.5, gamma_vd=0.08, ignore_index=255):
        super().__init__()
        self.alpha_fd = alpha_fd
        self.gamma_vd = gamma_vd
        self.ignore_index = ignore_index

    def _multi_scale_sizes(self, H, W, device):
        min_dim = min(H, W)
        n = int(torch.floor(torch.log2(torch.tensor(min_dim, dtype=torch.float32))).item())
        return 2 ** torch.arange(n, 1, -1, device=device)    

    def _multi_scale_fd(self, pred, gt, sizes):
        B = pred.shape[0]
        total = 0.0

        for size in sizes:
            size = int(size.item())
            pool = nn.AvgPool2d(kernel_size=size, stride=size, ceil_mode=True)

            pred_p = pool(pred)
            gt_p = pool(gt)

            total += size * ((pred_p - gt_p) ** 2).mean()

        return torch.sqrt(total) / B

    def forward(self, y_pred, y_true, ignore_mask=None):
        """
        y_pred: [B, C, H, W]
        y_true: [B, H, W] or [B, 1, H, W]
        """

        if y_true.dim() == 4 and y_true.size(1) == 1:
            y_true = y_true.squeeze(1)
        y_true = y_true.long()

        B, C, H, W = y_pred.shape
        device = y_pred.device

        probs = F.softmax(y_pred, dim=1)

        # Handle ignore_index
        valid = (y_true != self.ignore_index)
        if ignore_mask is not None:
            valid = valid & ignore_mask.bool()

        y_safe = y_true.clone()
        y_safe[~valid] = 0

        y_onehot = F.one_hot(y_safe, num_classes=C).permute(0, 3, 1, 2).float()
        valid_f = valid.unsqueeze(1).float()

        probs = probs * valid_f
        y_onehot = y_onehot * valid_f

        # Only artery & vein (classes 1 and 2)
        vessel_pred = probs[:, 1:3]
        vessel_gt = y_onehot[:, 1:3]

        # ---- Vessel density loss ----
        loss_vd = torch.abs(
            vessel_pred.sum(dim=[2, 3]) - vessel_gt.sum(dim=[2, 3])
        ).mean() / (H * W)

        # ---- Multi-scale fractal loss ----
        sizes = self._multi_scale_sizes(H, W, device)
        loss_fd = self._multi_scale_fd(vessel_pred, vessel_gt, sizes)

        return self.alpha_fd * loss_fd + self.gamma_vd * loss_vd


### COMPOUND LOSS ####
def get_loss(loss1, loss2=None, alpha1=1., alpha2=0., problem_type="multi_class"):
    
    if loss1 == loss2 and alpha2 != 0.:
        warnings.warn('using same loss twice, you sure?')
    
    loss_dict = dict()

    loss_dict['ce'] = CELoss()
    loss_dict['dice'] = DiceLoss()
    loss_dict['cldice'] = CLDiceLoss()
    loss_dict['clce'] = CLCELoss()
    loss_dict['bce_ignore'] = BCEWithIgnore()
    loss_dict['dice_multilabel'] = DiceLossMultiLabel()
    loss_dict['cldice_multilabel'] = CLDiceLossMultiLabel()

    loss_dict[None] = None

    if problem_type == "multi_class" or problem_type == "binary_class":
        loss_fn = CompoundLoss(loss_dict[loss1], loss_dict[loss2], alpha1, alpha2)

    else:
        if loss1 == "ce":
            loss1 = 'bce_ignore'
        elif loss1 == "dice":
            loss1 = 'dice_multilabel'
        elif loss1 == "cldice":
            loss1 = 'cldice_multilabel'
        if loss2 == "dice":
            loss2 = 'dice_multilabel'
        elif loss2 == "cldice":
            loss2 = 'cldice_multilabel'
        elif loss2 == "ce":
            loss2 = 'bce_ignore'

        loss_fn = CompoundLoss(loss_dict[loss1], loss_dict[loss2], alpha1, alpha2)
        
    return loss_fn

def get_loss_dict(args: argparse.Namespace,
                  zone_losses: str,
                  problem_type: str,
                  cf_loss: bool = False):

    """
    Define zone losses based on input string.
    
    :param zone_losses: string with zones to consider for auxiliary losses, separated by /
    :type zone_losses: str
    :param problem_type: type of problem, e.g., "multi_class" or "multi_label"
    :type problem_type: str

    Returns:
        dict: dictionary with zone loss functions
    """

    #Loss av (here always)
    if cf_loss:
        loss_av = CompoundLoss(loss1=CELoss(), loss2=CFStructuralLoss(), alpha1=1.1, alpha2=1.0)
        print('* Instantiating loss function {:.2f}*{} + {:.2f}*CFStructuralLoss'.format(1.1, 'ce', 1.0))

    else:
        loss_av = get_loss(args.loss1, args.loss2, args.alpha1, args.alpha2, problem_type=problem_type)
        print('* Instantiating loss function {:.2f}*{} + {:.2f}*{}'.format(args.alpha1, args.loss1, args.alpha2, args.loss2))

    losses_dict = dict()
    losses_dict['av'] = loss_av

    #Zone losses
    if problem_type == "multi_class" and zone_losses is not None:
        zones = zone_losses.split('/')

        for z in zones:
            if z == 'junctions':
                base_loss= get_loss(args.loss1, args.loss2, alpha1=args.alpha1, alpha2=args.alpha2, problem_type=problem_type)
                losses_dict['junctions'] = ZoneMaskedLoss(base_loss, zone='junctions')
                print('* Adding Junctions loss')

            elif z == 'arcades':
                #arcades is Dice + clDice
                base_loss=get_loss('dice', 'cldice', alpha1=1.0, alpha2=1.0, problem_type=problem_type)
                losses_dict['arcades'] = ZoneMaskedLoss(base_loss, zone='arcades')
                print('* Adding Arcades loss')

            else:
                print(f'* Unknown zone {z} for multi-class, skipping.')

    elif problem_type == "multi_label" and zone_losses is not None:
        
        zones = zone_losses.split('/')

        #Same loss as base loss (except for arcades which is always dice+cldice in multi-label version)
        for z in zones:
            if z == 'junctions':
                base_loss=get_loss(args.loss1, args.loss2, alpha1=args.alpha1, alpha2=args.alpha2, problem_type=problem_type)
                losses_dict['junctions'] = base_loss
                print('* Adding Junctions loss')

            elif z == 'arcades':
                #arcades is Dice + clDice
                base_loss=get_loss('dice', 'cldice', alpha1=1.0, alpha2=1.0, problem_type=problem_type)
                losses_dict['arcades'] = base_loss
                print('* Adding Arcades loss')

            else:
                print(f'* Unknown zone {z} for multi-label, skipping.')

    return losses_dict


def get_lambda_dict(args: argparse.Namespace,   
                    zone_losses: str):
    """
    Define lambda values for zone losses based on input string.
    
    :param zone_losses: string with zones to consider for auxiliary losses, separated by /
    :type zone_losses: str

    Returns:
        dict: dictionary with lambda values for zone losses
    """

    lambda_dict = dict()
    lambda_dict['av'] = 1.0  # Main loss weight

    if zone_losses is not None:
        zones = zone_losses.split('/')
        lambdas = [float(l) for l in args.lambdas.split('/')]
        
        if len(zones) != len(lambdas):
            raise ValueError('Number of zones and lambdas must match.')

        for z, l in zip(zones, lambdas):
            lambda_dict[z] = l

    return lambda_dict
