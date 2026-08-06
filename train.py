import sys, os, time, random
import os.path as osp
import numpy as np
import torch
import pathlib
import argparse
from utils.data_load import get_retinal_seg_train_val_loaders
from utils.optimizer_factory import get_optimizer
from utils.model_factory_seg import get_model
from utils.loss_factory import get_loss_dict, get_lambda_dict
from utils.validation import validate_multilabel, validate_multiclass
from utils.training import train_one_epoch_multilabel, train_one_epoch_multiclass
from utils.ExperimentManager import ExperimentManager

def get_args_parser():
    import argparse

    def str2bool(v):
        # as seen here: https://stackoverflow.com/a/43357954/3208255
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', 'yes'):
            return True
        elif v.lower() in ('false', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('boolean value expected.')

    parser = argparse.ArgumentParser(description='Training for 2d Biomedical Image Segmentation')
    parser.add_argument('--csv_path_tr', type=str, default='data_splits/tr_av_segmentation_f1.csv', help='csv path training data')
    parser.add_argument('--model_name', type=str, default='base_unet_repvgg_b3', help='architecture')
    parser.add_argument('--problem_type', type=str, default='multi_class', help='problem type: multi_class, binary_class, multi_label')
    parser.add_argument('--ignore_od', type=str2bool, nargs='?', const=True, default=True, help='whether to ignore optic disc region during training')
    parser.add_argument('--resize_mode', type=str, default='resize', help='resize or pad to target size')
    parser.add_argument('--ffa', type=str2bool, nargs='?', const=True, default=False, help='whether to use FFA module in the model')
    parser.add_argument('--cfp_norm_mode', type=str, default='imagenet', help='normalization mode for CFP module: imagenet or batchnorm')
    parser.add_argument('--ffa_norm_mode', type=str, default='per_image', help='normalization mode for FFA module: per_image or batchnorm')
    parser.add_argument('--n_channels', type=int, default=3, help='useful for multi-modal images')
    parser.add_argument('--loss1', type=str, default='ce', help='1st loss')
    parser.add_argument('--loss2', type=str, default=None, help='2nd loss')
    parser.add_argument('--load_path', type=str, default='', help='path to weight of pretrained model, if any')
    parser.add_argument('--alpha1', type=float, default=1., help='multiplier in alpha1*loss1+alpha2*loss2')
    parser.add_argument('--alpha2', type=float, default=0., help='multiplier in alpha1*loss1+alpha2*loss2')
    parser.add_argument('--im_size', type=str, default='1024/1024', help='im size/spatial xy dimension')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--optimizer', type=str, default='nadam', help='optimizer choice')
    parser.add_argument('--lr', type=float, default=1e-4, help='max learning rate')
    parser.add_argument('--n_epochs', type=int, default=2, help='training epochs')
    parser.add_argument('--vl_interval', type=int, default=1, help='how often we check performance and maybe save')
    parser.add_argument('--cyclical_lr', type=str2bool, nargs='?', const=True, default=True, help='re-start lr each vl_interval epochs')
    parser.add_argument('--metric', type=str, default='dsc', help='which metric to use for monitoring progress (AUC)')
    parser.add_argument('--seed', type=int, default=None, help='fixes random seed (slower!)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of parallel (multiprocessing) workers')
    parser.add_argument('--crossings_is_class', type=str2bool, nargs='?', const=True, default=True, help='whether crossings are their own class in multi-class problem')
    parser.add_argument('--zones', type=str, default=None , help='which anatomical zones to consider for auxiliary losses, separated by /')  # e.g. junctions/arcades
    parser.add_argument('--lambdas', type=str, default=None , help='lambdas for zone losses, separated by /')
    parser.add_argument('--cf_loss', type=str2bool, nargs='?', const=True, default=False, help='whether to use the CFStructuralLoss as a second loss in a compound loss for artery/vein segmentation')
    parser.add_argument('--half_precision', type=str2bool, nargs='?', const=True, default=False, help='whether to train with half precision (fp16). Only recommended for 16GB+ GPU RAM and if you want to speed up training at the cost of some numerical precision. Not compatible with all losses (e.g. dice loss) or schedulers (e.g. CosineAnnealingWarmRestarts). Use with --cyclical_lr False and --loss2 ce for best results.')
    parser.add_argument('--five_fold', type=str2bool, nargs='?', const=True, default=False, help='whether to do 5-fold cross validation (only for multi-class problem)')
    args = parser.parse_args()

    return args


def set_seeds(seed_value, use_cuda, use_mps, MAC:bool=False):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda and not MAC:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
    
    if MAC and use_mps:
        torch.mps.manual_seed(seed_value)
    

def set_tr_info(tr_info, epoch=0, ovft_metrics=None, vl_metrics=None, best_epoch=False):
    """
    Update training info dictionary with metrics and losses.

    ovft_metrics and vl_metrics should be tuples:
        (metrics_dict, avg_losses_dict)
    where metrics_dict contains keys like:
        'global', 'junctions', 'junction_recalls', 'arcades_cldice'
    """
    if best_epoch:
        tr_info['best_tr_dsc_per_class'] = tr_info['tr_dscs_per_class'][-1]
        tr_info['best_vl_dsc_per_class'] = tr_info['vl_dscs_per_class'][-1]
        tr_info['best_tr_loss'] = tr_info['tr_losses'][-1]
        tr_info['best_vl_loss'] = tr_info['vl_losses'][-1]
        tr_info['best_epoch'] = epoch
    else:
        ovft_metrics_dict, ovft_losses = ovft_metrics
        vl_metrics_dict, vl_losses = vl_metrics

        # Per-class Dice
        tr_info['tr_dscs_per_class'].append(ovft_metrics_dict['global'])
        tr_info['vl_dscs_per_class'].append(vl_metrics_dict['global'])

        # Macro Dice
        tr_info['tr_dscs'].append(float(np.nanmean(ovft_metrics_dict['global'])))
        tr_info['vl_dscs'].append(float(np.nanmean(vl_metrics_dict['global'])))

        # Losses
        tr_info['tr_losses'].append(ovft_losses)
        tr_info['vl_losses'].append(vl_losses)

        tr_info['tr_junction_dsc'].append(ovft_metrics_dict.get('junctions', np.nan))
        tr_info['vl_junction_dsc'].append(vl_metrics_dict.get('junctions', np.nan))

        # Individual junction recalls
        for k in ["artery_bif", "vein_bif", "crossings"]:
            tr_info[f"tr_{k}_recalls"].append(ovft_metrics_dict.get('junction_recalls', {}).get(k, np.nan))
            tr_info[f"vl_{k}_recalls"].append(vl_metrics_dict.get('junction_recalls', {}).get(k, np.nan))

        tr_info['tr_arcades_cldice'].append(ovft_metrics_dict.get('arcades_cldice', np.nan))
        tr_info['vl_arcades_cldice'].append(vl_metrics_dict.get('arcades_cldice', np.nan))

    return tr_info


def get_eval_string(tr_info, epoch, finished=False, vl_interval=1):
    ep_idx = len(tr_info['tr_losses']) - 1
    if finished:
        ep_idx = epoch
        epoch = (epoch + 1) * vl_interval - 1

    tr_losses = tr_info['tr_losses'][ep_idx]
    vl_losses = tr_info['vl_losses'][ep_idx]
    tr_pc = tr_info['tr_dscs_per_class'][ep_idx]
    vl_pc = tr_info['vl_dscs_per_class'][ep_idx]

    tr_macro = float(np.nanmean(tr_pc))
    vl_macro = float(np.nanmean(vl_pc))

    pad = ' ' * 9
    lines = []

    # --- Header: macro DSC and total loss ---
    lines.append(
        f"Ep. {str(epoch + 1).zfill(3)}: Train/Val DSC: {tr_macro:.2f}/{vl_macro:.2f} - "
        f"Loss: {tr_losses['total']:.4f}/{vl_losses['total']:.4f}"
    )

    # --- Per-class Dice ---
    for j, c in enumerate(range(1, len(CLASS_DICT))):
        lines.append(f"{pad}Train/Val DSC - {CLASS_DICT[c]}: {tr_pc[j]:.2f}/{vl_pc[j]:.2f}")

    lines.append('')

    # --- Junctions Dice (per class) ---
    tr_junctions = tr_info['tr_junction_dsc'][ep_idx]
    vl_junctions = tr_info['vl_junction_dsc'][ep_idx]
    for j, c in enumerate(range(1, len(CLASS_DICT))):
        lines.append(f"{pad}Junctions DSC - {CLASS_DICT[c]}: Train {tr_junctions[j]:.2f} / Val {vl_junctions[j]:.2f}")

    lines.append('')

    # --- Junction recalls ---
    for k in ["artery_bif", "vein_bif", "crossings"]:
        lines.append(
            f"{pad}Junction recall - {k}: "
            f"Train {tr_info[f'tr_{k}_recalls'][ep_idx]:.4f} / "
            f"Val {tr_info[f'vl_{k}_recalls'][ep_idx]:.4f}"
        )
    lines.append('')

    # --- Arcades clDice (per class) ---
    tr_arcades = tr_info['tr_arcades_cldice'][ep_idx]
    vl_arcades = tr_info['vl_arcades_cldice'][ep_idx]
    for j, c in enumerate(range(1, len(CLASS_DICT))):
        lines.append(f"{pad}Arcades clDice - {CLASS_DICT[c]}: Train {tr_arcades[j]:.2f} / Val {vl_arcades[j]:.2f}")
    lines.append('')

    # --- Loss breakdown ---
    loss_keys = [k for k in tr_losses.keys() if k != 'total' and k not in
                 ["junctions", "junction_recalls", "arcades_cldice"]]
    if len(loss_keys) > 0:
        lines.append(pad + 'Loss breakdown (Train / Val):')
        for k in loss_keys:
            lines.append(f"{pad}  {k}: {tr_losses[k]:.4f} / {vl_losses.get(k, float('nan')):.4f}")

    # Print in terminal
    print('\n'.join(lines), flush=True)
    return '\n'.join(lines)


def init_tr_info():
    """
    Initialize training info dictionary with all necessary keys for metrics and losses.
    """
    tr_info = dict()
    tr_info['best_epoch'] = 0
    tr_info['best_tr_dsc'], tr_info['best_vl_dsc'] = 0, 0
    tr_info['best_tr_loss'], tr_info['best_vl_loss'] = float('inf'), float('inf')
    tr_info['tr_dscs'], tr_info['vl_dscs'] = [], []
    tr_info['tr_dscs_per_class'], tr_info['vl_dscs_per_class'] = [], []
    tr_info['tr_losses'], tr_info['vl_losses'] = [], []

    tr_info['tr_junction_dsc'], tr_info['vl_junction_dsc'] = [], []

    # Junction recalls
    for k in ["artery_bif", "vein_bif", "crossings"]:
        tr_info[f"tr_{k}_recalls"] = []
        tr_info[f"vl_{k}_recalls"] = []

    tr_info['tr_arcades_cldice'], tr_info['vl_arcades_cldice'] = [], []

    return tr_info


def train_model(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer, 
                losses_dict: dict, 
                lambda_dict: dict,
                tr_loader: torch.utils.data.DataLoader, 
                ovft_loader: torch.utils.data.DataLoader, 
                vl_loader: torch.utils.data.DataLoader, 
                scheduler: torch.optim.lr_scheduler._LRScheduler, 
                device:torch.device,
                n_epochs: int, 
                vl_interval: int, 
                save_path: str, 
                CLASS_DICT: dict,
                ckpt_dir: str,
                MAC: bool = False,
                half_precision: bool = False):
    
    best_metric, best_epoch = 0, 0
    tr_info = init_tr_info()

    for epoch in range(n_epochs):
        print('Epoch {:d}/{:d}'.format(epoch + 1, n_epochs))
        
        #Train one epoch
        if problem_type == "multi_class":
            train_one_epoch_multiclass(model=model, 
                                       loader=tr_loader, 
                                       losses_dict=losses_dict, 
                                       lambda_dict=lambda_dict, 
                                       optimizer=optimizer, 
                                       scheduler=scheduler, 
                                       device=device,
                                       half_precision=half_precision)
        
        elif problem_type == "multi_label":
            train_one_epoch_multilabel(model=model, 
                                       loader=tr_loader, 
                                       losses_dict=losses_dict,
                                       lambda_dict=lambda_dict,
                                       optimizer=optimizer, 
                                       scheduler=scheduler, 
                                       device=device,
                                       half_precision=half_precision)
        
        if (epoch + 1) % vl_interval == 0:
            
            with torch.no_grad():
                #Validate on overfit set
                if problem_type == "multi_class":
                    ovft_metrics, avg_ovft_losses = validate_multiclass(model=model,
                                                                         CLASS_DICT=CLASS_DICT, 
                                                                         loader=ovft_loader, 
                                                                         losses_dict=losses_dict, 
                                                                         lambda_dict=lambda_dict, 
                                                                         device=device,
                                                                         half_precision=half_precision)
                    
                    vl_metrics, avg_vl_losses = validate_multiclass(model=model, 
                                                                    CLASS_DICT=CLASS_DICT, 
                                                                    loader=vl_loader, 
                                                                    losses_dict=losses_dict, 
                                                                    lambda_dict=lambda_dict, 
                                                                    device=device,
                                                                    half_precision=half_precision)

                elif problem_type == "multi_label":
                    ovft_metrics, avg_ovft_losses = validate_multilabel(model=model, CLASS_DICT=CLASS_DICT, loader=ovft_loader, losses_dict=losses_dict, lambda_dict=lambda_dict, device=device)
                    vl_metrics, avg_vl_losses = validate_multilabel(model=model, CLASS_DICT=CLASS_DICT, loader=vl_loader, losses_dict=losses_dict, lambda_dict=lambda_dict, device=device)
            
            #Update training info
            tr_info = set_tr_info(tr_info=tr_info,
                                    epoch=epoch,
                                    ovft_metrics=(vl_metrics, avg_vl_losses),
                                    vl_metrics=(vl_metrics, avg_vl_losses))
            s = get_eval_string(tr_info, epoch, finished=False, vl_interval=vl_interval)
            print(s, flush=True)

            with open(osp.join(save_path, 'train_log.txt'), 'a') as f:
                print(s, file=f)
            
            #check if performance was better than anyone before and checkpoint if so
            curr_dsc = tr_info['vl_dscs'][-1]   
            is_better = curr_dsc > best_metric
            if is_better:
                print('-------- Best dsc attained. {:.2f} --> {:.2f} --------'.format(best_metric, curr_dsc))
                torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_metric': curr_dsc,}, 
                            osp.join(ckpt_dir, 'best_model.pth')
                    )
                best_metric, best_epoch = curr_dsc, epoch + 1

                tr_info = set_tr_info(tr_info, epoch+1, best_epoch=True)
            else:
                print('-------- Best dsc so far {:.2f} at epoch {:d} | Current {:.2f} --------'.format(best_metric, best_epoch, curr_dsc))
                
    del model, tr_loader, vl_loader
    
    #maybe this works also? tr_loader.dataset._fill_cache
    if not MAC and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    elif MAC and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    else:
        pass

    return tr_info


if __name__ == '__main__':

    args = get_args_parser()
    #1. Define device, seeds for reproducibility
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    if use_mps:
        device = torch.device("mps")
    else:
        device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('* Using device {}'.format(device))

    seed_value = args.seed if args.seed is not None else 0
    MAC = sys.platform == "darwin"
    set_seeds(seed_value, use_cuda, use_mps, MAC=MAC)

    #2. Gather parse arguments, create model, dataloaders, optimizer, loss
    model_name = args.model_name
    problem_type = args.problem_type
    ignore_od = args.ignore_od
    lr, bs = args.lr, args.batch_size
    n_epochs, vl_interval, metric = args.n_epochs, args.vl_interval, args.metric
    csv_path_tr = args.csv_path_tr
    crossings_is_class = args.crossings_is_class    
    resize_mode = args.resize_mode  
    zone_losses = args.zones
    ffa, cfp_norm_mode, ffa_norm_mode = args.ffa, args.cfp_norm_mode, args.ffa_norm_mode
    cf_loss = args.cf_loss
    hp = args.half_precision
    five_fold = args.five_fold
    n_folds = 5 if five_fold else 1

    #sizes
    im_size = args.im_size.split('/')
    im_size = tuple(map(int, im_size))
    
    CLASS_DICT = dict()
    if problem_type == "multi_class": 
        # training classes (model will have num_classes = len(CLASS_DICT) == 3)
        CLASS_DICT = {0: 'background', 1: 'artery', 2: 'vein',  3: 'crossings'}
        num_classes = len(CLASS_DICT)

    elif problem_type == "multi_label":
        CLASS_DICT = {0: 'background', 1: 'artery', 2: 'vein',  3: 'crossings'}
        num_classes = 2
    
    elif problem_type == "binary_class":
        CLASS_DICT = {0: 'background', 1: 'vessels'}
        num_classes = len(CLASS_DICT)
    
    in_c = args.n_channels
    model = get_model(args.model_name, num_classes=num_classes, in_c=in_c)
    print('* Instantiating a {} modeln num_classes={}'.format(model_name, num_classes)) 
    print(args.load_path)
    if not five_fold and args.load_path != '':
        model.load_state_dict(torch.load(osp.join(args.load_path, 'best_model.pth'), weights_only=True))
        print('* Loaded pretrained weights')

    for fold in range(1, n_folds + 1):

        csv_path_tr_new = args.csv_path_tr
        if five_fold:
            csv_path_tr_new = args.csv_path_tr.replace('_f1.csv', f'_f{fold}.csv')
        args.csv_path_tr = csv_path_tr_new

        if five_fold and args.load_path != '':
            #Load model from curent fold to fine-tune
            ckpt = torch.load(
                    osp.join(args.load_path, f'fold_{fold}', 'checkpoints', 'best_model.pth'),
                    map_location=device,
                )
            model.load_state_dict(ckpt['model_state_dict'])
            print(f'* Loaded pretrained weights for fold {fold} from {args.load_path}')

        #2. Create save directory and save config file
        exp_manager = ExperimentManager(args, fold=fold)
        save_path = exp_manager.setup()
        exp_dir = exp_manager.get_experiment_dir()
        print('* Experiment directory: {}'.format(exp_dir))

        args.csv_path_tr = csv_path_tr
        with open(os.path.join(exp_dir, "experiment_path.txt"), "w") as f:
            f.write(exp_dir)
        #dataloaders
        print('* Creating Dataloaders, batch size={}'.format(bs))
        tr_loader, ovft_loader, vl_loader = get_retinal_seg_train_val_loaders(csv_path_tr=csv_path_tr_new,
                                                                            batch_size=bs,
                                                                            tg_size=im_size,
                                                                            num_workers=args.num_workers,
                                                                            resize_mode=resize_mode,
                                                                            problem_type=problem_type,
                                                                            crossings_is_class=crossings_is_class,
                                                                            cfp_norm_mode=cfp_norm_mode,
                                                                            ffa_norm_mode=ffa_norm_mode,
                                                                            ignore_od=ignore_od,
                                                                            ffa=ffa)

        #optimizer, scheduler, loss_fn
        model = model.to(device)
        print('* Total params: {0:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        print('*\n Optimizing with {} algorithm'.format(args.optimizer))
        optimizer = get_optimizer(args.optimizer, model, lr)

        if args.cyclical_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=vl_interval*len(tr_loader), eta_min=0)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs*len(tr_loader), eta_min=0)

        loss_dict = get_loss_dict(args=args,
                                zone_losses=zone_losses,
                                problem_type=problem_type,
                                cf_loss=cf_loss)
        
        lambda_dict = get_lambda_dict(args=args,
                                    zone_losses=zone_losses)

        #4. Train model
        print('* Starting to train\n', '-' * 10)
        start = time.time()
        ckpt_dir = exp_manager.get_checkpoint_dir()
        tr_info = train_model(model=model,
                                optimizer=optimizer,
                                losses_dict=loss_dict,
                                lambda_dict=lambda_dict,
                                tr_loader=tr_loader,
                                ovft_loader=ovft_loader,
                                vl_loader=vl_loader,
                                scheduler=scheduler,
                                device=device,
                                n_epochs=n_epochs,
                                vl_interval=vl_interval,
                                CLASS_DICT=CLASS_DICT,
                                save_path=save_path,
                                ckpt_dir=ckpt_dir,
                                MAC=MAC,
                                half_precision=hp)
        end = time.time()

        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print('Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))

        #5. Save training log
        with open(osp.join(save_path, 'log.txt'), 'a') as f:
            
            best_ep_idx = max(tr_info['best_epoch'] - 1, 0) // vl_interval
            s_best = 'Best epoch = {}/{}'.format(tr_info['best_epoch'], n_epochs)
            
            print(s_best, file=f)
            print(get_eval_string(tr_info, epoch=best_ep_idx, finished=True, vl_interval=vl_interval), file=f)
            print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)

        print('Done. Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))

    #Print for shell capture
    print(f"EXPERIMENT_PATH={pathlib.Path(str(exp_dir)).parent}")
