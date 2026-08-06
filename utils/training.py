import torch
from tqdm import trange

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def update_running_losses_and_postfix_weighted(t, 
                                               running_losses: dict, 
                                               batch_losses: dict, 
                                               lambdas: dict, 
                                               batch_size: int, 
                                               optimizer: torch.optim.Optimizer = None,
                                               problem_type: str = "multi_class"):
    """
    Update running losses and set tqdm postfix, including weighted total and LR.
    Only includes losses that are actually used.
    """
    # Initialize count if not exists
    if 'count' not in running_losses:
        running_losses['count'] = 0

    if problem_type == "multi_class":
        running_losses['count'] += batch_size
    
    elif problem_type == "multi_label":
        running_losses['count'] += 1  # already averaged over batch, so count by batch, not samples

    # Update running sums for each loss
    for k, v in batch_losses.items():
        if k not in running_losses:
            running_losses[k] = 0.0
        
        if problem_type == "multi_class" :
            running_losses[k] += v * batch_size  # weighted sum
        
        elif problem_type == "multi_label":
            running_losses[k] += v # already averaged over batch, so just sum

    # Compute average losses (over all samples seen so far)
    avg_losses = {k: running_losses[k] / running_losses['count'] for k in batch_losses.keys()}

    # Compute weighted total
    loss_total = sum(lambdas.get(k, 0.0) * avg_losses[k] for k in batch_losses.keys())

    # Prepare postfix
    postfix = {k: f"{avg_losses[k]:.4f}" for k in avg_losses}
    postfix['total'] = f"{loss_total:.4f}"
    if optimizer is not None:
        postfix['lr'] = f"{get_lr(optimizer):.6f}"

    t.set_postfix(postfix)
    t.update()

def train_one_epoch_multiclass(model: torch.nn.Module, 
                               loader: torch.utils.data.DataLoader, 
                               losses_dict: dict,
                               lambda_dict: dict,
                               optimizer: torch.optim.Optimizer, 
                               scheduler=None,
                               device: torch.device = None,
                               half_precision: bool = False):
   
    model.train()
    running_losses = {}
    if half_precision:
        scaler = torch.cuda.amp.GradScaler()

    with trange(len(loader)) as t:
        
        for i_batch, batch_data in enumerate(loader):
            
            inputs, labels, zones, _, _, _= batch_data
            inputs, labels = inputs.to(device), labels.to(device)
            if losses_dict.get('arcades') or losses_dict.get('junctions'):
                zones = {k: v.to(device) for k, v in zones.items()}

            optimizer.zero_grad(set_to_none=True)

            if half_precision:
                with torch.cuda.amp.autocast():
                    logits = model(inputs)

                    # Compute all losses
                    batch_losses_log = {}
                    loss_total = 0.0
                    for key, loss_fn in losses_dict.items():
                        if key == 'av':
                            loss_val = loss_fn(logits, labels)
                        elif key == 'arcades':
                            arcade_mask = zones['major_arteries'] | zones['major_veins']
                            loss_val = loss_fn(logits, labels, arcade_mask)
                        elif key == 'junctions':
                            junction_mask = zones['bifurcations_arteries'] | zones['bifurcations_veins'] | zones['crossings_roi']
                            loss_val = loss_fn(logits, labels, junction_mask)
                        else:
                            continue  #skip unknown keys
                        batch_losses_log[key] = loss_val.detach().item()
                        loss_total += lambda_dict.get(key, 0.0) * loss_val

                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            else:
                logits = model(inputs)

                # Compute all losses
                batch_losses_log = {}
                loss_total = 0.0
                for key, loss_fn in losses_dict.items():
                    if key == 'av':
                        loss_val = loss_fn(logits, labels)
                    elif key == 'arcades':
                        arcade_mask = zones['major_arteries'] | zones['major_veins']
                        loss_val = loss_fn(logits, labels, arcade_mask)
                    elif key == 'junctions':
                        junction_mask = zones['bifurcations_arteries'] | zones['bifurcations_veins'] | zones['crossings_roi']
                        loss_val = loss_fn(logits, labels, junction_mask)
                    else:
                        continue  #skip unknown keys
                    batch_losses_log[key] = loss_val.detach().item()
                    loss_total += lambda_dict.get(key, 0.0) * loss_val
        
                
                # Backprop
                loss_total.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            #Update tqdm and running losses
            update_running_losses_and_postfix_weighted(
                t=t,
                running_losses=running_losses,
                batch_losses=batch_losses_log,
                lambdas=lambda_dict,
                batch_size=inputs.shape[0],
                optimizer=optimizer
            )

    avg_losses = {k: running_losses[k] / running_losses['count'] if 'count' in running_losses else running_losses[k] for k in running_losses}
    return avg_losses
            

def train_one_epoch_multilabel(model: torch.nn.Module, 
                               loader: torch.utils.data.DataLoader, 
                               losses_dict: dict,
                               lambda_dict: dict,
                               optimizer: torch.optim.Optimizer, 
                               scheduler: torch.optim.lr_scheduler._LRScheduler,
                               device: torch.device = None,
                               half_precision: bool = False):
    """
    Train a model for one epoch for multi-label segmentation.
    
    Multi-label setup:
      - y_true: [B, C, H, W], binary labels
      - ignore_mask: optional [B, 1, H, W], 1=valid, 0=ignore
    """
    model.train()    
    
    with trange(len(loader)) as t:
        
        running_losses = {}
        
        for i_batch, batch_data in enumerate(loader):
            
            inputs, labels, zones, _, ignore_mask, _ = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_art_vein = labels[:, :2, :, :]  # only artery and vein channels
            if losses_dict.get('arcades') or losses_dict.get('junctions'):
                zones = {k: v.to(device) for k, v in zones.items()}

            if ignore_mask is not None:
                ignore_mask = ignore_mask.to(device)
            
            #Forward pass
            logits = model(inputs)
            
            #Compute loss (supporting multi-label)
            #Compute all losses
            batch_losses = {}
            batch_losses_log = {}
            for key, loss_fn in losses_dict.items():
                if key == 'av':
                    if ignore_mask is not None:
                        loss = loss_fn(logits, labels_art_vein.float(), ignore_mask)
                    else:
                        loss = loss_fn(logits, labels_art_vein.float())
                
                elif key == 'arcades':
                    arcade_mask = zones['major_arteries'] | zones['major_veins']
                    valid = ignore_mask & arcade_mask.unsqueeze(1) if ignore_mask is not None else arcade_mask.unsqueeze(1)
                    loss = loss_fn(logits, labels_art_vein.float(), valid)

                elif key == 'junctions':
                    junction_mask = zones['bifurcations_arteries'] | zones['bifurcations_veins'] | zones['crossings_roi']
                    valid = ignore_mask & junction_mask.unsqueeze(1) if ignore_mask is not None else junction_mask.unsqueeze(1)
                    loss = loss_fn(logits, labels_art_vein.float(), valid)

                batch_losses[key] = loss
                batch_losses_log[key] = loss.detach().item()

            #Compute total loss with lambda weights
            loss_total = sum(lambda_dict.get(k, 0.0) * loss for k, loss in batch_losses.items())

            #Backpropagation
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            #Logging
            update_running_losses_and_postfix_weighted(
                t=t,
                running_losses=running_losses,
                batch_losses=batch_losses_log,
                lambdas=lambda_dict,
                batch_size=inputs.shape[0],
                optimizer=optimizer
            )
    
    avg_losses = {k: running_losses[k] / running_losses['av'] if 'av' in running_losses else running_losses[k] for k in running_losses}
    return avg_losses
