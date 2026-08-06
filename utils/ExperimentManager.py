import os
import os.path as osp
import json
from datetime import datetime


class ExperimentManager:
    def __init__(self, args, fold=None, root='experiments/artery_vein_segmentation/'):
        self.args = args
        self.root = root
        self.fold = fold

        self.problem_type = args.problem_type
        self.model_name = args.model_name
        self.resize_mode = args.resize_mode
        # 1024/1024 -> 1024
        self.im_size = args.im_size.split('/')[0]
        self.batch_size = args.batch_size
        self.cf_loss = args.cf_loss

        self.exp_dir = None
        self.exp_id = None
        self.half_precision = args.half_precision
        self.five_folds = args.five_fold

    # Public API
    def setup(self):
        self.exp_dir, self.exp_id = self._create_experiment_dir()
        self._save_config()
        self._save_losses_summary()
        return self.exp_dir

    def get_checkpoint_dir(self):
        path = osp.join(self.exp_dir, 'checkpoints')
        os.makedirs(path, exist_ok=True)
        return path

    def get_log_path(self):
        return osp.join(self.exp_dir, 'train.log')

    def get_metrics_path(self):
        return osp.join(self.exp_dir, 'metrics.json')

    # Internals
    def _base_dir(self):
        return osp.join(
            self.root,
            self.problem_type,
            self.model_name,
            self.resize_mode,
            self.im_size,
            f"bs{self.batch_size}"
        )

    def _create_experiment_dir(self):

        base = self._base_dir()
        os.makedirs(base, exist_ok=True)
        if self.fold == 1:

            # --- Determine experiment ID ---
            existing = [
                d for d in os.listdir(base)
                if d.startswith('exp_') and osp.isdir(osp.join(base, d))
            ]
            if len(existing) == 0:
                next_id = 1
            else:
                ids = [int(d.split('_')[1]) for d in existing]
                next_id = max(ids) + 1

            exp_id = f"exp_{next_id:03d}"

        else:
            if self.five_folds:
                #Find the last existing experiment folder for the previous fold
                prev_fold = self.fold - 1
                prev_base = self._base_dir()
                prev_existing = [
                    d for d in os.listdir(prev_base)
                    if d.startswith('exp_') and osp.isdir(osp.join(prev_base, d))
                ]
                if len(prev_existing) == 0:
                    raise ValueError(f"No existing experiments found for fold {prev_fold} in {prev_base}. Cannot continue with fold {self.fold}.")
                
                prev_ids = [int(d.split('_')[1]) for d in prev_existing]
                current_id = max(prev_ids)
                exp_id = f"exp_{current_id:03d}"


            
            else:
                raise ValueError("Fold number specified but five_fold is False. Please set five_fold to True to use fold numbers.")


        # --- Build descriptive suffix ---
        suffix_parts = []

        # Losses
        loss_names = []
        if self.args.loss1 is not None:
            loss_names.append(f"{self.args.loss1}{int(self.args.alpha1)}")
        if self.args.loss2 is not None:
            loss_names.append(f"{self.args.loss2}{int(self.args.alpha2)}")
        if loss_names:
            suffix_parts.append("_".join(loss_names))

        if self.cf_loss:
            suffix_parts.append("cf_loss")

        if self.half_precision:
            suffix_parts.append("hp")

        # Zones + lambdas
        if self.args.zones is not None and self.args.lambdas is not None:
            zones = self.args.zones.split('/')
            lambdas = [float(x) for x in self.args.lambdas.split('/')]
            zone_parts = [f"{z}{l:.3g}" for z, l in zip(zones, lambdas)]
            suffix_parts.append("_".join(zone_parts))

        suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

        # Final experiment folder
        exp_dir = osp.join(base, exp_id + suffix)
        if self.five_folds:
            exp_dir = osp.join(exp_dir, f"fold_{self.fold}")
            print(f"Creating experiment directory for fold {self.fold}: {exp_dir}")
        os.makedirs(exp_dir, exist_ok=False)
        return exp_dir, exp_id


    def get_experiment_dir(self):
        return self.exp_dir

    def _save_config(self):
        cfg = vars(self.args).copy()

        # Add metadata
        cfg['experiment_id'] = self.exp_id
        cfg['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Parse zones & lambdas nicely
        cfg['zones_parsed'] = self._parse_zones()
        cfg['lambdas_parsed'] = self._parse_lambdas()

        config_path = osp.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)

    def _save_losses_summary(self):
        """Save a human-readable losses summary inside the experiment folder."""
        summary_lines = []

        # Main losses
        summary_lines.append("Main losses:")
        summary_lines.append(f"  loss1: {self.args.loss1}, alpha1={self.args.alpha1}")
        if self.args.loss2:
            summary_lines.append(f"  loss2: {self.args.loss2}, alpha2={self.args.alpha2}")

        # Zone losses
        if self.args.zones and self.args.lambdas:
            zones = self._parse_zones()
            lambdas = self._parse_lambdas()
            summary_lines.append("Zone losses:")
            for z, l in zip(zones, lambdas):
                summary_lines.append(f"  {z}: {l}")

        summary_path = osp.join(self.exp_dir, 'losses_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))

    def _parse_zones(self):
        if self.args.zones is None:
            return []
        return self.args.zones.split('/')

    def _parse_lambdas(self):
        if self.args.lambdas is None:
            return []
        return [float(x) for x in self.args.lambdas.split('/')]

    def _build_loss_folder_str(self):
        """Generate a short string representing the main loss combination and alphas."""
        parts = []
        if self.args.loss1:
            parts.append(f"{self.args.loss1}{int(self.args.alpha1)}")
        if self.args.loss2:
            parts.append(f"{self.args.loss2}{int(self.args.alpha2)}")
        # Optionally, you could also add zone losses here, e.g., zoneB0.5_arcades0.8
        return '_'.join(parts)