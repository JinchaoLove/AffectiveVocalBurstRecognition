import os
import warnings
import logging as _logging
import datetime
import hydra
from omegaconf import OmegaConf
import rich
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import wandb
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import ConcatDataset
from transformers import logging, get_linear_schedule_with_warmup

from trainer.callbacks import EarlyStopping, LRScheduler, WandbLogger, EpochScoring
from trainer.utils import set_ncclSocket, seed_everything, pl_print, hydra_init_no_call, to_numpy
from utils.metrics import EvalMetrics
from lite import Trainer


set_ncclSocket()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings("ignore")
logging.set_verbosity(40)
_logging.disable(_logging.INFO)  # undo: logging.disable(logging.NOTSET)


class CCCScoring(EpochScoring):
    # pylint: disable=unused-argument,arguments-differ,signature-differs
    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, dataset_test=None, **kwargs):
        """ Score when matchs rank, on_train and val_interval. """
        dataset = {True: dataset_train, False: dataset_valid, "": dataset_test}[self.on_train]
        if dataset is None or len(self.y_trues_) == 0 or len(self.y_preds_) == 0:
            return
        y_true = np.concatenate([to_numpy(y) for y in self.y_trues_])
        y_pred = np.concatenate([to_numpy(y) for y in self.y_preds_])
        ccc = EvalMetrics.CCC(y_true, y_pred)
        self._record_score(net.history, ccc)
        if self._is_best_score(ccc):
            self.best_scores_ = [y_true, y_pred]
            if net.local_rank == 0 and self.save:
                np.savez(os.path.join(net.saved_dir, self.name + '_best' + '.npz'), y_true=y_true, y_pred=y_pred)


def set_callbacks(cfg):
    monitor = cfg.logger.monitor
    callbacks = [
        ('trn_acc', CCCScoring('ccc', name='trn_ccc', on_train=True, save=True)),
        ('val_acc', CCCScoring('ccc', name='val_ccc', on_train=False, save=True)),
        ('tst_acc', CCCScoring('ccc', name='tst_ccc', on_train="", save=True)),
    ]
    lower_is_better = 'loss' in monitor
    early_stoper = EarlyStopping(monitor=monitor, patience=cfg.logger.earlystop, lower_is_better=lower_is_better,
                                 save_last=False, min_epochs=getattr(cfg.logger, 'min_epochs', 0))
    callbacks.append(('early_stoper', early_stoper))
    if cfg.logger.enable_wandb:
        save_model = ['best', 'last'] if cfg.logger.enable_wandb > 1 else []
        wb_args = {'dir': cfg.logger.dir, 'config': OmegaConf.to_container(cfg, resolve=True)} | dict(cfg.logger.wandb)
        callbacks.append(WandbLogger(mointor=monitor + '_best', save_model=save_model, **wb_args))
    if cfg.logger.scheduler:
        if cfg.logger.scheduler == 1:
            lr_scheduler = LRScheduler(
                policy=get_linear_schedule_with_warmup,
                event_name='lr',
                num_warmup_steps=5,
                num_training_steps=cfg.trainer.max_epochs)
        elif cfg.logger.scheduler == 2:
            lr_scheduler = LRScheduler(
                policy=CosineAnnealingLR,
                event_name='lr',
                T_max=cfg.T_max,
                eta_min=1e-6,
            )
        else:
            lr_scheduler = LRScheduler(
                policy=ReduceLROnPlateau,
                event_name='lr',
                monitor='val_loss',
                factor=0.8,
                patience=3)
        callbacks.append(lr_scheduler)
    return callbacks


def save_results(clf):
    results = {'cv_fold': clf.cfg.cv_fold} | dict(clf.callbacks_)['WandbLogger'].saved_vals | {'dir': clf.saved_dir}
    df = pd.DataFrame(results, index=[clf.cfg.cv_fold])
    os.makedirs(os.path.join(clf.saved_dir, 'results'), exist_ok=True)
    res_out = os.path.join(clf.saved_dir, 'results', clf.cfg.exp_name + '.csv')
    df.to_csv(res_out, mode='a', header=not os.path.exists(res_out), index=False, sep='\t')
    clf.print(f"^_^ Finished! Results saved in: {res_out}:\n{results}")


def train(cfg):
    # rich.print(OmegaConf.to_yaml(cfg, resolve=True))
    seed_everything(getattr(cfg, 'seed', None))
    if cfg.logger.dir.split(os.sep)[-1].split(':')[0] == 'None':
        cfg.logger.dir = os.path.dirname(cfg.logger.dir)
    os.makedirs(cfg.logger.dir, exist_ok=True)

    # configs of env/model/criterion/optimizer/callbacks in trainer
    cfg.optimizer.lr_ft = cfg.optimizer.lr / cfg.ft_ratio
    if cfg.debug:
        cfg.trainer.batch_size = 2
        cfg.trainer.max_epochs = 1
        cfg.logger.enable_wandb = 0
    if not torch.cuda.is_available():
        cfg.lite = dict(cfg.lite) | {'devices': 'auto', 'strategy': None}
    tcfg = dict(cfg.trainer) | {'saved_dir': cfg.logger.dir}  # saved_dir will be changed to W&B files dir if enabled

    # model
    for cls in ['module', 'criterion', 'optimizer', 'iterator']:
        cls_name, cls_cfg = hydra_init_no_call(getattr(cfg, cls))
        if cls_name is not None:
            tcfg[cls] = cls_name
        for k, v in cls_cfg.items():
            tcfg[cls + '__' + k] = v

    # callbacks
    tcfg['callbacks'] = set_callbacks(cfg)
    # for scorer in ['trn_acc', 'val_acc', 'tst_acc']:  # weighted accuracy (recursively set params)
    #     tcfg['callbacks__' + scorer + '__average'] = 'weighted'
    #     tcfg['callbacks__' + scorer + '__num_classes'] = cfg.data.num_labels

    # prepare data (remember to set seed)
    trn_data = hydra.utils.get_class(cfg.dataset._target_)(phase='train', cfg=cfg)
    val_data = hydra.utils.get_class(cfg.dataset._target_)(phase='val', cfg=cfg)
    # tst_data = hydra.utils.get_class(cfg.dataset._target_)(phase='test', cfg=cfg) if cfg.cv_fold >= 0 else None

    tcfg['iterator__collate_fn'] = trn_data.collate_fn

    # Trainer & env setup
    clf = Trainer(cfg=cfg, **cfg.lite, **tcfg)

    # set params based on Trainer properties
    # clf.set_params(criterion__weight=torch.FloatTensor(weights).to(clf.device))
    if cfg.debug:
        clf.set_params(batch_size=max(clf.num_devices * 2, 2))  # change params before call `initialize`

    # warm_start
    if tcfg['warm_start'] and cfg.ckpt is not None:
        clf.initialize()
        ckpt = os.path.abspath(cfg.ckpt)
        if not os.path.exists(ckpt) and 'wandb' in ckpt and clf.local_rank == 0:
            # wandb download
            api = wandb.Api()
            wbrun = api.run(f"jinchaolove/{cfg.logger.wandb.project}/{ckpt.split(os.sep)[-3].split('-')[-1]}")
            for file in wbrun.files():
                if os.path.basename(ckpt) in file.name:
                    file.download(replace=True)
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            os.system(f"mv {os.path.basename(ckpt)} {os.path.dirname(ckpt)}")
        clf.barrier()
        try:
            clf.load_params(f_module=ckpt)
        except:
            # only load part of state_dict
            state_dict = torch.load(ckpt, map_location=clf.device)
            used_keys = ['model']
            if cfg.module.proj_dim == 128:
                used_keys += ['pool', 'norm', 'proj']
                if cfg.module.shared_dim == 64:
                    used_keys += ['share', 'val', 'cnt', 'voc']
            state_dict = {k: v for k, v in state_dict.items() if any([sub in k for sub in used_keys])}
            clf.print(clf.note, f"Loaded {len(state_dict.keys())} params")
            clf.load_params(f_module=state_dict)
    else:
        clf.initialize()
    # training
    clf.print("\033[0;32m{:·^80s}\033[0m".format(f" Fitting fold: {cfg.cv_fold} "))
    if cfg.cv_fold == -2:
        clf.fit(X=ConcatDataset([trn_data, val_data]), X_val=val_data)
    else:
        clf.fit(X=trn_data, X_val=val_data)
    # log
    if clf.local_rank == 0:
        clf.save_params(f_history=os.path.join(clf.saved_dir, 'history.json'))  # clf history (loss, bsz, etc.)
    # clf.clean_up(destroy=True)
    return clf


@hydra.main(config_path='config', config_name='default', version_base=None)
def main(cfg):
    """ hydra app """
    # rich.print(OmegaConf.to_yaml(cfg, resolve=True))
    clf = train(cfg)
    if clf.local_rank == 0 and cfg.cv_fold >= 0 and clf.cfg.logger.enable_wandb:
        save_results(clf)


if __name__ == '__main__':
    # to overide hydra configs: https://hydra.cc/docs/advanced/override_grammar/basic
    # to mulit-run: https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run
    main()
