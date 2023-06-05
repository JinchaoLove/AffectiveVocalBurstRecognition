import os
from pickle import FALSE
import warnings
import logging
from transformers import logging as ppb_logging
import hydra
from tqdm.auto import tqdm
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import Trainer, set_callbacks, hydra_init_no_call, seed_everything, to_numpy
from utils.metrics import EvalMetrics, store_results

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)  # undo: logging.disable(logging.NOTSET)
ppb_logging.set_verbosity(40)  # WARN: 30, ERROR: 40

def validation(clf, data):
    val_loader = clf.setup_dataloaders(clf.get_iterator(data, training=""), replace_sampler=False)
    val_fid_lst, val_true_lst, val_pred_lst = [], [], []
    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='val'):
        val_fid, val_true = batch['fid'], batch['labels']
        with torch.no_grad():
            val_pred = clf.module_(batch)['preds']
        val_fid_lst.append(to_numpy(val_fid))
        val_true_lst.append(to_numpy(val_true))
        val_pred_lst.append(to_numpy(val_pred))
    val_fid_lst = np.concatenate(val_fid_lst)
    val_true_lst = np.concatenate(val_true_lst)
    val_pred_lst = np.concatenate(val_pred_lst)
    clf.print(f"val size: true {val_true_lst.shape}, pred {val_pred_lst.shape}")
    val_ccc, val_ccc_each = EvalMetrics.CCC(y_true=val_true_lst, y_pred=val_pred_lst, single=True)
    clf.print("val ccc:", val_ccc)
    clf.print(data.classes, val_ccc_each)
    return val_fid_lst, val_true_lst, val_pred_lst

def test(clf, data, task, log_dir):
    test_loader = clf.setup_dataloaders(clf.get_iterator(data, training=""), replace_sampler=False)
    tst_fid_lst, tst_pred_lst = [], []
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='test'):
        tst_fid = batch['fid']
        with torch.no_grad():
            tst_pred = clf.module_(batch)['preds']
        assert not tst_pred.isnan().any(), f"Get nan in {tst_fid}: {tst_pred}"
        tst_fid_lst.append(to_numpy(tst_fid))
        tst_pred_lst.append(to_numpy(tst_pred))
    tst_fid_lst = np.concatenate(tst_fid_lst).tolist()
    tst_pred_lst = np.concatenate(tst_pred_lst)
    clf.print(f"test size: id {len(tst_fid_lst)}, pred {tst_pred_lst.shape}")
    # store
    try:
        store_results(task, tst_pred_lst, tst_fid_lst, log_dir=log_dir)
    except:
        torch.save(tst_pred_lst, 'pred.pt')
        torch.save(tst_fid_lst, 'fid.pt')

@hydra.main(config_path='config', config_name='default')
def evaluation(cfg):
    # rich.print(OmegaConf.to_yaml(cfg, resolve=True))
    seed_everything(getattr(cfg, 'seed', None))
    # if cfg.logger.dir.split(os.sep)[-1].split(':')[0] == 'None':
    #     cfg.logger.dir = os.path.dirname(cfg.logger.dir)
    cfg.logger.dir = './results'
    os.makedirs(cfg.logger.dir, exist_ok=True)

    # configs of env/model/criterion/optimizer/callbacks in trainer
    cfg.optimizer.lr_ft = cfg.optimizer.lr / cfg.ft_ratio
    if cfg.debug:
        cfg.trainer.batch_size = 2
        cfg.trainer.max_epochs = 1
        cfg.logger.enable_wandb = 0
    cfg.lite = dict(cfg.lite) | {'devices': '0,', 'strategy': None}
    cfg.trainer.ddp_test = False
    tcfg = dict(cfg.trainer) | {'saved_dir': cfg.logger.dir}  # saved_dir will be changed to W&B files dir if enabled
    tcfg['iterator__batch_size'] = 500
    # model
    for cls in ['module', 'criterion', 'optimizer', 'iterator']:
        cls_name, cls_cfg = hydra_init_no_call(getattr(cfg, cls))
        if cls_name is not None:
            tcfg[cls] = cls_name
        for k, v in cls_cfg.items():
            tcfg[cls + '__' + k] = v

    # callbacks
    tcfg['callbacks'] = set_callbacks(cfg)
    # prepare data (remember to set seed)
    # trn_data = hydra.utils.get_class(cfg.dataset._target_)(phase='train', cfg=cfg)
    val_data = hydra.utils.get_class(cfg.dataset._target_)(phase='val', cfg=cfg)
    tst_data = hydra.utils.get_class(cfg.dataset._target_)(phase='test', cfg=cfg)
    tcfg['iterator__collate_fn'] = val_data.collate_fn
    tcfg['iterator__drop_last'] = False

    # Trainer & env setup
    clf = Trainer(cfg=cfg, **cfg.lite, **tcfg)

    # set params based on Trainer properties
    # clf.set_params(criterion__weight=torch.FloatTensor(weights).to(clf.device))
    if cfg.debug:
        clf.set_params(batch_size=max(clf.num_devices * 2, 2))  # change params before call `initialize`

    # warm_start
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
    # modules eval
    clf.module_.cuda()
    clf.set_training(False)
    # test
    # test(clf, tst_data, cfg.task, cfg.logger.dir)
    # validation
    validation(clf, val_data)

@hydra.main(config_path='config', config_name='default', version_base=None)
def main(cfg):
    """ hydra app """
    evaluation(cfg)


if __name__ == '__main__':
    main()
