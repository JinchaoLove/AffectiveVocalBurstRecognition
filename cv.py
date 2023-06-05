import os
import warnings
import logging as _logging
from transformers import logging
import hydra
from omegaconf import open_dict
import pandas as pd
from train import train
from trainer.utils import set_ncclSocket

set_ncclSocket()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_QUIET'] = 'true'  # WANDB_QUIET (q), WANDB_SILENT (qq)
warnings.filterwarnings("ignore")
logging.set_verbosity(40)
_logging.disable(_logging.INFO)  # undo: logging.disable(logging.NOTSET)


def cross_validation(cfg):
    # make dirs
    if cfg.logger.dir.split(os.sep)[-1].split(':')[0] == 'None':
        cfg.logger.dir = os.path.dirname(cfg.logger.dir)
    res_out_path = os.path.join(cfg.logger.dir, 'results')
    os.makedirs(res_out_path, exist_ok=True)

    results = []
    for i in range(cfg.n_splits):
        cfg_i = cfg.copy()
        cfg_i.cv_fold = i
        cfg_i.logger.wandb.exp_name = cfg_i.logger.wandb.exp_name + f"fold_{i}"
        if i > 0:
            with open_dict(cfg_i):
                cfg_i.trainer.summary__verbose = 0
        clf_i = train(cfg_i)
        if clf_i.local_rank == 0 and clf_i.cfg.logger.enable_wandb:
            results.append(
                {'cv_fold': clf_i.cfg.cv_fold} | dict(clf_i.callbacks_)['WandbLogger'].saved_vals | {'dir': clf_i.saved_dir}
            )
    df = pd.DataFrame(results)
    clf_i.print(df)
    df.to_csv(os.path.join(res_out_path, cfg.exp_name + '.csv'), sep='\t')
    clf_i.print(f"Cross validation finished, saved in {cfg.logger.dir}")


@hydra.main(config_path='config', config_name='default', version_base=None)
def main(cfg):
    """ hydra app """
    cross_validation(cfg)


if __name__ == '__main__':
    main()
