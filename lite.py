import hydra
from omegaconf import DictConfig
import torch
from torch import nn

from trainer import trainer

class Combined(torch.nn.Module):
    def __init__(self, backbone, model):
        super().__init__()
        self.backbone = backbone
        self.model = model

    def forward(self, batch):
        feat = self.backbone(batch['wav'])  # [bsz, seq_len, feat_dim]
        preds = self.model(feat, batch)
        return preds

class Trainer(trainer.Trainer):
    def initialize_module(self):
        cfg = DictConfig(self.get_params_for('module'))
        # backbone = hydra.utils.get_class(cfg.feature_extractor._target_)(cfg=cfg)
        # model = hydra.utils.get_class(cfg.model._target_)(cfg=cfg)
        # self.module_ = Combined(backbone, model)
        self.module_ = hydra.utils.get_class(cfg.model._target_)(cfg=cfg)

    def forward(self, batch, **fit_params):
        # fid, country, labels, wav
        if not bool(self.training):
            out = self.module_(batch | {'labels': None, 'country': None})
        else:
            out = self.module_(batch)
        # emo preds
        loss = self.criterion_(out['preds'], batch['labels'])['loss']
        weight = 0.1
        if self.training:
            loss0 = loss.item()
            # valance, arousal
            va_loss = self.criterion_(out['va'], batch['VA'])['loss']
            loss += va_loss * weight
            cnt_loss = nn.CrossEntropyLoss()(out['country'], batch['Country'])
            loss += cnt_loss * weight # / 4
            voc_loss = nn.CrossEntropyLoss()(out['voc'], batch['Voc_Type'])
            loss += voc_loss * weight # / 8
            if 'emo' in out.keys():
                emo_loss = self.criterion_(out['emo'], batch['emo'])['loss']
                loss += emo_loss * weight
            if len(self.history) <= 1 and len(self.history[-1]['batches']) <= 1:
                self.print(loss0, va_loss.item(), cnt_loss.item(), voc_loss.item(), emo_loss.item() if 'emo' in out.keys() else '')
                # 1.3358  5.5251  8.5082  0.2439  0.2830
                # 0.3490  1.6845 (emo loss2)  2.2642  0.1811  0.1458
        return {
            'loss': loss / self.accumulate_steps,
            'y_pred': out['preds'],
            'y_true': batch['labels']
        }

    def optim_params(self, model, optimizer_args, ft_name=['model.model'], no_decay=["bias", "norm"]):
        """ params groups for different lr and weight_decay. """
        pg0, pg1, pg2, pg3 = [], [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(sub in n.lower() for sub in ft_name):
                if optimizer_args['lr_ft'] == 0:
                    p.requires_grad = False
                    continue
                if any(sub in n.lower() for sub in no_decay):
                    pg0.append(p)
                else:
                    pg1.append(p)
            else:
                if any(sub in n.lower() for sub in no_decay):
                    pg2.append(p)
                else:
                    pg3.append(p)
        self.print("params groups lens:", len(pg0), len(pg1), len(pg2), len(pg3))
        if not torch.all(torch.Tensor([len(pg0), len(pg1), len(pg2), len(pg3)])):
            self.print("Some params groups are empty.")
        return [
            {'params': pg0, 'lr': optimizer_args['lr_ft'], 'weight_decay': 0},
            {'params': pg1, 'lr': optimizer_args['lr_ft'], 'weight_decay': optimizer_args['weight_decay']},
            {'params': pg2, 'lr': optimizer_args['lr'], 'weight_decay': 0},
            {'params': pg3, 'lr': optimizer_args['lr'], 'weight_decay': optimizer_args['weight_decay']},
        ]
