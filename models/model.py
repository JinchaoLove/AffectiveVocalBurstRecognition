from os.path import join
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import is_initialized
import hydra
from omegaconf import DictConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, Wav2Vec2Model
from models.utils import grad_reverse, sf_argmax, load_ssl_model
from models.basics import SelfAttentionPooling, mask_mean

class BaselineModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_dim = cfg.proj_dim
        bn_type = nn.SyncBatchNorm if is_initialized() else nn.BatchNorm1d
        self.model = nn.Sequential(
            bn_type(feat_dim),
            nn.Linear(feat_dim, 512),
            # bn_type(1024),
            # nn.LeakyReLU(),
            # nn.Linear(1024, 512),
            bn_type(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            bn_type(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            bn_type(64),
            nn.LeakyReLU(),
            nn.Linear(64, 10),
            nn.Sigmoid()
        )

    def forward(self, feat, batch):
        if feat.ndim == 3:
            feat = feat.mean(1)
        pred = self.model(feat)
        return pred  # dict(pred_final=pred)

class CoarseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_dim = cfg.proj_dim
        #self.inorm = nn.InstanceNorm1d(feat_dim)
        self.model = nn.Sequential(nn.Linear(feat_dim * 2, 10), nn.Sigmoid())
        self.a = nn.Linear(feat_dim, 1, bias=False)
        self.main_prediction = nn.Sequential(nn.Linear(feat_dim, 10))
        self.main_embedding = nn.Embedding(10, feat_dim)

    def forward(self, feat, batch):
        # feat: [#B, #seqlen, #feat_dim]
        #out = self.inorm(feat.unsqueeze(1)).squeeze(1)
        #out = feat - out
        weight = torch.softmax(self.a(feat), 1)
        feat = torch.sum(weight * feat, dim=1)
        main_emotion = self.main_prediction(feat)

        #if batch.get('main_emotion') is not None:
        #   main_emb = self.main_embedding(batch['main_emotion'])
        #else:
        dist = torch.softmax(main_emotion, dim=-1)
        main_emb = torch.sum(self.main_embedding.weight.unsqueeze(0) * dist.unsqueeze(-1), dim=1)
        feat = torch.cat([feat, main_emb], dim=-1)
        out = self.model(feat)
        return dict(pred_final=out, main_emotion=main_emotion)

class PoolingModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_dim = cfg.proj_dim
        self.model = nn.Sequential(nn.Linear(feat_dim, 10), nn.Sigmoid())
        self.a = nn.Linear(feat_dim, 1, bias=False)
        #self.a2 = nn.Linear(feat_dim, 1, bias=False)
        #self.dropout = nn.Dropout(0.5)

    def forward(self, feat, batch):
        # feat: [B, L, F]
        weight = torch.softmax(self.a(feat), 1)
        feat = torch.sum(weight * feat, dim=1)
        #feat = self.dropout(feat)
        score = self.model(feat)
        return dict(pred_final=score)
    '''
    def forward(self, feat, batch):
        # feat: [layer, B, L, F]
        weight1 = torch.softmax(self.a(feat), dim=2)
        feat = torch.sum(weight1 * feat, dim=2).transpose(0, 1) #[B, layer, F]
        weight2 = torch.softmax(self.a2(feat), dim=1)
        feat = torch.sum(weight2 * feat, dim=1)
        score = self.model(feat)
        return dict(pred_final=score)
    '''

class StackModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_dim = cfg.proj_dim
        self.layer1 = nn.Sequential(nn.Linear(feat_dim, 10), nn.Sigmoid())
        self.layer2 = nn.Sequential(nn.Linear(feat_dim + 10, 10), nn.Sigmoid())
        self.a = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, feat, batch):
        # feat: [#B, #seqlen, #feat_dim]
        weight = torch.softmax(self.a(feat), 1)
        feat = torch.sum(weight * feat, dim=1)
        prescore = self.layer1(feat)
        feat = torch.cat([feat, prescore], dim=-1)
        score = self.layer2(feat)
        return {'pred_1': prescore, 'pred_final': score}

class RNNCCModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_dim = cfg.proj_dim
        self.c = 10
        self.a = nn.Linear(feat_dim, 1, bias=False)
        self.zero_score = nn.Parameter(torch.zeros(1).unsqueeze(-1), requires_grad=False)
        self.gru = nn.GRU(feat_dim + 1, feat_dim, batch_first=True)
        self.out_layer = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_prob(self):
        k = 1
        p = (k + 1) / (k + math.exp(self.epoch / k))
        return p

    def forward(self, feat, batch):
        '''
            Args:
                feat: [B, seqlen, feat_dim]
        '''
        gt = batch.get('emotion')
        weight = torch.softmax(self.a(feat), dim=1)
        feat = torch.sum(weight * feat, dim=1)  # [B, feat_dim]
        hidden_state = None
        out_buf = []
        input_feat = torch.cat([feat, self.zero_score.expand(feat.shape[0], -1)], dim=-1).unsqueeze(1)  # [B, 1, feat_dim+1]
        for i in range(self.c):
            out_state, hidden_state = self.gru(input_feat, hidden_state)
            score = self.out_layer(out_state.squeeze(1))  # [B, 1]
            if gt is None or self.cfg.chain_strategy == 'pred':
                input_feat = torch.cat([feat, score], dim=-1).unsqueeze(1)
            elif gt is not None and self.cfg.chain_strategy == 'ss':
                p = torch.rand(feat.shape[0], dtype=feat.dtype, device=feat.device).unsqueeze(-1)
                threshold = self.get_prob()
                next_input = torch.where(p >= threshold, score, gt[:, i:i + 1].type_as(score))
                input_feat = torch.cat([feat, next_input], dim=-1).type_as(feat).unsqueeze(1)
            elif gt is not None and self.cfg.chain_strategy == 'gt':
                input_feat = torch.cat([feat, gt[:, i:i + 1].type_as(feat)], dim=-1).unsqueeze(1)
            out_buf.append(score)
        out = torch.cat(out_buf, -1)
        return {'pred_final': out}

class DirectRegressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_dim = cfg.proj_dim
        num_classes = {'high': 10, 'two': 2, 'culture': 40, 'type': 8}
        self.c = num_classes[self.cfg.task]
        # self.norm = nn.SyncBatchNorm(feat_dim) if is_initialized() else nn.BatchNorm1d(feat_dim)
        self.norm = nn.LayerNorm(feat_dim)
        self.fc = nn.Linear(feat_dim, self.c)
        self.sigmoid = nn.Sigmoid()
        self.dp = nn.Dropout(cfg.dropout)

    def forward(self, feat, batch):
        src, attn_mask = feat['src'], feat['attn_mask']
        src = mask_mean(src, attn_mask)
        assert src.ndim == 2, f"Input shape error: {src.shape}"
        out = self.norm(src)
        out = self.sigmoid(self.fc(self.dp(out)))
        return out

class AttnRegressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_dim = cfg.proj_dim
        self.num_classes = {'high': 10, 'two': 2, 'culture': 40, 'type': 8}[self.cfg.task]
        self.attn_pool = SelfAttentionPooling(feat_dim, n_heads=self.num_classes)
        self.norm = nn.SyncBatchNorm(self.num_classes) if is_initialized() else nn.BatchNorm1d(self.num_classes)
        self.fc = nn.Linear(feat_dim, self.num_classes)
        self.fc_emo = nn.Linear(self.num_classes * self.num_classes, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dp = nn.Dropout(0.2)

    def forward(self, feat, batch):
        # B, T, F
        src, attn_mask = feat['src'], feat['attn_mask']
        assert src.ndim == 3, f"Input shape error: {src.shape}"
        out = self.attn_pool(src, attn_mask).transpose(1, 2)  # (B, num_classes, F)
        out = self.norm(out)
        out = self.fc(self.dp(out)).reshape(out.shape[0], -1)
        out = self.fc_emo(out)
        out = self.sigmoid(out)
        return out

    def attention(self, q, k=None, v=None):
        k = q if k is None else k
        v = q if v is None else v
        # (B, Nt, F) x (B, F, Ns) -> (B, Nt, Ns)
        scores = torch.bmm(q, k.transpose(1, 2)) / (q.shape[2] ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        # (B, Nt, Ns) x (B, Ns, F) -> (B, Nt, F)
        return torch.bmm(attn_weights, v)

class EmbAttnRegressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feat_dim = cfg.proj_dim
        self.cfg = cfg
        self.num_classes = {'high': 10, 'two': 2, 'culture': 40, 'type': 8}[self.cfg.task]
        self.label_emb = nn.Parameter(torch.empty(self.num_classes, self.feat_dim))  # !!! not work
        nn.init.kaiming_uniform_(self.label_emb, a=math.sqrt(5))
        # self.EmoTimeAttn = nn.MultiheadAttention(self.feat_dim, num_heads=1, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(self.feat_dim, 1)
        self.fc_emo = nn.Linear(self.num_classes, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat, batch):
        src, attn_mask = feat['src'], feat['attn_mask']
        assert src.ndim == 3, f"Input size should be (B, T, F), but get: {src.size}"
        label_embs = self.label_emb.repeat(src.shape[0], 1, 1)
        assert label_embs.shape[1:] == torch.Size([self.num_classes, self.feat_dim]), f"Repeat error, {label_embs.shape}"
        # print(*[f"{c:.4f}" for c in self.label_emb[:, 0]])
        out = self.attention(label_embs, src, src)  # (B, num_classes, F)
        # out = self.attention(out)  # attention across emotions
        # out, _ = self.EmoTimeAttn(label_embs, src, src)
        # out, _ = self.EmoAttn(out, out, out)
        out = self.fc(out).squeeze(-1)
        out = self.fc_emo(out)
        out = self.sigmoid(out)
        return out

    def attention(self, q, k=None, v=None):
        k = q if k is None else k
        v = q if v is None else v
        # (B, Nt, F) x (B, F, Ns) -> (B, Nt, Ns)
        scores = torch.bmm(q, k.transpose(1, 2)) / (q.shape[2] ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        # (B, Nt, Ns) x (B, Ns, F) -> (B, Nt, F)
        return torch.bmm(attn_weights, v)


class ChainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_dim = cfg.proj_dim
        num_classes = {'high': 10, 'two': 2, 'culture': 40, 'type': 8}
        self.c = num_classes[self.cfg.task]
        self.attn_pool = SelfAttentionPooling(feat_dim)
        self.chain = nn.ModuleList()
        for i in range(self.c):
            linear = nn.Linear(feat_dim + i, 1)
            self.chain.append(linear)
        self.sigmoid = nn.Sigmoid()
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_prob(self):
        # p =  (1 + e(-x)) / 2
        k = 2
        p = (k + 1) / (k + math.exp(self.epoch / k))
        return p

    def forward(self, feat, batch=None):
        # feat: [B, seqlen, feat_dim]
        src, attn_mask = feat['src'], feat['attn_mask']
        gt = batch.get('labels') if batch is not None else None
        assert src.ndim == 3, f"Input shape error: {src.shape}"
        input_feat = self.attn_pool(**feat)  # [B, feat_dim]
        out_buf = []
        chain_order = list(range(self.c))
        for i in chain_order:
            score = self.sigmoid(self.chain[i](input_feat))
            if gt is None or self.cfg.chain_strategy == 'pred':  # or not self.training:
                input_feat = torch.cat([input_feat, score], dim=-1)
            elif gt is not None and self.cfg.chain_strategy == 'ss':
                p = torch.rand(input_feat.shape[0], dtype=input_feat.dtype, device=input_feat.device).unsqueeze(-1)
                threshold = self.get_prob()
                next_input = torch.where(p >= threshold, score, gt[:, i:i + 1].type_as(score))
                input_feat = torch.cat([input_feat, next_input], dim=-1).type_as(input_feat)
            elif gt is not None and self.cfg.chain_strategy == 'gt':
                input_feat = torch.cat([input_feat, gt[:, i:i + 1].type_as(input_feat)], dim=-1).type_as(input_feat)
            out_buf.append(score)
        out = torch.cat(out_buf, -1)
        return out

class BiChain(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_dim = cfg.proj_dim
        num_classes = {'high': 10, 'two': 2, 'culture': 40, 'type': 8}
        self.c = num_classes[self.cfg.task]
        # self.a = nn.Linear(feat_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.epoch = 0
        self.chain = nn.ModuleList()
        self.chain_reversed = nn.ModuleList()
        for i in range(self.c):
            self.chain.append(nn.Linear(feat_dim + i, 1))
            self.chain_reversed.append(nn.Linear(feat_dim + i, 1))
        # self.chain_weights = nn.Parameter(torch.ones(self.c, 2) / 2.0)  # should be [0, 1]
        # self.dp = nn.Dropout(0.2)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_prob(self):
        k = 1
        p = (k + 1) / (k + math.exp(self.epoch / k))
        return p

    def forward(self, feat, batch):
        # feat: [B, feat_dim]
        src, attn_mask = feat['src'], feat['attn_mask']
        gt = batch.get('labels')
        assert src.ndim == 2, f"Input shape error{src.shape}."
        out_buf, out_buf_rev = [], []
        input_feat, input_feat_rev = src, src
        for i in range(self.c):
            score = self.sigmoid(self.chain[i](input_feat))
            score_rev = self.sigmoid(self.chain_reversed[i](input_feat_rev))  # gt reversed
            if gt is None or self.cfg.chain_strategy == 'pred':  # or not self.training
                input_feat = torch.cat([input_feat, score], dim=-1)
                input_feat_rev = torch.cat([input_feat_rev, score_rev], dim=-1)
            elif gt is not None and self.cfg.chain_strategy == 'ss': # and self.training:
                p = torch.rand(input_feat.shape[0], dtype=input_feat.dtype, device=input_feat.device).unsqueeze(-1)
                threshold = self.get_prob()
                next_input = torch.where(p >= threshold, score, gt[:, i:i + 1].type_as(score))
                input_feat = torch.cat([input_feat, next_input], dim=-1).type_as(input_feat)
                next_input_rev = torch.where(p >= threshold, score_rev, gt[:, self.c - 1 - i:self.c - i].type_as(score_rev))
                input_feat_rev = torch.cat([input_feat_rev, next_input_rev], dim=-1).type_as(input_feat_rev)
            # else:  # see gt in validation for check
            elif gt is not None and self.cfg.chain_strategy == 'gt':
                input_feat = torch.cat([input_feat, gt[:, i:i + 1].type_as(input_feat)], dim=-1).type_as(input_feat)
                input_feat_rev = torch.cat([input_feat_rev, gt[:, self.c - 1 - i:self.c - i].type_as(input_feat_rev)], dim=-1).type_as(input_feat_rev)
            out_buf.append(score)
            out_buf_rev.append(score_rev)
        out = (torch.cat(out_buf, -1) + torch.cat(out_buf_rev, -1).flip(-1)) / 2.0
        # out = torch.stack([torch.cat(out_buf, -1), torch.cat(out_buf_rev, -1).flip(-1)], dim=-1)  # (B, C, 2)
        # out = (out * F.softmax(self.chain_weights, dim=-1).view(1, -1, 2)).sum(-1)  # softmax for channel dim, then sum
        return out


def prepare_mask(length, shape, dtype, device):
    #Modified from huggingface
    mask = torch.zeros(
        shape, dtype=dtype, device=device
    )
    # these two operations makes sure that all values
    # before the output lengths indices are attended to
    mask[
        (torch.arange(mask.shape[0], device=device), length - 1)
    ] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask

class EmptyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

class Wav2vecWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wav2vec2 = load_ssl_model(self.cfg.ssl_model)
        if cfg.ssl_ser_ckpt is not None:
            self.load_ser_ckpt()
        self.wav2vec_config = self.wav2vec2.config
        self.num_hidden_layers = self.wav2vec_config.num_hidden_layers

    def load_ser_ckpt(self):
        path = join(hydra.utils.get_original_cwd(), self.cfg.ssl_ser_ckpt)
        ckpt = torch.load(path)
        sd = ckpt['state_dict']
        # adjust key
        sd = {'.'.join(k.split('.')[2:]): v for k, v in sd.items()}
        # check the ckpt
        for k in sd:
            if k.startswith('wav2vec2'):
                k2 = k[9:]
            else:
                continue
            assert sd[k].equal(sd[k2]), f"SER ckpt, {k} and {k2}, the same key but value of parameter are different"
        keys = list(self.wav2vec2.state_dict())
        sd = {k: v for k, v in sd.items() if k in keys}
        # load
        print(f'Load wav2vec2 parameters from SER ckpt {path}')
        self.wav2vec2.load_state_dict(sd)

    def forward(self, wav):
        out = self.wav2vec2(wav, output_hidden_states=True)
        #feat = torch.stack(out.hidden_states[-self.num_hidden_layers:]) # last hidden states
        #weight = torch.softmax(self.weight, 0)
        #feat = (feat * weight.reshape(self.num_hidden_layers, 1, 1, 1)).sum(0)
        feat = out.hidden_states[-1]
        #feat = torch.stack(out.hidden_states[-self.num_hidden_layers:]) # [layer, B, L, F]
        return feat

class Wav2vecPretrainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wav2vec2PT = load_ssl_model(self.cfg.ssl_model)
        self.wav2vec2 = self.wav2vecPT.wav2vec2

    def forward(self, x, length=None):
        with torch.no_grad():
            batch_size, sequence_length = x.size()
            sequence_length = self.get_feat_extract_output_lengths(sequence_length)
            feat_shape = (batch_size, sequence_length)
            length = self.get_feat_extract_output_lengths(length)
            attn_mask = prepare_mask(length, feat_shape, x.dtype, x.device)
            mask_time_indices = _compute_mask_indices(
                feat_shape,
                self.wav2vec2PT.config.mask_time_prob,
                self.wav2vec2PT.config.mask_time_length,
                min_masks=2,
                device=x.device,
                attention_mask=attn_mask
            )
        x = self.wav2vec2PT(x, mask_time_indices=mask_time_indices)  # , attention_mask=attn_mask)
        return x


if __name__ == "__main__":
    with torch.no_grad():
        # shape = (2, 16000 * 5)
        shape = (2, 249, 512)
        # shape = (2, 512)
        x = torch.rand(*shape)
        # x= torch.ones(*shape)
        print('in shape:', x.shape)
        batch = {'labels': torch.rand(2, 10)}
        # model = BiChain(feat_dim=x.shape[-1], cfg=DictConfig({'task': 'high', 'chain_strategy': 'ss'}))
        # model = BaselineModel(feat_dim=x.shape[-1], cfg=DictConfig({'task': 'high'}))
        model = AttnRegressor(feat_dim=x.shape[-1], cfg=DictConfig({'task': 'high'}))
        out = model(x, batch)
        print('out shape:', out.shape)  # [2, 10]
        print('out:', out)
