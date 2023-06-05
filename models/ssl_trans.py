import os
import sys
import re
from omegaconf import DictConfig
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed import is_initialized
import torchaudio.transforms as T
import transformers as ppb
try:
    from models import basics, leaf
except:
    import sys
    sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
    from models import basics, leaf
from models.conformer import Conformer

class Upstream(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.mode
        self.weighted_sum = cfg.weighted_sum
        self.mask = 1
        self.output_stage = cfg.output_stage  # 0: timepooling,  >0: with time info
        self.std = int(self.output_stage == 0)
        self.feat_type = cfg.feat_type
        self.sr = cfg.sr
        self.max_len = int(cfg.max_wav_length * self.sr)

        # raw features
        if self.feat_type is not None:
            self.get_feature(cfg.proj_dim)
            return
        # ignore below if feat_type is not None
        self.name_path = cfg.name_path
        self.load_name_path(self.name_path)
        if self.mode in ['frozen', 'f']:
            self.freeze(False)
        elif self.mode in ['partial_finetune', 'pf']:
            # (high->low): 22, 21, 24, 16, 15, 23, 9, 20, 1, 14, 4, 12, 17, 19, 6, 7, 8, 5, 11, 3, 10, 13, 2, 18, 0
            # (0-24): 0.0008 0.0519 0.0056 0.0065 0.0349 0.0108 0.0194 0.0146 0.0131 0.0550 0.0064 0.0106 0.0241 0.0057 0.0428 0.0621 0.0734 0.0235 0.0021 0.0207 0.0542 0.1226 0.2001 0.0606 0.0785
            # culture: 0.0000 0.0000 0.0001 0.0003 0.0003 0.0001 0.0008 0.0070 0.0037 0.0125 0.9446 0.0257 0.0031 0.0006 0.0004 0.0002 0.0001 0.0001 0.0002 0.0000 0.0000 0.0000 0.0000 0.0001 0.0001
            # feature_extractor, feature_projection, [attention|layer_norm|feed_forward|final_layer_norm]
            self.partial_freeze(['feature_projection'], training=True)
            # self.partial_freeze(['feature_extractor', 'embeddings'], training=False)
        elif self.mode == 'pf9':
            self.partial_freeze(['feature_projection', 'encoder.layers.9.*'], training=True)
        elif self.mode in ['attn']:
            self.partial_freeze(['attention'], training=True)  # only finetune attention part
        if self.weighted_sum:
            if int(self.weighted_sum) == 2:
                self.layer_weights = basics.CNN2D(self.num_layers, self.hidden_size)
            else:
                self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)

    def get_feature(self, hidden_size=None):
        self.hidden_size = 128 if hidden_size is None else hidden_size
        if any(sub in self.feat_type.lower() for sub in ['fbank', 'mel']):  # fbanks
            self.model = nn.Sequential(
                basics.PreEmphasis(),
                T.MelSpectrogram(sample_rate=self.sr, n_mels=self.hidden_size, normalized=True, power=2),
                T.AmplitudeToDB(stype='power'),
            )
            return
        if any(sub in self.feat_type.lower() for sub in ['leaf']):
            self.model = leaf.Leaf(sample_rate=self.sr, n_filters=self.hidden_size)
            return
        if any(sub in self.feat_type.lower() for sub in ['pcen']):
            self.pcen = leaf.PCENLayer(trainable=True, learn_smooth_coef=True)
            self.pcen.build(num_channels=self.hidden_size)
            self.model = nn.Sequential(
                basics.PreEmphasis(),
                T.MelSpectrogram(sample_rate=self.sr, n_mels=self.hidden_size, normalized=True, power=2),
                self.pcen
            )
            return
        raise ValueError(f"Feature type {self.feat_type} not included in `Upstream`.")

    def load_name_path(self, name_path):
        config = ppb.AutoConfig.from_pretrained(name_path).to_dict()
        # for k in config.keys():
        #     if 'drop' in k or 'prob' in k:
        #         config[k] = 0  # no dropout for regression
        config = config | {'torch_dtype': 'auto', 'layerdrop': 0}
        # config = config | {'local_files_only': True}  # uncomment to avoid SSL connection error
        # config = config | {'force_download': True}  # uncomment to (re-)download models
        self.model = ppb.AutoModel.from_pretrained(name_path, **config)
        self.model.gradient_checkpointing_enable()
        # self.model.freeze_feature_encoder()
        # do not use Wav2vec2FeatureExtractor (very slow!)
        self.num_layers = self.model.config.num_hidden_layers + 1  # layers + input embeddings
        self.hidden_size = self.model.config.hidden_size
        return self

    @staticmethod
    def match(name, group):
        name = name.lower()
        for sub in group:
            if '*' in sub or '|' in sub:
                sub = sub.replace('?', '.').replace('*', '.*')
                if re.search(sub, name) is not None:
                    return True
            elif sub in name:
                return True
        return False

    def partial_freeze(self, group=['attention'], excludes=[], training=True):
        """ Freeze or unfreeze parameters in `group` exclude `excludes`. """
        for name, param in self.model.named_parameters():
            param.requires_grad = not training
            if self.match(name, group):
                param.requires_grad = training
            if self.match(name, excludes):
                param.requires_grad = not training

    def freeze(self, training=False):
        for param in self.model.parameters():
            param.requires_grad = training

    def preprocess(self, X):
        """ torch version of Wav2Vec2FeatureExtractor """
        max_len = min(max([x.shape[-1] for x in X]), self.max_len)
        X, padding_mask = basics.truncate_or_pad(X, max_len=max_len)
        assert X.ndim == 2 and X.shape[-1] > 1600, f"Shape error: {X.shape}"
        # X = basics.zero_mean_unit_var_norm(X, padding_mask)
        X = F.layer_norm(X, X.shape[1:])  # standardized
        return {'input_values': X, 'attention_mask': padding_mask}

    def weighted_sum_layers(self, hidden_states):
        # hidden_states: (N, num_layers, seq_len, feat_dim) -> (N, seq_len, feat_dim)
        hidden_states = torch.stack(hidden_states, dim=1)
        if int(self.weighted_sum) > 1:
            hidden_states = self.layer_weights(hidden_states)
        else:
            norm_weights = F.gumbel_softmax(self.layer_weights, dim=-1)
            hidden_states = hidden_states * norm_weights.view(-1, 1, 1)
        if hidden_states.ndim == 4:
            return hidden_states.sum(dim=1)
        return hidden_states

    def statistics(self, src, padding_mask, dim=1):
        pooled_mean = src.sum(dim=dim) / padding_mask.sum(dim=dim).view(-1, 1)
        pooled_std = torch.sqrt((
            padding_mask.unsqueeze(-1) * (src - pooled_mean.unsqueeze(dim))**2
        ).sum(dim=dim) / padding_mask.sum(dim=dim).view(-1, 1) + 1e-10)
        pooled_output = torch.cat((pooled_mean, pooled_std), dim=dim)
        return pooled_output

    def post_process(self, outputs, X):
        if self.weighted_sum:
            hidden_states = self.weighted_sum_layers(outputs['hidden_states'])
        else:
            hidden_states = outputs['last_hidden_state']  # last_hidden_state: (B, T, C)
            # hidden_states = torch.stack(outputs['hidden_states'], dim=0)[17]  # !!! only layer 17 used
        padding_mask = basics.get_padding_mask(
            hidden_states.shape[1], X['attention_mask'],
            self.model.config.conv_kernel, self.model.config.conv_stride)
        hidden_states[~padding_mask] = 0.0
        if self.output_stage > 0:  # weighted_sum or last_hidden_state
            return {'src': hidden_states, 'attn_mask': ~padding_mask}
        pooled_output = self.statistics(hidden_states, padding_mask)
        return {'src': pooled_output, 'attn_mask': None}

    def forward_feat_type(self, X):
        src = self.model(X['input_values']).transpose(1, 2)
        padding_mask = basics.get_padding_mask(src.shape[1], X['attention_mask'], [1], [160])  # win 401, pad 200 * 2, hop 160
        if self.output_stage > 0:
            return {'src': src, 'attn_mask': ~padding_mask}
        pooled_output = self.statistics(X, padding_mask)
        return {'src': pooled_output, 'attn_mask': None}

    def forward(self, X, **kwargs):
        X = self.preprocess(X)
        if self.feat_type is not None:
            return self.forward_feat_type(X)
        outputs = self.model(**X, output_hidden_states=True, return_dict=True)
        outputs = self.post_process(outputs, X)
        return outputs


class Featurizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Upstream(cfg)
        std = 2 if self.model.std else 1
        self.fc = nn.Linear(self.model.hidden_size * std, cfg.proj_dim)
        self.norm = nn.LayerNorm(cfg.proj_dim)
        # self.norm = nn.SyncBatchNorm(proj_dim) if is_initialized() else nn.BatchNorm1d(proj_dim)
        self.dp = nn.Dropout(cfg.dropout)

    def forward(self, X):
        # input: [B, T, F]
        X = self.model(X)
        src = self.norm(self.fc(self.dp(X['src'])))
        return {'src': src, 'attn_mask': X['attn_mask']}


class BiFeaturizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Upstream(cfg)
        std = 2 if self.model.std else 1
        self.bilstm = basics.BiLSTMAttention(self.model.hidden_size * std, hidden_size=cfg.proj_dim)
        self.norm = nn.SyncBatchNorm(cfg.proj_dim) if is_initialized() else nn.BatchNorm1d(cfg.proj_dim)
        self.dp = nn.Dropout(cfg.dropout)

    def forward(self, X):
        # input: [B, T, F]
        X = self.model(X)
        src = self.bilstm(**X)  # B, hidden_size
        X = self.norm(X)
        return {'src': src, 'attn_mask': None}


class SpecFeature(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Upstream(cfg)
        self.hidden_size = cfg.proj_dim
        std = 2 if self.model.std else 1
        self.conformer = Conformer(
            input_dim=self.model.hidden_size * std, num_heads=cfg.num_heads, ffn_dim=cfg.proj_dim, num_layers=cfg.num_layers,
            dropout=cfg.dropout, depthwise_conv_kernel_size=cfg.depthwise_conv_kernel_size,)

    def forward(self, X):
        # input: [B, T, F]
        X = self.model(X)
        src, _ = self.conformer(X['src'], (~X['attn_mask']).cumsum(dim=-1)[:, -1].to(torch.long))
        return {'src': src, 'attn_mask': X['attn_mask']}


class MTL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = {'high': 10, 'two': 2, 'culture': 40, 'type': 8}[cfg.task]
        cfg.output_stage = 1
        self.model = eval(cfg.upstream)(cfg)
        self.dp = nn.Dropout(cfg.dropout)
        self.pool = basics.SelfAttentionPooling(self.model.hidden_size)
        self.norm = nn.SyncBatchNorm(self.model.hidden_size) if is_initialized() else nn.BatchNorm1d(self.model.hidden_size)
        self.proj = nn.Linear(self.model.hidden_size, cfg.proj_dim)
        self.share = nn.Linear(cfg.proj_dim, cfg.shared_dim)

        self.val = nn.Sequential(nn.Linear(cfg.proj_dim + cfg.shared_dim, 2), nn.Sigmoid())  # CCC

        # country, voc
        self.cnt = nn.Linear(cfg.shared_dim + 2, 4)  # CE
        self.voc = nn.Linear(cfg.shared_dim + 2, 8)  # CE
        # self.emo = nn.Sequential(nn.Linear(cfg.shared_dim + 2 + 4 + 8, 10), nn.Sigmoid())  # CCC
        self.emo = basics.BiChainLayer(cfg.shared_dim + 2 + 4 + 8, 10)
        if self.cfg.task == 'culture':
            # self.emo2 = nn.Sequential(nn.Linear(cfg.shared_dim + 2 + 4 + 8 + 10, self.num_classes), nn.Sigmoid())  # CCC
            self.emo2 = basics.BiChainLayer(cfg.shared_dim + 2 + 4 + 8 + 10, self.num_classes)


    def forward(self, batch):
        # input: [B, T, F]
        feat = self.model(batch['wav'])
        feat = self.pool(**feat)
        feat = self.dp(self.norm(feat))

        proj = F.gelu(self.proj(feat))
        shared = self.share(proj)

        va = self.val(torch.cat([shared, proj], dim=-1))  # output: 2
        val_shared = torch.cat([shared, va], dim=-1)

        cnt = self.cnt(val_shared)  # country output
        voc = self.voc(val_shared)  # voc type output
        emo = self.emo(torch.cat([shared, va, cnt, voc], dim=-1))  # emo output
        if self.cfg.task == 'culture':
            emo2 = self.emo2(torch.cat([shared, va, cnt, voc, emo], dim=-1))  # emo output
            return {'preds': emo2, 'va': va, 'country': cnt, 'voc': voc, 'emo': emo}
        return {'preds': emo, 'va': va, 'country': cnt, 'voc': voc}


if __name__ == "__main__":
    with torch.no_grad():
        shape = (2, 16000 * 5)
        x = {'wav': torch.rand(*shape)}
        cfg = DictConfig({
            'proj_dim': 512,
            'shared_dim': 128,
            'tasks': 'country, voc, valence, arousal',
            'name_path': 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition',
            'upstream': 'Upstream',
            'feat_type': None,
            'output_stage': 1,
            'weighted_sum': 1,
            'dropout': 0.25,
            'mode': 'pf',
            'sr': 16000,
            'max_wav_length': 5,
            'task': 'high'})
        model = MTL(cfg)
        out = model(x)
        print({k: v.shape for k, v in out.items()})  # [2, 249, 64]
        print(out['preds'])
