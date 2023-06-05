import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from models.basics import *

class Linear(nn.Module):
    def __init__(self, input_dim=16, num_labels=2, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, src, src_mask=None, **kwargs):
        if src.ndim == 3:
            # src: (bsz, seq_len, input_dim)
            if src_mask is not None:
                return self.fc(mask_mean(src, src_mask))
            return self.fc(src.mean(1) if src.ndim > 2 else src)
        return self.fc(src)

class SelfAttn(nn.Module):
    def __init__(self, input_dim=16, num_labels=2, **kwargs):
        super().__init__()
        self.attn = nn.MultiheadAttention(input_dim, 2, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, src, src_mask=None, **kwargs):
        # src: (bsz, seq_len, input_dim)
        if src_mask is not None:
            out, _ = self.attn(src, src, src, ~src_mask.bool())
        else:
            out, _ = self.attn(src, src, src)
        if src.ndim == 3:
            # src: (bsz, seq_len, input_dim)
            if src_mask is not None:
                return self.fc(mask_mean(out, src_mask))
            return self.fc(out.mean(1))
        return self.fc(out)

class TEL(nn.Module):
    def __init__(self, input_dim=16, num_labels=2, dropout=0.5, **kwargs):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(input_dim, 2, 2, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, src, src_mask=None, **kwargs):
        # src: (bsz, seq_len, input_dim)
        if src_mask is not None:
            out = self.attn(src, src_key_padding_mask=~src_mask.bool())
        else:
            out = self.attn(src)
        if src.ndim == 3:
            # src: (bsz, seq_len, input_dim)
            if src_mask is not None:
                return self.fc(mask_mean(out, src_mask))
            return self.fc(out.mean(1))
        return self.fc(out)

class ContextConformer(nn.Module):
    def __init__(self, input_dim=16, num_labels=2, dropout=0.5, **kwargs):
        super().__init__()
        self.attn = Attention(dim=input_dim, dim_head=4, heads=4, dropout=dropout)
        self.conv = ConformerConvModule(
            dim=input_dim, causal=False, expansion_factor=4 / input_dim, kernel_size=3, dropout=dropout)
        self.fc = nn.Linear(input_dim, num_labels)
        self.dp = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, **kwargs):
        out = self.attn(src, mask=src_mask) + src
        out = self.conv(out) + out
        out = self.dp(out.mean(1))
        out = self.fc(out.view(out.size(0), -1))
        return out

class ConvConformerRNN(nn.Module):
    def __init__(self, input_dim=16, num_labels=2, dropout=0.5,
                 rnn='lstm', hid_dim=16, num_layers=1, context_win=4, **kwargs):
        super().__init__()
        self.attn = Attention(dim=input_dim, dim_head=4, heads=4, dropout=dropout)
        self.conv = ConformerConvModule(
            dim=input_dim, causal=False, expansion_factor=4 / input_dim, kernel_size=3, dropout=dropout)
        self.rnn = getattr(nn, rnn.upper())(
            input_size=input_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Linear(hid_dim * (2 * int(context_win) + 1), num_labels)

    def forward(self, src, src_mask=None, **kwargs):
        out = self.attn(src, mask=src_mask) + src
        out = self.conv(out) + out
        # out = self.dp(out.mean(1))
        out, _ = self.rnn(out)
        out = self.fc(out.reshape(out.size(0), -1))
        return out

class ContextConformerFull(nn.Module):
    def __init__(self, input_dim=16, num_labels=2, dropout=0.5, **kwargs):
        super().__init__()
        self.conv = ConformerBlock(
            dim=input_dim, dim_head=4, heads=4, ff_mult=2,
            expansion_factor=2, kernel_size=3, dropout=dropout)
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, src, src_mask=None, **kwargs):
        out = self.conv(src, mask=src_mask)
        out = self.fc(mask_mean(out, src_mask))
        return out

class ContextConformerRNN(nn.Module):
    def __init__(self, input_dim=16, num_labels=2, dropout=0.5,
                 rnn='lstm', hid_dim=4, num_layers=1, **kwargs):
        super().__init__()
        self.conv = ConformerBlock(
            dim=input_dim, dim_head=4, heads=4, ff_mult=4 / input_dim,
            expansion_factor=4 / input_dim, kernel_size=3, dropout=dropout)
        self.rnn = getattr(nn, rnn.upper())(
            input_size=input_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Linear(hid_dim, num_labels)
        self.dp = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, **kwargs):
        out = self.conv(src, mask=src_mask)  # (N, seq_len, feat_dim)
        out, _ = self.rnn(out)  # (N, seq_len, hidden_dim)
        out = self.dp(out.mean(1))
        out = self.fc(out.reshape(out.size(0), -1))
        return out

class TDNN_RNN(nn.Module):
    def __init__(self, input_dim=16, num_labels=2, context=2, context_win=4, dropout=0.25,
                 rnn='lstm', hid_dim=8, num_layers=1, **kwargs):
        super().__init__()
        self.tdnn = TDNN(context=[-context, context], input_dim=input_dim, output_dim=hid_dim)
        self.rnn = getattr(nn, rnn.upper())(
            input_size=hid_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Linear(hid_dim, num_labels)

    def forward(self, src, src_mask=None, **kwargs):
        """ x: [bsz, seq_len, input_dim] """
        valid_range = self.tdnn.get_valid_steps(self.tdnn.context, src_mask.shape[-1])
        src_mask = src_mask[:, valid_range]
        out = self.tdnn(src)
        out, _ = self.rnn(out)
        out = self.fc(mask_mean(out, src_mask))
        return out

class TDNN_mean(nn.Module):
    def __init__(self, input_dim=16, num_labels=2, context=2, dropout=0.5, **kwargs):
        super().__init__()
        self.tdnn = nn.Sequential(
            TDNN(context=[-context, context], input_dim=input_dim, output_dim=input_dim),
            TDNN(context=[-context, context], input_dim=input_dim, output_dim=input_dim),
            TDNN(context=[-context, context], input_dim=input_dim, output_dim=input_dim),
        )
        self.fc = nn.Linear(input_dim, num_labels)
        self.dp = nn.Dropout(dropout)

    def forward(self, src, **kwargs):
        """ x: [bs, seq_len, input_dim] """
        out = self.tdnn(src)
        out = self.fc(self.dp(out.mean(1)))
        return out

class ContextTDNN(nn.Module):
    def __init__(self, input_dim=16, num_labels=2, hid_dim=4, context_win=4, dropout=0.25, **kwargs):
        super().__init__()
        self.tdnn = TDNN(context=[-2, 2], input_dim=input_dim, output_dim=hid_dim)
        out_seq_len = len(self.tdnn.get_valid_steps(self.tdnn.context, 2 * context_win + 1))
        self.fc = nn.Linear(out_seq_len * hid_dim, num_labels)
        self.dp = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, **kwargs):
        """ x: [bs, seq_len, input_dim] """
        out = self.dp(self.tdnn(src))
        out = self.fc(out.view(out.size(0), -1))
        return out

class CRNN1(nn.Module):
    def __init__(self, input_dim=16, num_labels=2, hid_dim=16, num_layers=1, context_win=4, rnn='lstm', dropout=0.5, **kwargs):
        super().__init__()
        conv_args = {'kernel_size': 3, 'stride': 1, 'padding': 'same'}
        # (bsz, feat_dim, seq_len)
        self.cnn = nn.Sequential(
            Rearrange('b c n -> b n c'),
            nn.Conv1d(input_dim, input_dim, **conv_args),
            nn.LeakyReLU(),
            Rearrange('b n c -> b c n'),
            nn.LayerNorm(input_dim),

            Rearrange('b c n -> b n c'),
            nn.Conv1d(input_dim, input_dim, **conv_args),
            nn.LeakyReLU(),
            Rearrange('b n c -> b c n'),
            nn.LayerNorm(input_dim),
        )
        # (bsz, seq_len, feat_dim)
        self.rnn = getattr(nn, rnn.upper())(
            input_size=input_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Linear(hid_dim, num_labels)
        self.dp = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, **kwargs):
        """ src: (bsz, seq_len, feat_dim) """
        bsz, seq_len, feat_dim = src.shape
        out = self.cnn(src)
        out, _ = self.rnn(out)
        out = self.fc(self.dp(mask_mean(out, src_mask)))
        return out


class CRNN2(nn.Module):
    def __init__(self, input_dim=16, num_labels=2, num_layers=1, context_win=4, rnn='lstm', dropout=0.5, **kwargs):
        super().__init__()
        conv_args = {'kernel_size': 3, 'stride': 1, 'padding': 'valid'}
        # (bsz, channels, seq_len, feat_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 2, **conv_args),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(2, 4, **conv_args),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(4, 1, **conv_args),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, 1),
        )
        # (bsz, seq_len, feat_dim)
        self.rnn = getattr(nn, rnn.upper())(
            input_size=input_dim - 2 * 3 - 2 * 3,
            hidden_size=input_dim - 2 * 3 - 2 * 3,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.mp = nn.MaxPool1d(3, 1)
        self.fc = nn.Linear(input_dim - 2 * 3 - 2 * 4, num_labels)
        self.dp = nn.Dropout(dropout)

    def forward(self, src, **kwargs):
        """ src: (bsz, seq_len, feat_dim) """
        bsz, seq_len, feat_dim = src.shape
        out = self.cnn(src.unsqueeze(1)).squeeze(1)
        out, _ = self.rnn(out)
        out = self.mp(out)
        out = self.fc(self.dp(out.mean(1)))
        return out

class ContextClassifier(nn.Module):
    def __init__(self, input_dim=768 * 2, num_labels=2, hid_dim=64, att_dim=64, num_heads=5, **kwargs):
        super().__init__()
        self.layer_bert = nn.Linear(input_dim, hid_dim)
        self.bert_atten = SelfAttentiveLayer(hid_dim, att_dim, num_heads)
        self.fc = nn.Linear(att_dim * num_heads, num_labels)

    def forward(self, X, att_mask=None, **kwargs):
        # X: 16, 7, 768 * 2
        bert = []
        for i in range(X.shape[1]):
            bert.append(F.leaky_relu(self.layer_bert(X[:, i, :])).unsqueeze(1))
        bert_all = torch.cat((bert), 1)  # 16, 7, 64
        out, A = self.bert_atten(bert_all)  # 16, 5, 64
        out = out.view(out.size(0), -1).contiguous()  # 16, 320
        logits = self.fc(out)
        return logits


class CNNAtt1(nn.Module):
    def __init__(
        self,
        input_dim,
        num_labels=2,
        project_dim=256,
        hidden_dim=16,
        kernel_size=5,
        pooling=5,
        padding=2,
        dropout=0.4,
        mdevice='cuda',
        **kwargs
    ):
        super().__init__(**kwargs)
        # self.projector = nn.Linear(input_dim, project_dim)
        # self.pooling = pooling
        # self.selector = nn.AvgPool1d(pooling)
        self.model_seq = nn.Sequential(
            # nn.Dropout(p=dropout),
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding),
            nn.LeakyReLU(),
        )
        self.attpooling = SelfAttentionPooling(hidden_dim)
        self.out_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, features, att_mask=None):
        # features = self.projector(features)
        # features = self.selector(features)
        if features.ndim == 2:
            features = features.unsqueeze(1)
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        out = self.attpooling(out, att_mask).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted


class CNNAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        num_labels=2,
        project_dim=256,
        hidden_dim=80,
        kernel_size=5,
        pooling=5,
        padding=2,
        dropout=0.4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.projector = nn.Linear(input_dim, project_dim)
        self.pooling = pooling
        self.selector = nn.AvgPool1d(pooling)
        self.model_seq = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv1d(project_dim // pooling, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
        )
        self.attpooling = SelfAttentionPooling(hidden_dim)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, features, att_mask=None):
        features = self.projector(features)
        features = self.selector(features)
        if features.ndim == 2:
            features = features.unsqueeze(1)
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        out = self.attpooling(out, att_mask).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted
