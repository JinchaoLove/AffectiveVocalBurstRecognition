import torch
from torch import nn
import torch.nn.functional as F

def make_mask(src):
    return (torch.sum(torch.abs(src), dim=-1) == 0).unsqueeze(1).unsqueeze(2)

def mask_mean(src, attn_mask=None):
    """src: (bsz, seq_len, feat_dim), attn_mask: (bsz, seq_len)"""
    if attn_mask is None:
        if src.ndim > 2:
            return src.mean(1)
        # no time dimension
        return src
    src[attn_mask.bool()] = 0.0
    return src.sum(dim=1) / (~attn_mask.bool()).sum(dim=1).view(-1, 1)

def slice_win(x, win_len=512, hop_len=128):
    """ Windowing slicing, will discard len < hop_len. """
    slices = []
    for i in range(len(x) // hop_len):  # if not drop_last: int(np.ceil(len(x) / hop_len))
        slices.append(x[i * hop_len: i * hop_len + win_len])
    return slices

def truncate_or_pad(X, max_len):
    """ Truncate or pad inputs.
        Note: padding_mask is [[1]*len, zeros], while attention_mask makes mask part to 0 (~padding_mask).
    """
    X = [x.squeeze(0)[:max_len] for x in X]
    padding_mask = torch.BoolTensor([[True] * len(x) + [False] * (max_len - len(x)) for x in X])
    src = torch.stack([F.pad(x, [0, max_len - len(x)]) for x in X], dim=0)
    return src.type_as(X[0]), padding_mask.type_as(X[0])

def zero_mean_unit_var_norm(input_values, padding_mask=None, padding_value=0.0):
    """ Every array in the list is normalized to have zero mean and unit variance 
        (torch version of Wav2Vec2FeatureExtractor)
        Caution: ddp should `all_gather`
    """
    if padding_mask is None:
        max_len = max([len(x) for x in input_values])
        padding_mask = torch.BoolTensor([[True] * len(x) + [False] * (max_len - len(x)) for x in input_values])
    normed_input_values = []
    for vector, length in zip(input_values, padding_mask.sum(-1)):
        normed_slice = (vector - vector[:length].mean()) / torch.sqrt(vector[:length].var() + 1e-7)
        if length < normed_slice.shape[0]:
            normed_slice[length:] = padding_value
        normed_input_values.append(normed_slice)
    return torch.stack(normed_input_values, dim=0).type_as(input_values[0])

def get_conv_output_lengths(input_lengths, conv_kernels, conv_strides):
    """Computes the output length of the convolutional layers"""
    def _conv_out_len(input_length, kernel_size, stride):
        return torch.div(input_length - kernel_size, stride, rounding_mode='floor') + 1
    for kernel_size, stride in zip(conv_kernels, conv_strides):
        input_lengths = _conv_out_len(input_lengths, kernel_size, stride)
    return input_lengths

def get_padding_mask(feature_vector_length, attention_mask, conv_kernels=None, conv_strides=None):
    output_lengths = attention_mask.cumsum(dim=-1)[:, -1].to(torch.long)
    if conv_kernels is not None and conv_strides is not None:
        output_lengths = get_conv_output_lengths(output_lengths, conv_kernels, conv_strides).to(torch.long)
    batch_size = attention_mask.shape[0]
    padding_mask = torch.zeros(
        (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
    )
    # these two operations makes sure that all values before the output lengths idxs are attended to
    padding_mask[(torch.arange(padding_mask.shape[0], device=padding_mask.device), output_lengths - 1)] = 1
    padding_mask = padding_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return padding_mask

def scaledDotProductAttention(query, key=None, value=None, mask=None, need_weights=False):
    key = query if key is None else key
    value = key if value is None else value
    score = torch.bmm(query, key.transpose(1, 2)) / query.shape[2] ** 0.5
    if mask is not None:
        score.masked_fill_(mask.view(query.shape[0], 1, key.shape[1]), -float('Inf'))
    attn_weights = F.softmax(score, -1)
    outputs = torch.bmm(attn_weights, value)
    if need_weights:
        return outputs, attn_weights
    return outputs

def attention(q, k=None, v=None, attn_mask=None, key_padding_mask=None, need_weights=False):
    """ Scaled dot product Attention. """
    k = q if k is None else k
    v = q if v is None else v
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    scores = torch.bmm(q, k.transpose(1, 2)) / q.shape[2] ** 0.5
    # merge key padding and attention masks
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.view(q.shape[0], 1, k.shape[1])
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
    # attention masks
    if attn_mask is not None:
        # convert mask to float
        if attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        scores = scores + attn_mask
    attn_weights = F.softmax(scores, dim=-1)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    outputs = torch.bmm(attn_weights, v)
    if need_weights:
        return outputs, attn_weights
    return outputs


class ChainLayer(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.c = num_classes
        self.chain = nn.ModuleList()
        for i in range(self.c):
            linear = nn.Linear(feat_dim + i, 1)
            self.chain.append(linear)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_feat):
        # input_feat: [B, feat_dim]
        out_buf = []
        for i in range(self.c):
            score = self.sigmoid(self.chain[i](input_feat))
            input_feat = torch.cat([input_feat, score], dim=-1)
            out_buf.append(score)
        out = torch.cat(out_buf, -1)
        return out


class BiChainLayer(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.c = num_classes
        self.sigmoid = nn.Sigmoid()
        self.chain, self.chain_reversed = nn.ModuleList(), nn.ModuleList()
        for i in range(self.c):
            self.chain.append(nn.Linear(feat_dim + i, 1))
            self.chain_reversed.append(nn.Linear(feat_dim + i, 1))
        self.chain_weights = nn.Parameter(torch.ones(self.c, 2) / 2.0)  # should be [0, 1]

    def forward(self, input_feat):
        # feat: [B, feat_dim]
        out_buf, out_buf_rev = [], []
        input_feat_rev = input_feat.clone()
        for i in range(self.c):
            score = self.sigmoid(self.chain[i](input_feat))
            score_rev = self.sigmoid(self.chain_reversed[i](input_feat_rev))  # gt reversed
            input_feat = torch.cat([input_feat, score], dim=-1)
            input_feat_rev = torch.cat([input_feat_rev, score_rev], dim=-1)
            out_buf.append(score)
            out_buf_rev.append(score_rev)
        # out = (torch.cat(out_buf, -1) + torch.cat(out_buf_rev, -1).flip(-1)) / 2.0
        out = torch.stack([torch.cat(out_buf, -1), torch.cat(out_buf_rev, -1).flip(-1)], dim=-1)  # (B, C, 2)
        out = (out * F.softmax(self.chain_weights, dim=-1).view(1, -1, 2)).sum(-1)  # softmax for channel dim, then sum
        return out


class CNN2D(nn.Module):
    def __init__(self, num_layers, feat_dim, **kwargs):
        """ CNN2D weighted sum. """
        super().__init__()
        # (batch, seq_len, feat_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(num_layers, 4, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # (L - k + 2 * p) / s + 1
            nn.LayerNorm(feat_dim),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, src, **kwargs):
        """ src: (bsz, num_layers, seq_len, feat_dim) """
        out = self.cnn(src).squeeze(1)
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, proj_dim=16, dropout=0.5, retrun_dict=True, **kwargs):
        super().__init__()
        self.retrun_dict = retrun_dict
        # (*, feat_dim)
        self.fc = nn.Linear(input_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.dp = nn.Dropout(dropout)

    def forward(self, src, attn_mask=None, **kwargs):
        """ src: (batch, features) """
        src = self.norm(self.fc(self.dp(src)))
        if self.retrun_dict:
            return {'src': src, 'attn_mask': attn_mask}
        return src

class CNN1d(nn.Module):
    def __init__(self, input_dim, proj_dim=16, norm='LayerNorm', mean=True, **kwargs):
        super().__init__()
        self.mean = mean
        # [bsz, feat_dim(channels), seq_len]
        conv_args = {'kernel_size': 3, 'stride': 1, 'padding': 'same'} | kwargs
        self.cnn = nn.Conv1d(input_dim, proj_dim, **conv_args)
        self.norm = getattr(nn, norm)(proj_dim)

    def forward(self, src, attn_mask=None, **kwargs):
        """ src: (batch, seq_len, features) """
        src = self.cnn(src.transpose(1, 2))
        src = F.leaky_relu(self.norm(src.transpose(1, 2)))
        if self.mean:
            src = src.sum(dim=1) / (~attn_mask.bool()).sum(dim=1).view(-1, 1)
        return src

class AverageTP(nn.Module):
    def __init__(self, input_dim, proj_dim=16, dropout=0.5, norm='LayerNorm'):
        super().__init__()
        self.proj = nn.AvgPool1d(input_dim // proj_dim)
        self.norm = getattr(nn, norm)(proj_dim)
        self.dp = nn.Dropout(dropout)

    def forward(self, src, **kwargs):
        src = self.dp(F.leaky_relu(self.norm(self.proj(src))))
        return src


class RNN(nn.Module):
    def __init__(self, input_dim=16, proj_dim=16, num_layers=1, rnn='lstm', dropout=0.5):
        super().__init__()
        self.rnn = getattr(nn, rnn.upper())(
            input_size=input_dim,
            hidden_size=proj_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

    def forward(self, src, **kwargs):
        """ src: (batch, seq_len, features) """
        out, _ = self.rnn(src)
        return out

class CRNN(nn.Module):
    def __init__(self, input_dim, proj_dim=16, num_layers=1, rnn='lstm', dropout=0.5, **kwargs):
        super().__init__()
        self.dropout = dropout
        # (batch, channels, seq_len)
        conv_args = {'kernel_size': 3, 'stride': 1, 'padding': 'same'} | kwargs
        self.cnn = nn.Conv1d(input_dim, proj_dim, **conv_args)
        # (batch, seq_len, features)
        self.rnn = getattr(nn, rnn.upper())(
            input_size=proj_dim,
            hidden_size=proj_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

    def forward(self, src, **kwargs):
        """ src: (batch, seq_len, features) """
        conv = self.cnn(src.transpose(1, 2))  # (batch, proj_dim, seq_len)
        out, _ = self.rnn(conv.transpose(1, 2))
        return out

class RCNN(nn.Module):
    def __init__(self, input_dim, proj_dim=16, num_layers=1, rnn='lstm', dropout=0.5, **kwargs):
        super().__init__()
        # (batch, seq_len, features)
        self.rnn = getattr(nn, rnn.upper())(
            input_size=input_dim,
            hidden_size=proj_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        # (batch, channels, seq_len)
        conv_args = {'kernel_size': 3, 'stride': 1, 'padding': 'same'} | kwargs
        self.cnn = nn.Conv1d(proj_dim, proj_dim, **conv_args)

    def forward(self, src, **kwargs):
        """ src: (batch, seq_len, features) """
        rnn_out, _ = self.rnn(src)
        out = self.cnn(rnn_out.transpose(1, 2))
        return out.transpose(1, 2)

class SelfAttentionPooling(nn.Module):
    """ Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim, n_heads=1):
        super(SelfAttentionPooling, self).__init__()
        self.n_heads = n_heads
        self.W = nn.Linear(input_dim, self.n_heads)
        self.softmax = F.softmax

    def forward(self, src, attn_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                src : size (N, T, H)
            attention_weight:
                att_w : size (N, T, n_heads)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(src)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))  # padding to attn mask
                attn_mask = new_attn_mask
            if attn_mask.dim() < 3:
                attn_mask = attn_mask.unsqueeze(-1).repeat(1, 1, self.n_heads)
            att_logits = attn_mask + att_logits
        att_w = self.softmax(att_logits, dim=1)
        # utter_rep = torch.sum(src * att_w, dim=1)  # original
        # (B, F, Nt) x (B, Nt, n_head) -> (B, F, n_heads)
        utter_rep = torch.bmm(src.transpose(1, 2), att_w).squeeze(-1)
        return utter_rep

class MHA(nn.Module):
    def __init__(self, input_dim, proj_dim=16, dropout=0.5, nolin=F.relu, **kwargs):
        super().__init__()
        # (batch, seq_len, feat_dim), change need_weights to output attn_weights
        self.proj = nn.Linear(input_dim, proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.nolin = nolin

    def attention(self, q, k, v, attn_mask=None, key_padding_mask=None):
        """ Scaled dot product Attention. """
        scores = torch.bmm(q, k.transpose(1, 2)) / q.shape[-1] ** 0.5
        if attn_mask is None:
            attn_mask = key_padding_mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            if key_padding_mask is not None:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
            scores = scores + attn_mask
        attn_weight = F.softmax(scores, dim=-1)
        outputs = torch.bmm(self.dropout(attn_weight), v)
        return outputs, attn_weight

    def forward(self, src, **kwargs):
        """ src: (batch, seq_len, feat_dim) """
        bsz, seq_len, feat_dim = src.shape
        src = self.nolin(self.proj(src))
        out, _ = self.attention(src, src, src)
        return out

class MHA2(nn.Module):
    def __init__(self, input_dim, proj_dim=16, num_heads=16, dropout=0.5, nolin=F.relu, **kwargs):
        super().__init__()
        # (N, S, E) or stacked (N*num_heads, S, E/num_heads), key_padding_mask: (N, S)
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            input_dim // self.num_heads, self.num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(input_dim, proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.nolin = nolin

    def split_heads(self, x):
        """ Split feat_dim into num_heads. """
        bsz, seq_len, feat_dim = x.shape  # batch first
        x = x.transpose(0, 1)
        x = x.contiguous().view(seq_len, bsz * self.num_heads, feat_dim // self.num_heads)
        return x.transpose(0, 1)

    def forward(self, src, attn_mask=None, **kwargs):
        """ output_stage: 2
        src: (bsz, seq_len, feat_dim), attn_mask: (bsz, seq_len)
        """
        bsz, seq_len, feat_dim = src.shape
        src = self.split_heads(src)
        # attn_mask = attn_mask.reshape(bsz, -1).repeat_interleave(repeats=self.num_heads, dim=0)
        # out, _ = self.attn(q, k, k, attn_mask=attn_mask)
        out, _ = self.attn(src, src, src)
        out = out.contiguous().view(bsz, seq_len, feat_dim)
        out = out.mean(1)  # self.statistics(out, attn_mask)
        out = self.nolin(self.proj(out))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000, **kwargs):
        """ Positional encoding. """
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))  # Create a long enough `P`
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class MHA3(nn.Module):
    """ Multihead Attention with positional encoding.
        https://zh-v2.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html
    """

    def __init__(self, input_dim, proj_dim=16, num_heads=16, dropout=0.5, nolin=F.relu, mean=True, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.mean = mean
        self.pos_encoding = PositionalEncoding(input_dim, dropout)
        self.attn = nn.MultiheadAttention(
            input_dim // self.num_heads, self.num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(input_dim, proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.nolin = nolin

    def split_heads(self, x, batch_first=True):
        """ Split feat_dim into num_heads. """
        bsz, seq_len, feat_dim = x.shape  # batch first
        x = x.transpose(0, 1)
        x = x.contiguous().view(seq_len, bsz * self.num_heads, feat_dim // self.num_heads)
        return x.transpose(0, 1)

    def forward(self, src, attn_mask=None, **kwargs):
        """ output_stage: 2
        src: (bsz, seq_len, feat_dim), attn_mask: (bsz, seq_len)
        stacked to (bsz*num_heads, seq_len, feat_dim/num_heads)
        """
        bsz, seq_len, feat_dim = src.shape
        src = self.pos_encoding(src * seq_len ** 0.5)
        src = self.split_heads(src)
        # attn_mask = attn_mask.reshape(bsz, -1).repeat_interleave(repeats=self.num_heads, dim=0)
        # out, _ = self.attn(q, k, k, attn_mask=attn_mask)
        out, _ = self.attn(src, src, src)
        out = out.contiguous().view(bsz, seq_len, feat_dim)
        out = self.nolin(self.proj(out))
        if self.mean:
            out = out.mean(1)
        return out


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class BiLSTMAttention(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers=1, num_directions=2):
        super().__init__()

        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        # lstm的输入维度为 [seq_len, batch, input_size]
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=(num_directions == 2))
        self.tanh = nn.Tanh()
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        # self.fc = nn.Linear(hidden_size, num_classes)
        # self.act_func = nn.Softmax(dim=1)

    def forward(self, x, attn_mask=None):
        # x [batch_size, sentence_length, embedding_size]
        x = x.permute(1, 0, 2)  # [sentence_length, batch_size, embedding_size]

        #由于数据集不一定是预先设置的batch_size的整数倍，所以用size(1)获取当前数据实际的batch
        batch_size = x.size(1)

        #设置lstm最初的前项输出
        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).type_as(x)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).type_as(x)

        # out: [seq_len, batch, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        # h_n, c_n: [num_layers * num_directions, batch, hidden_size]
        out, (h_n, _) = self.lstm(x, (h_0, c_0))
        out = self.tanh(out)

        #将双向lstm的输出拆分为前向输出和后向输出
        (forward_out, backward_out) = torch.chunk(out, 2, dim=2)
        out = forward_out + backward_out  # [seq_len, batch, hidden_size]
        out = out.permute(1, 0, 2)  # [batch, seq_len, hidden_size]

        #为了使用到lstm最后一个时间步时，每层lstm的表达，用h_n生成attention的权重
        h_n = h_n.permute(1, 0, 2)  # [batch, num_layers * num_directions, hidden_size]
        h_n = torch.sum(h_n, dim=1).squeeze(dim=1)  # [batch, hidden_size]

        attention_w = self.attention_weights_layer(h_n)  # [batch, hidden_size]

        attention_context = torch.bmm(attention_w.unsqueeze(dim=1), out.transpose(1, 2))  # [batch, 1, seq_len]
        if attn_mask is not None:  # added by jcli
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))  # padding to attn mask
                attn_mask = new_attn_mask
            attention_context = attention_context + attn_mask
        softmax_w = F.softmax(attention_context, dim=-1)  # [batch, 1, seq_len],权重归一化

        x = torch.bmm(softmax_w, out).squeeze(dim=1)  # [batch, hidden_size]
        # x = self.fc(x)
        # x = self.act_func(x)
        return x


if __name__ == "__main__":
    with torch.no_grad():
        # src, mask = torch.randn(512, 14, 16), torch.randn(512, 14).bool()
        # out = scaledDotProductAttention(query=src, mask=mask)
        # print(out.shape)
        # seq_len, batch, input_size
        src = torch.randn(2, 249, 1024)
        model = BiLSTMAttention(1024, 16, 10)
        print(model(src).shape)  # 2, 10
