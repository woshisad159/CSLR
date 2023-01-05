# ----------------------------------------
# Written by Jing Li
# ----------------------------------------
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
import copy

def unpad_padded(x, xl, dim=0):
    dims = list(range(len(x.shape)))
    dims.insert(0, dims.pop(dim))
    x = x.permute(*dims)
    return [xi[:xli] for xi, xli in zip(x, xl)]

def key_padding_mask(l):
    """Blank is True
    Args:
        l: lenghts (b)
    Returns:
        mask: (b l)
    """
    mask = torch.zeros(len(l), max(l)).bool()
    for i, li in enumerate(l):
        mask[i, li:] = True
    return mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, rpe_q=None, rpe_v=None):
        """
        Args:
            q: query (*, query_len, dim)
            k: key (*, key_len, dim)
            v: value (*, key_len, dim)
            mask: (*, query_len, key_len), True will be masked out
            rpe_q : (query_len, key_len, dim)
            rpe_v : (query_len, key_len, dim)
        Returns:
            context: (*, query_len, dim)
            alignment: (*, query_len, key_len)
        """
        dim = q.shape[-1]

        q /= dim ** 0.5
        energy = q @ k.transpose(-2, -1)

        if rpe_q is not None:
            energy += torch.einsum("...qd,qkd->...qk", q, rpe_q)

        if mask is not None:
            energy = energy.masked_fill(mask, np.NINF)

        alignment = torch.softmax(energy, dim=-1)
        context = self.dropout(alignment) @ v

        if rpe_v is not None:
            context += torch.einsum("...qk,qkd->...qd", alignment, rpe_v)

        return context, alignment

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout, rpe_k=0):
        assert (
            dim % heads == 0
        ), "dim should be a multiple of heads, \
            got {} and {}".format(
            dim, heads
        )

        super().__init__()

        self.dim = dim
        self.heads = heads

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)

        self.rpe_k = rpe_k
        if rpe_k > 0:
            self.rpe_w = nn.Embedding(rpe_k * 2 + 1, 2 * dim // heads)

        self.attn = ScaledDotProductAttention(dropout)
        self.fc = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: query (batch, query_len, dim)
            k: key (batch, key_len, dim)
            v: value (batch, key_len, dim)
            mask: (batch, query_len, key_len)
        Returns:
            context: (batch, query_len, dim)
            alignment: (bs, head, ql, kl)
        """

        bs, ql, kl = (*q.shape[:2], k.shape[1])

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        split_heads = lambda x: rearrange(x, "b t (h d) -> b h t d", h=self.heads)
        q, k, v = map(split_heads, (q, k, v))

        # add head dim for mask
        if mask is not None:
            mask = mask.unsqueeze(1)

        if self.rpe_k > 0:
            distance = self.relative_distance(max(ql, kl), self.rpe_k)
            distance = distance[:ql, :kl].to(q.device)
            rpe_q, rpe_v = self.rpe_w(distance).chunk(2, dim=-1)
            context, alignment = self.attn(q, k, v, mask, rpe_q, rpe_v)
        else:
            context, alignment = self.attn(q, k, v, mask)

        # swap len and head back
        context = rearrange(context, "b h t d -> b t (h d)")
        context = self.fc(context)

        return context, alignment

    @staticmethod
    def relative_distance(length, k):
        indices = torch.arange(length)
        indices = indices.unsqueeze(1).expand(-1, length)
        distance = indices - indices.transpose(0, 1)
        distance = distance.clamp(-k, k) + k
        return distance

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, ffn_dim, dropout):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(torch.relu(self.w1(x))))

class PreNorm(nn.Module):
    def __init__(self, dim, model):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.model = model

    def forward(self, x):
        return self.model(self.norm(x))

class Residual(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)

    def forward(self, x):
        return super().forward(x) + x

class Applier(nn.Module):
    def __init__(self, model, applier):
        super().__init__()
        self.model = model
        self.applier = applier

    def forward(self, x):
        return self.applier(self.model, x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dropout=0.1, rpe_k=0):
        super().__init__()
        attn = MultiHeadAttention(dim, heads, dropout, rpe_k)
        ffn = PositionwiseFeedForward(dim, 4 * dim, dropout)
        wrap = lambda m: Residual(PreNorm(dim, m), nn.Dropout(dropout))
        self.attn = wrap(Applier(attn, lambda m, x: m(x, x, x, self.xm)[0]))
        self.ffn = wrap(ffn)

    def forward(self, x, xm):
        # hack the mask here
        self.xm = xm
        x = self.attn(x)
        del self.xm
        x = self.ffn(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads, num_layers, dropout=0.1, rpe_k=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)
        for i in range(num_layers):
            self.layers += [
                TransformerEncoderLayer(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    rpe_k=rpe_k,
                )
            ]

    def forward(self, x):
        """
        Args:
            x: [(t d)]
        Returns:
            x: [(t d)]
        """
        xl = list(map(len, x))
        xm = key_padding_mask(xl).to(x.device)
        xm = xm.unsqueeze(dim=1)  # repeat mask for all targets
        for layer in self.layers:
            x = layer(x, xm)
        x = self.norm(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = [(i//2) for i in feat_len]
            else:
                feat_len = [(i - int(ks[1]) + 1) for i in feat_len]
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        return {
            "visual_feat": visual_feat,
            "feat_len": lgt,
        }

class TemporalUpSampleBlock(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, resFlag=False, scale=1):
        super(TemporalUpSampleBlock, self).__init__()
        self.resFlag = resFlag

        if resFlag:
            self.conv1D = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size, stride=1,
                                 padding=1, groups=hidden_size)
        else:
            if kernel_size == 1:
                padding = 0
            elif kernel_size == 3:
                padding = 1

            self.conv1D = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size, stride=scale,
                                    padding=padding)

        self.batchNorm1d = nn.BatchNorm1d(hidden_size)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputData):
        inputData1 = inputData

        inputData = self.conv1D(inputData)
        inputData = self.batchNorm1d(inputData)

        if self.resFlag:
            inputData = inputData1 + inputData

        inputData = self.relu(inputData)

        return inputData

class TemporalUpSample(nn.Module):
    def __init__(self, input_size, scale):
        super(TemporalUpSample, self).__init__()

        self.scale = scale

        hidden_size1 = input_size * 2
        hidden_size3 = input_size * 4

        self.temporalUpSampleBlock1 = TemporalUpSampleBlock(input_size, hidden_size1)
        self.temporalUpSampleBlock2 = TemporalUpSampleBlock(hidden_size1, hidden_size1, resFlag=True)
        self.temporalUpSampleBlock3 = TemporalUpSampleBlock(hidden_size1, hidden_size1, resFlag=True)
        self.temporalUpSampleBlock4 = TemporalUpSampleBlock(hidden_size1, hidden_size1, resFlag=True)
        self.temporalUpSampleBlock5 = TemporalUpSampleBlock(hidden_size1, hidden_size1, resFlag=True)
        self.temporalUpSampleBlock6 = TemporalUpSampleBlock(hidden_size1, hidden_size1, resFlag=True)
        self.temporalUpSampleBlock7 = TemporalUpSampleBlock(hidden_size1, hidden_size1, resFlag=True)
        self.temporalUpSampleBlock8 = TemporalUpSampleBlock(hidden_size1, hidden_size1, resFlag=True)

        self.temporalUpSampleBlock9 = nn.ConvTranspose1d(in_channels=hidden_size1, out_channels=hidden_size3,
                                                         kernel_size=5, stride=scale,
                                                         padding=2, output_padding=(scale - 1))

        self.temporalUpSampleBlock10 = TemporalUpSampleBlock(hidden_size3, hidden_size3, resFlag=True)
        self.temporalUpSampleBlock11 = TemporalUpSampleBlock(hidden_size3, hidden_size3, resFlag=True)
        self.temporalUpSampleBlock12 = TemporalUpSampleBlock(hidden_size3, hidden_size3, resFlag=True)
        self.temporalUpSampleBlock13 = TemporalUpSampleBlock(hidden_size3, hidden_size3, resFlag=True)
        self.conv1D = nn.Conv1d(in_channels=hidden_size3, out_channels=input_size, kernel_size=3, stride=1,
                                 padding=1)

        self.batchNorm1d0 = nn.BatchNorm1d(hidden_size3)
        self.batchNorm1d1 = nn.BatchNorm1d(input_size)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, frames):
        inputData = frames

        inputDataCat = Upsample(inputData, self.scale)

        inputData = self.temporalUpSampleBlock1(inputData)
        inputData = self.temporalUpSampleBlock2(inputData)
        inputData = self.temporalUpSampleBlock3(inputData)
        inputData = self.temporalUpSampleBlock4(inputData)
        inputData = self.temporalUpSampleBlock5(inputData)
        inputData = self.temporalUpSampleBlock6(inputData)
        inputData = self.temporalUpSampleBlock7(inputData)
        inputData = self.temporalUpSampleBlock8(inputData)

        inputData = self.temporalUpSampleBlock9(inputData)
        inputData = self.batchNorm1d0(inputData)
        inputData = self.relu(inputData)

        inputData = self.temporalUpSampleBlock10(inputData)
        inputData = self.temporalUpSampleBlock11(inputData)
        inputData = self.temporalUpSampleBlock12(inputData)
        inputData = self.temporalUpSampleBlock13(inputData)

        inputData = self.conv1D(inputData)
        inputData = self.batchNorm1d1(inputData)

        inputData = inputDataCat + inputData

        inputData = self.relu(inputData)

        return inputData

def Upsample(x, scale):
    inputData = nn.Upsample(scale_factor=scale, mode='nearest')(x)

    shiftNum = scale // 2

    inputData[:, :, :-shiftNum] = inputData[:, :, shiftNum:]

    return inputData