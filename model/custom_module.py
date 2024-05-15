import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
import math
from functools import partial

from einops import rearrange


class MultiScaleDiscriminator(nn.Module):
    def __init__(
            self,
            # input [1, N]
            channels=16,  # 32
            layers=2,  # 3
            groups=(2, 4),
            chan_max=1024,
            input_channels=1
    ):
        super().__init__()
        self.init_conv = nn.Conv1d(input_channels, channels, 5, padding=2)
        # [input_channels, N] -> [channels ,N]
        self.conv_layers = nn.ModuleList([])

        curr_channels = channels

        for _, group in zip(range(layers), groups):
            chan_out = min(curr_channels * 2, chan_max)
            # loop [channels, N] -> [channels #*2#, N #/2#]
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(curr_channels, chan_out, 7, stride=2, padding=3, groups=group),
                nn.LeakyReLU(negative_slope=0.1)
            ))

            curr_channels = chan_out

        self.final_conv = nn.Sequential(
            nn.Conv1d(curr_channels, curr_channels, 5, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(curr_channels, input_channels, 3, padding=1),
        )

    def forward(
            self,
            x,
            return_intermediates=False
    ):
        x = self.init_conv(x)
        # [1, N] -> [channels ,N]
        intermediates = []

        for layer in self.conv_layers:
            # loop [channels, N] -> [channels *2, N /2]
            x = layer(x)
            intermediates.append(x)
        # [channels, N] -> [input_channels, N]
        out = self.final_conv(x)

        if not return_intermediates:
            return out

        return out, intermediates


class MLP_with_stats(nn.Module):
    def __init__(
            self,
            *,
            output_dim,
            layers=2,
            input_channels=1,
            channels=64,
            chan_max=1024,
    ):
        super().__init__()
        self.init_conv = nn.Conv1d(channels, 1, 5, padding=2)

        curr_channels = channels
        self.linear_layers = nn.ModuleList([])
        self.init_linear_cat = nn.Sequential(nn.Linear(input_channels + 1, curr_channels), nn.ReLU())
        for i in range(layers):
            chan_out = int(curr_channels)
            self.linear_layers.append(nn.Sequential(
                nn.Linear(curr_channels, chan_out),
                nn.ReLU()
            ))
            curr_channels = chan_out
        self.final_linear = nn.Linear(curr_channels, output_dim, bias=False)

    def forward(
            self,
            x,
            stats
    ):
        batch_size = x.shape[0]
        x = x.transpose(1, 2)
        x = self.init_conv(x)

        x = torch.cat([x, stats], dim=1)
        x = x.view(batch_size, -1)

        x = self.init_linear_cat(x)

        for layer in self.linear_layers:
            x = layer(x)

        out = self.final_linear(x)
        return out


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=5, padding=padding, padding_mode='replicate')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = torch.permute(x, dims=[0, 2, 1])
        x = x.permute(0, 2, 1)

        return self.pe[:, :x.size(1)].transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class DSAttention(nn.Module):
    def __init__(self, dropout=0.05, output_attention=True):
        super(DSAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / math.sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S
        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        A = self.dropout(torch.nn.functional.softmax(scale * scores, dim=-1))

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class DSAttention_Layer(nn.Module):
    def __init__(self, attention, d_model=16, dropout=0.05, n_heads=8, activation="gelu"):
        super(DSAttention_Layer, self).__init__()
        d_keys = (d_model // n_heads)
        d_values = (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu
        self.n_heads = n_heads

    def forward(self, x, tau=None, delta=None):
        # x = [BS, C, L] , C=channel, L=sample_len
        queries, keys, values = x, x, x
        BS, C, L = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        # print(queries.shape)

        queries = self.query_projection(queries).view(BS, C, H, -1)
        keys = self.key_projection(keys).view(BS, S, H, -1)
        values = self.value_projection(values).view(BS, S, H, -1)
        # print(queries.shape, keys.shape, values.shape)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            tau, delta
        )
        out = out.view(BS, C, -1)
        new_x = self.out_projection(out)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        out = self.norm2(x + y)
        out = self.norm_out(out)

        return out, attn


class Attention(nn.Module):
    def __init__(self, dropout=0.05, output_attention=True):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / math.sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta
        A = self.dropout(torch.nn.functional.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class Attention_Layer(nn.Module):
    def __init__(self, attention, d_model=32, dropout=0.05, n_heads=8, activation="gelu"):
        super(Attention_Layer, self).__init__()
        d_keys = (d_model // n_heads)
        d_values = (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu
        self.n_heads = n_heads
        self.projection = nn.Linear(d_model, 1)

    def forward(self, x, tau=None, delta=None):
        # x = [BS, C, L] , C=channel, L=sample_len
        queries, keys, values = x, x, x
        BS, C, L = queries.shape
        H = self.n_heads
        # print(queries.shape)

        queries = self.query_projection(queries).view(BS, C, H, -1)
        keys = self.key_projection(keys).view(BS, C, H, -1)
        values = self.value_projection(values).view(BS, C, H, -1)
        # print(queries.shape, keys.shape, values.shape)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            tau,
            delta
        )
        out = out.view(BS, C, -1)
        new_x, attn = self.out_projection(out), attn
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        # print(y.transpose(-1, 1).shape)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        out = self.norm2(x + y)
        out = self.norm_out(out)

        if self.projection is not None:
            out = self.projection(out)
        return out, attn


class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, pad_mode='constant', **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)
        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        # x = x.unsqueeze(1)

        x = F.pad(x.unsqueeze(1), (self.causal_padding, 0, 0, 0), mode=self.pad_mode)
        x = x.squeeze(1)

        return self.conv(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def ResidualUnit(chan_in, chan_out, kernel_size=7, squeeze_excite=False, pad_mode='reflect'):
    return Residual(nn.Sequential(
        CausalConv1d(chan_in, chan_out, kernel_size, pad_mode=pad_mode),
        nn.ELU(),
        CausalConv1d(chan_out, chan_out, 1, pad_mode=pad_mode),
        nn.ELU(),
        # SqueezeExcite(chan_out) if squeeze_excite else None
    ))


def EncoderBlock(chan_in, chan_out, stride, squeeze_excite=False, pad_mode='reflect'):
    residual_unit = partial(ResidualUnit, squeeze_excite=squeeze_excite, pad_mode=pad_mode)

    return nn.Sequential(
        residual_unit(chan_in, chan_in),
        CausalConv1d(chan_in, chan_out, 2 * stride, stride=stride)
    )


def DecoderBlock(chan_in, chan_out, stride, squeeze_excite=False, pad_mode='reflect'):
    even_stride = (stride % 2 == 0)
    padding = (stride + (0 if even_stride else 1)) // 2
    output_padding = 0 if even_stride else 1

    residual_unit = partial(ResidualUnit, squeeze_excite=squeeze_excite, pad_mode=pad_mode)

    return nn.Sequential(
        CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride=stride),
        residual_unit(chan_out, chan_out),
    )


class CausalConvTranspose1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]
        # print('before ConvTranspose1d', x.shape)
        out = self.conv(x)
        # print('after ConvTranspose1d', out.shape)
        out = out[..., :(n * self.upsample_factor)]

        return out


class SqueezeExcite(nn.Module):
    def __init__(self, dim, reduction_factor=4, dim_minimum=8):
        super().__init__()
        dim_inner = max(dim_minimum, dim // reduction_factor)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, 1),
            nn.SiLU(),
            nn.Conv1d(dim_inner, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seq, device = x.shape[-2], x.device

        # cumulative mean - since it is autoregressive
        cum_sum = x.cumsum(dim=-2)
        # print('cum_sum', cum_sum.shape)
        denom = torch.arange(1, seq + 1, device=device).float()
        cum_mean = cum_sum / rearrange(denom, 'n -> n 1')

        # glu gate

        gate = self.net(cum_mean)
        # print(gate.shape)

        return x * gate


if __name__ == '__main__':
    # print(MultiScaleDiscriminator())
    # summary(MultiScaleDiscriminator(), (42, 1, 64))
    # test = MLP_with_stats(output_dim=1)(x=torch.randn((42, 1, 64)), stats=torch.randn((42, 1, 1)))

    # summary(MLP_with_stats(output_dim=1), input_size=[(42, 1, 64), (42, 1, 1)])

    # summary(PositionalEmbedding(d_model=64), input_size=(42, 1, 2800))
    # summary(TokenEmbedding(c_in=1, d_model=1), input_size=(42, 1, 64))
    # summary(DataEmbedding(c_in=1, d_model=1), input_size=(42, 1, 64))

    summary(DSAttention_Layer(DSAttention()), input_size=(42, 1, 64))
