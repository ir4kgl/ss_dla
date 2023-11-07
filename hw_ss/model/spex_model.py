# inspired by https://github.com/gemengtju/SpEx_Plus/blob/master/nnet/conv_tas_net.py

import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F


class SpeechEncoder(nn.Module):
    def __init__(self, embed_dim, L, L1, L2, L3):
        super().__init__()
        self.embed_dim = embed_dim
        self.short_encoder = nn.Conv1d(1, embed_dim, L, stride=L // 2)
        self.middle_encoder = nn.Conv1d(1, embed_dim, L, stride=L // 2)
        self.long_encoder = nn.Conv1d(1, embed_dim, L, stride=L // 2)
        self.relu = nn.ReLU()
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

    def forward(self, x):
        x1 = self.short_encoder(x)
        len_middle = (x1.shape[-1] - 1) * (self.L1 // 2) + self.L2
        len_long = (x1.shape[-1] - 1) * (self.L1 // 2) + self.L3
        x2 = F.pad(x, (0, len_middle - x.shape[-1], "constant", 0))
        x3 = F.pad(x, (0, len_long - x.shape[-1], "constant", 0))
        x2 = self.middle_encoder(x2)
        x3 = self.middle_encoder(x3)
        return self.relu(x1), self.relu(x2), self.relu(x3)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=3):
        super().__init__()
        self.first_layers = Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.last_layers = Sequential(
            nn.PReLU(),
            nn.MaxPool1d(max_pool)
        )

    def forward(self, x):
        x_prime = x.copy()
        x = self.first_layers(x)
        x = x + x_prime
        return self.last_layers(x)


class SpeakerEncoder(nn.Module):
    def __init__(self, embed_dim, hidden, L1):
        super().__init__()
        self.L1 = L1
        self.layers = Sequential(
            self.layers.append(ChannelWiseLayerNorm(3*embed_dim)),
            nn.Conv1d(3*embed_dim, embed_dim, 1),
            ResNetBlock(embed_dim, embed_dim),
            ResNetBlock(embed_dim, hidden),
            ResNetBlock(hidden, hidden),
            nn.Conv1d(hidden, embed_dim, 1),
            nn.AvgPool1d((1, 3))
        )

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], 1)
        return self.layers(x)


class GlobalChannelLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        self.beta = nn.Parameter(torch.zeros(dim, 1))
        self.gamma = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x):
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        return x


class TCN(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.layers = Sequential(
            nn.Conv1D(in_channels, hidden, 1),
            nn.PReLU(),
            GlobalChannelLayerNorm(hidden),
            nn.Conv1d(hidden, hidden, kernel_size, groups=hidden,
                      padding=dilation*(kernel_size-1), dilation=dilation),
            nn.PReLU(),
            GlobalChannelLayerNorm(hidden),
            nn.Conv1d(hidden, out_channels, 1)
        )

    def forward(self, x):
        return self.layers(x)


class TCNResidual(TCN):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, dilation=1):
        super().__init__(in_channels, hidden, out_channels, kernel_size, dilation)

    def forward(self, x):
        x_prime = x.copy()
        x = self.layers(x)
        return x + x_prime


class TCNBlock(nn.Module):
    def __init__(self, in_channels, hidden, speaker_dim, B):
        super().__init__()
        self.first = TCN(in_channels + speaker_dim, hidden, in_channels)
        self.layers = []
        for i in range(B-1):
            self.layers.append(TCNResidual(in_channels, hidden, in_channels))
        self.layers = Sequential(self.layers)

    def forward(self, x, v):
        v = torch.unsqueeze(v, -1)
        v = v.repeat(1, 1, x.shape[-1])
        x_prime = x.copy()
        x = self.first(torch.cat([x, v], 1))
        x + x_prime
        return self.layers(x)


class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = super().forward(x)
        x = torch.transpose(x, 1, 2)
        return x


class SpeakerExtractor(nn.Module):
    def __init__(self, embed_dim, B=8, R=4):
        super().__init__()
        self.tail = Sequential(ChannelWiseLayerNorm(
            3*embed_dim), nn.Conv1D(3*embed_dim, B, 1))
        self.tcn = []
        for i in range(R):
            self.tcn.append(TCNBlock(B))
        self.head1 = Sequential(nn.Conv1D(B, embed_dim, 1), nn.ReLU())
        self.head2 = Sequential(nn.Conv1D(B, embed_dim, 1), nn.ReLU())
        self.head3 = Sequential(nn.Conv1D(B, embed_dim, 1), nn.ReLU())

    def forward(self, y1, y2, y3, v):
        y = self.tail(torch.cat([y1, y2, y3], 1))
        for layer in self.tcn:
            y = layer(y, v)
        m1 = self.head1(y)
        m2 = self.head2(y)
        m3 = self.head3(y)
        return m1, m2, m3


class ConvTrans1D(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        return torch.squeeze(x)


class SpeechDecoder(nn.Module):
    def __init__(self, embed_dim, L):
        super().__init__()
        self.decoder1 = nn.ConvTranspose1d(
            embed_dim, 1, kernel_size=L, stride=L//2, bias=True)
        self.decoder2 = nn.ConvTranspose1d(
            embed_dim, 1, kernel_size=L, stride=L//2, bias=True)
        self.decoder3 = nn.ConvTranspose1d(
            embed_dim, 1, kernel_size=L, stride=L//2, bias=True)

    def forward(self, s1, s2, s3):
        s1 = torch.unsqueeze(s1, 1)
        s2 = torch.unsqueeze(s2, 1)
        s3 = torch.unsqueeze(s3, 1)
        s1 = self.decoder1(s1)
        s2 = self.decoder1(s2)
        s3 = self.decoder1(s3)
        return torch.squeeze(s1), torch.squeeze(s2), torch.squeeze(s3)


class SpexSimple(nn.Module):  # no classification head
    def __init__(self, embed_dim=256, speaker_encoder_hidden=512, L=20, B=8, R=4, n_r=3, L1=20, L2=80, L3=160):
        super().__init__()
        self.speech_encoder = SpeechEncoder(embed_dim, L, L1, L2, L3)
        self.speaker_extractor = SpeakerEncoder(
            embed_dim, speaker_encoder_hidden, L1)
        self.speaker_encoder = SpeakerExtractor(embed_dim, B, R)
        self.speech_decoder = SpeechDecoder(embed_dim, L)

    def forward(self, y, x):
        X1, X2, X3 = self.speech_encoder(x)
        Y1, Y2, Y3 = self.speech_encoder(y)
        v = self.speaker_encoder(X1, X2, X3)
        M1, M2, M3 = self.speaker_extractor(Y1, Y2, Y3, v)
        S1 = Y1 * M1
        S2 = Y2 * M2
        S3 = Y3 * M3
        return self.speech_decoder(S1, S2, S3)
