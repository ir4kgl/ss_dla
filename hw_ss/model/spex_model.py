# inspired by https://github.com/gemengtju/SpEx_Plus/blob/master/nnet/conv_tas_net.py

import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F

from torch.nn import ConstantPad2d


class SpeechEncoder(nn.Module):
    def __init__(self, embed_dim, L, L1, L2, L3):
        super().__init__()
        self.embed_dim = embed_dim
        self.short_encoder = nn.Conv1d(1, embed_dim, L, stride=L // 2)
        self.middle_encoder = nn.Conv1d(1, embed_dim, L2, stride=L // 2)
        self.long_encoder = nn.Conv1d(1, embed_dim, L3, stride=L // 2)
        self.relu = nn.ReLU()
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        nn.init.xavier_uniform_(self.short_encoder.weight)
        self.short_encoder.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.middle_encoder.weight)
        self.middle_encoder.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.long_encoder.weight)
        self.long_encoder.bias.data.fill_(0)

    def forward(self, x):
        x1 = self.short_encoder(x)
        len_middle = (x1.shape[-1] - 1) * (self.L1 // 2) + self.L2
        len_long = (x1.shape[-1] - 1) * (self.L1 // 2) + self.L3
        x2 = F.pad(x, (0, len_middle - x.shape[-1]), "constant", 0)
        x3 = F.pad(x, (0, len_long - x.shape[-1]), "constant", 0)
        x2 = self.middle_encoder(x2)
        x3 = self.long_encoder(x3)
        return self.relu(x1), self.relu(x2), self.relu(x3)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_layers = Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        if in_channels != out_channels:
            self.last_conv = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, bias=False)
            nn.init.xavier_uniform_(self.last_conv.weight)
        self.last_layers = Sequential(
            nn.PReLU(),
            nn.MaxPool1d(max_pool)
        )
        nn.init.xavier_uniform_(self.first_layers[0].weight)
        nn.init.xavier_uniform_(self.first_layers[3].weight)

    def forward(self, x):
        x_prime = x
        x = self.first_layers(x)
        if self.in_channels != self.out_channels:
            x_prime = self.last_conv(x_prime)
        x = x + x_prime
        return self.last_layers(x)


class SpeakerEncoder(nn.Module):
    def __init__(self, embed_dim, hidden, L1):
        super().__init__()
        self.L1 = L1
        self.layers = Sequential(
            ChannelWiseLayerNorm(3*embed_dim),
            nn.Conv1d(3*embed_dim, embed_dim, 1),
            ResNetBlock(embed_dim, embed_dim),
            ResNetBlock(embed_dim, hidden),
            ResNetBlock(hidden, hidden),
            nn.Conv1d(hidden, embed_dim, 1),
            nn.AvgPool1d(3)
        )
        nn.init.xavier_uniform_(self.layers[1].weight)
        self.layers[1].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.layers[5].weight)
        self.layers[5].bias.data.fill_(0)

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], 1)
        return self.layers(x).mean(-1)


class GlobalChannelLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-07):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
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
            nn.Conv1d(in_channels, hidden, 1),
            nn.PReLU(),
            GlobalChannelLayerNorm(hidden),
            nn.Conv1d(hidden, hidden, kernel_size, groups=hidden,
                      padding=(dilation*(kernel_size-1)) // 2, dilation=dilation),
            nn.PReLU(),
            GlobalChannelLayerNorm(hidden),
            nn.Conv1d(hidden, out_channels, 1)
        )
        nn.init.xavier_uniform_(self.layers[0].weight)
        self.layers[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.layers[3].weight)
        self.layers[3].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.layers[6].weight)
        self.layers[6].bias.data.fill_(0)

    def forward(self, x):
        return self.layers(x)


class TCNResidual(TCN):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, dilation=1):
        super().__init__(in_channels, hidden, out_channels, kernel_size, dilation)

    def forward(self, x):
        x_prime = x
        x = self.layers(x)
        return x + x_prime


class TCNBlock(nn.Module):
    def __init__(self, in_channels, hidden, speaker_dim, B):
        super().__init__()
        self.first = TCN(in_channels + speaker_dim, hidden, in_channels)
        self.layers = []
        for i in range(1, B):
            self.layers.append(TCNResidual(
                in_channels, hidden, in_channels, dilation=(2 ** i)))
        self.layers = Sequential(*self.layers)

    def forward(self, x, v):
        v = torch.unsqueeze(v, -1)
        v = v.repeat(1, 1, x.shape[-1])
        x_prime = x
        x = self.first(torch.cat([x, v], 1))
        x = x + x_prime
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
    def __init__(self, embed_dim, hidden, B=8, R=4):
        super().__init__()
        self.tail = Sequential(ChannelWiseLayerNorm(
            3*embed_dim), nn.Conv1d(3*embed_dim, hidden, 1))
        self.tcn = []
        for i in range(R):
            self.tcn.append(TCNBlock(embed_dim, 2*embed_dim, embed_dim, B))
        self.tcn = Sequential(*self.tcn)
        self.head1 = Sequential(nn.Conv1d(hidden, embed_dim, 1), nn.ReLU())
        self.head2 = Sequential(nn.Conv1d(hidden, embed_dim, 1), nn.ReLU())
        self.head3 = Sequential(nn.Conv1d(hidden, embed_dim, 1), nn.ReLU())

        nn.init.xavier_uniform_(self.tail[1].weight)
        self.tail[1].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.head1[0].weight)
        self.head1[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.head2[0].weight)
        self.head2[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.head3[0].weight)
        self.head3[0].bias.data.fill_(0)

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
    def __init__(self, embed_dim, L, L1, L2, L3):
        super().__init__()
        self.decoder1 = nn.ConvTranspose1d(
            embed_dim, 1, kernel_size=L, stride=L//2)
        self.decoder2 = nn.ConvTranspose1d(
            embed_dim, 1, kernel_size=L2, stride=L//2)
        self.decoder3 = nn.ConvTranspose1d(
            embed_dim, 1, kernel_size=L3, stride=L//2)
        nn.init.xavier_uniform_(self.decoder1.weight)
        self.decoder1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.decoder2.weight)
        self.decoder2.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.decoder3.weight)
        self.decoder3.bias.data.fill_(0)

    def forward(self, s1, s2, s3, required_shape):
        s1 = self.decoder1(s1)
        s2 = self.decoder2(s2)
        s3 = self.decoder3(s3)
        if s1.shape[-1] < required_shape:
            s1 = ConstantPad2d((0, required_shape - s1.shape[-1], 0, 0), 0)(s1)
        if s2.shape[-1] < required_shape:
            s2 = ConstantPad2d((0, required_shape - s2.shape[-1], 0, 0), 0)(s2)
        if s3.shape[-1] < required_shape:
            s3 = ConstantPad2d((0, required_shape - s3.shape[-1], 0, 0), 0)(s3)
        return torch.squeeze(s1, 1), torch.squeeze(s2, 1)[:, :s1.shape[-1]], torch.squeeze(s3, 1)[:, :s1.shape[-1]]


class ClassificatorHead(nn.Module):
    def __init__(self, num_classes, embed_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, v):
        return self.layers(v)


class SpexPlus(nn.Module):  # Spex with classification head
    def __init__(self, num_classes,  embed_dim=256, hidden=256, speaker_encoder_hidden=512, L=40, B=8, R=4, L1=40, L2=160, L3=320):
        super().__init__()
        self.speech_encoder = SpeechEncoder(embed_dim, L, L1, L2, L3)
        self.speaker_encoder = SpeakerEncoder(
            embed_dim, speaker_encoder_hidden, L1)
        self.speaker_extractor = SpeakerExtractor(embed_dim, hidden, B, R)
        self.speech_decoder = SpeechDecoder(embed_dim, L, L1, L2, L3)
        self.classificator = ClassificatorHead(num_classes)

    def forward(self, batch):
        # ref = batch["ref"] / batch["ref"].norm(dim=-1,  keepdim=True)
        # audio = batch["audio"] / batch["audio"].norm(dim=-1, keepdim=True)
        ref = batch["ref"]
        audio = batch["audio"]
        X1, X2, X3 = self.speech_encoder(ref)
        Y1, Y2, Y3 = self.speech_encoder(audio)

        v = self.speaker_encoder(X1, X2, X3)
        predicted_logits = self.classificator(v)

        M1, M2, M3 = self.speaker_extractor(Y1, Y2, Y3, v)
        S1 = Y1 * M1
        S2 = Y2 * M2
        S3 = Y3 * M3
        S1, S2, S3 = self.speech_decoder(S1, S2, S3, batch["audio"].shape[-1])
        predicted_audio = 20 * S1 / S1.norm(dim=-1,  keepdim=True), 20 * S2 / S2.norm(
            dim=-1,  keepdim=True), 20 * S3 / S3.norm(dim=-1,  keepdim=True)
        predicted_audio = S1, S2, S3
        return {
            "predicted_audio": predicted_audio,
            "predicted_logits": predicted_logits
        }
