import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetEncoder(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.pools = nn.ModuleList()

        for in_channels, out_channels, kernel_size, pool_size in layers_config:
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.pools.append(nn.MaxPool1d(kernel_size=pool_size))

    def forward(self, x):
        enc_features = []
        for layer, pool in zip(self.layers, self.pools):
            x = layer(x)
            enc_features.append(x)
            x = pool(x)
        return enc_features, x


class UNetDecoder(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layers_config) - 1, 0, -1):
            in_channels = layers_config[i][1]
            out_channels = layers_config[i - 1][1]
            kernel_size = layers_config[i - 1][2]
            pool_size = layers_config[i][3]
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels, out_channels, kernel_size=pool_size, stride=pool_size),
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x, enc_features):
        for i, layer in enumerate(self.layers):
            x = layer[0](x)
            x = torch.cat([x, enc_features[-(i + 2)]], dim=1)
            x = layer[1:](x)
        return x

