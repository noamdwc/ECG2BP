import torch.nn as nn


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        return self.relu(out)


class ResNet1DEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, out_dim=128, num_blocks=(2, 2, 2, 2)):
        """
        num_blocks: list or tuple, length = number of stages
        Each value = number of residual blocks in that stage.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        layers = []
        in_ch = base_channels
        for i, blocks in enumerate(num_blocks):
            out_ch = base_channels * (2 ** i)
            stride = 1 if i == 0 else 2
            layers.append(self._make_layer(in_ch, out_ch, blocks, stride))
            in_ch = out_ch  # next input = current output

        self.layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, out_dim)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = []
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        layers.append(BasicBlock1D(in_ch, out_ch, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


class CNNLSTMModel(nn.Module):
    def __init__(self, cnn_out_dim=128, lstm_hidden=128, num_classes=6, num_blocks=(2, 2, 2)):
        super().__init__()
        self.cnn = ResNet1DEncoder(in_channels=1, out_dim=cnn_out_dim, num_blocks=num_blocks)
        self.lstm = nn.LSTM(input_size=cnn_out_dim, hidden_size=lstm_hidden, dropout=0.3,
                            batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        B, T, L = x.shape
        x = x.view(B * T, 1, L)
        x = self.cnn(x)
        x = x.view(B, T, -1)
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out)
