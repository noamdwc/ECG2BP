import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import IOHModelBinaryClassifier, register_model
from .backbone.resnet import resnet18


# Chomping layer to handle sequence length issues in causal convolutions
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


# Temporal block in the TCN
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# TCN architecture that stacks multiple temporal blocks
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Embedding model based on ResNet architecture
def create_embedding_model(resnet_args: dict):
    return resnet18(**resnet_args)


# Main TCN + ResNet Model
@register_model
class CustomTCN(IOHModelBinaryClassifier):
    def __init__(self, tcn_model,
                 tcn_output_size,
                 criterion,
                 learning_rate,
                 seq_len,
                 output_size,
                 num_tokens,
                 pooling_type="mean",
                 embedding_model=None,
                 attention=None,
                 cfg=None):
        super().__init__(learning_rate=learning_rate)
        self.cfg = cfg or {}
        self.seq_len = seq_len
        self.embedding_model = embedding_model
        self.use_embedding = embedding_model is not None
        self.attention = attention
        self.use_attention = attention is not None
        self.num_tokens = num_tokens
        self.chunks_to_tokenize = round((seq_len / 2 ** 3) / num_tokens)
        self.tcn_model = tcn_model
        self.tcn_output_size = tcn_output_size
        self.fc = nn.Linear(32, output_size)  # 32 is the TCN output size (from the error message)
        self.criterion = criterion
        self.pooling_type = pooling_type


    @classmethod
    def from_config(cls, cfg):
        # Use embedding size for the TCN input size
        if cfg['MODEL']['KWARGS']['use_embedding']:
            embedding_model = create_embedding_model(dict(cfg['MODEL']['KWARGS']['RESNET_ARGS']))
            tcn_input_size = cfg['MODEL']['KWARGS']['RESNET_ARGS']['embedding_size']
        else:
            tcn_input_size = cfg['MODEL']['KWARGS']['seq_len']
            embedding_model = None

        tcn_model, tcn_output_size = cls.create_tcn_model(input_size=10,  # Change to 10 if ResNet outputs 10 channels
                                                          num_channels=cfg['MODEL']['KWARGS']['TCN_ARGS'][
                                                              'num_channels'],
                                                          kernel_size=cfg['MODEL']['KWARGS']['TCN_ARGS']['kernel_size'],
                                                          dropout=cfg['MODEL']['KWARGS']['TCN_ARGS']['dropout'])

        if cfg['MODEL']['KWARGS']['use_attention']:
            attention = nn.MultiheadAttention(tcn_output_size, cfg['MODEL']['KWARGS']['attention_head_size'],
                                              batch_first=True)
        else:
            attention = None

        if cfg['OPT']['LOSS'] == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss type {cfg['OPT']['LOSS']} not supported, only cross_entropy is supported for now.")

        return {
            "tcn_model": tcn_model,
            "tcn_output_size": tcn_output_size,
            "criterion": criterion,
            "learning_rate": cfg['OPT']['LR'],
            "seq_len": cfg['MODEL']['KWARGS']['seq_len'],
            "output_size": cfg['MODEL']['KWARGS']['output_size'],
            "num_tokens": cfg['MODEL']['KWARGS']['num_tokens'],
            "pooling_type": cfg['MODEL']['KWARGS']['pooling'],
            "embedding_model": embedding_model,
            "attention": attention,
            "cfg": cfg
        }

    @staticmethod
    def create_tcn_model(input_size, num_channels, kernel_size=2, dropout=0.2):
        return TemporalConvNet(input_size, num_channels, kernel_size, dropout), num_channels[-1]

    def forward(self, x):
        # Apply embedding model if used
        if self.use_embedding:
            x = self.embedding_model(x)

        # Reduce input dimensionality
        x = F.avg_pool1d(x, kernel_size=self.chunks_to_tokenize, stride=self.chunks_to_tokenize)

        # Apply TCN
        x = self.tcn_model(x.permute(0, 2, 1))  # Adjust shape for TCN (batch_size, channels, seq_len)

        # Apply attention if used
        if self.use_attention:
            x, _ = self.attention(x, x, x)  # Apply attention (batch_size, seq_len, features)

        # Pooling
        if self.pooling_type == 'last':
            x = x[:, -1, :]  # Use the last time step
        elif self.pooling_type == 'mean':
            x = torch.mean(x, dim=1)  # Global average pooling
        elif self.pooling_type == 'max':
            x, _ = torch.max(x, dim=1)  # Global max pooling

        # Fully connected layer for classification
        x = self.fc(x)
        return x
