import os
from .model import IOHModelBinaryClassifier, register_model
from torch import nn
from .backbone.resnet import resnet18
import torch
import torch.nn.functional as F


def create_rnn_model(rnn_type, input_size, hidden_size, rnn_layers, dropout):
    if rnn_type == "bi_lstm":
        rnn = nn.LSTM(input_size, hidden_size, num_layers=rnn_layers,
                           dropout=dropout, bidirectional=True, batch_first=True)
        rnn_output_size = hidden_size * 2
    elif rnn_type == "lstm":
        rnn = nn.LSTM(input_size, hidden_size, num_layers=rnn_layers,
                           dropout=dropout, batch_first=True)
        rnn_output_size = hidden_size
    elif rnn_type == "gru":
        rnn = nn.GRU(input_size, hidden_size, num_layers=rnn_layers,
                          dropout=dropout, batch_first=True)
        rnn_output_size = hidden_size
    else:
        raise ValueError(f"RNN type {rnn_type} not supported.")
    return rnn, rnn_output_size


def create_embedding_model(resnet_args:dict):
    return resnet18(**resnet_args)


@register_model
class CustomRNN(IOHModelBinaryClassifier):
    def __init__(self, rnn_model,
                 rnn_output_size,
                 criterion,
                 learning_rate,
                 seq_len,
                 output_size,
                 num_tokens,
                 pooling_type="mean",
                 embedding_model=None,
                 attention=None):
        super().__init__()
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.embedding_model = embedding_model
        self.use_embedding = embedding_model is not None
        self.attention = attention
        self.use_attention = attention is not None
        self.num_tokens = num_tokens
        self.chunks_to_tokenize = round((seq_len/2**3) / num_tokens)
        self.rnn_model = rnn_model
        self.rnn_output_size = rnn_output_size
        self.fc = nn.Linear(rnn_output_size, output_size)
        self.criterion = criterion
        self.pooling_type = pooling_type

    @classmethod
    def from_config(cls, cfg):
        if cfg.MODEL.KWARGS.use_embedding:
            embedding_model = create_embedding_model(dict(cfg.MODEL.KWARGS.RESNET_ARGS))
            rnn_input_size = embedding_model.embedding_size
        else:
            rnn_input_size = cfg.MODEL.KWARGS.seg_len
            embedding_model = None
        rnn_model, rnn_output_size = create_rnn_model(**dict(cfg.MODEL.KWARGS.RNN_ARGS), input_size=rnn_input_size)

        if cfg.MODEL.KWARGS.use_attention:
            attention = nn.MultiheadAttention(rnn_output_size,
                                              cfg.MODEL.KWARGS.attention_head_size,
                                              batch_first=True)
        else:
            attention = None

        if cfg.OPT.LOSS == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss type {cfg.OPT.LOSS} not supported only cross_entropy is supported for now.")

        return {
            "rnn_model": rnn_model,
            "num_tokens": cfg.MODEL.KWARGS.num_tokens,
            "rnn_output_size": rnn_output_size,
            "criterion": criterion,
            "learning_rate": cfg.OPT.LR,
            "seq_len": cfg.INPUT.DATASET_KWARGS.seq_len,
            "output_size": cfg.MODEL.KWARGS.output_size,
            "embedding_model": embedding_model,
            "attention": attention,
            "pooling_type": cfg.MODEL.KWARGS.pooling
        }

    def forward(self, x):
        if self.use_embedding:
            x = self.embedding_model(x)
            # x = x.view(x.size(0), -1)  # Flatten the output
            # x = x.unsqueeze(1).repeat(1, round(self.seq_len/2**3) // x.size(1), 1)  # Split and repeat for token size
        x = F.avg_pool1d(x, kernel_size=self.chunks_to_tokenize, stride=self.chunks_to_tokenize)
        x, _ = self.rnn_model(x.permute(0, 2, 1))  # RNN expects (batch, seq_len, input_size)
        if self.use_attention:
            x, _ = self.attention(x, x, x)

        # Apply the pooling method based on configuration
        if self.pooling_type == 'last':
            x = x[:, -1, :]  # Use the last output
        elif self.pooling_type == 'mean':
            x = torch.mean(x, dim=1)  # Global average pooling
        elif self.pooling_type == 'max':
            x, _ = torch.max(x, dim=1)  # Global max pooling

        x = self.fc(x)  # Use the aggregated output for classification

        return x

