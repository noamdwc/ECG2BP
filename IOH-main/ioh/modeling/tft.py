from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.metrics import CrossEntropy
from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
import torch
import torch.nn as nn
from .model import NNModel
from .backbone.resnet import resnet18
from .model import register_model


@register_model
class TFTWrapper(NNModel):
    def __init__(self, tft_model):
        super(TFTWrapper, self).__init__()
        self.tft_model = tft_model

    def forward(self, **kwargs):
        return self.tft_model(**kwargs)  # Pass through TFT

    @classmethod
    def from_config(cls, cfg, dataset: TimeSeriesDataSet = None):
        tft_model = TemporalFusionTransformer.from_dataset(dataset, dataset.data, loss=CrossEntropy(), **dict(cfg.MODEL.KWARGS.TFT_ARGS))
        return dict(tft_model=tft_model)


@register_model
class CustomTFTModel(NNModel):
    def __init__(self, embedding_model, tft_model, num_classes=2):
        super(CustomTFTModel, self).__init__()
        self.embedding_model = embedding_model
        self.tft_model = tft_model
        self.num_classes = num_classes

    def forward(self, x):
        B, N, L = x.size()
        x = x.view(B * N, 1, L)  # Reshape for ResNet (B*N, 1, L)
        x = self.embedding_model(x)  # Get embeddings
        x = x.view(B, N, -1)  # Reshape back to (B, N, embedding_dim)
        return self.tft_model(x)  # Pass through TFT

    @classmethod
    def from_config(cls, cfg, dataset: TimeSeriesDataSet=None):
        embedding_model = resnet18(**dict(cfg.MODEL.KWARGS.RESNET_ARGS))
        tft_model = TemporalFusionTransformer(
            loss=CrossEntropy(),
            **dict(cfg.MODEL.KWARGS.TFT_ARGS)
        )
        num_classes = cfg.MODEL.KWARGS.num_classes
        return dict(embedding_model=embedding_model, tft_model=tft_model, num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


