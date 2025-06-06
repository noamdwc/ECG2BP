import os
from .model import IOHModelBinaryClassifier, register_model
from torch import nn
from .backbone.unet import UNetEncoder, UNetDecoder


@register_model
class WaveSegIOH(IOHModelBinaryClassifier):
    def __init__(self, unet_encoder, unet_decoder, predictor, lambda_seg, learning_rate):
        super().__init__(learning_rate)
        self.une_encoder = unet_encoder
        self.unet_decoder = unet_decoder
        self.predictor = predictor
        self.lambda_seg = lambda_seg
        self.cross_entropy = nn.CrossEntropyLoss()

    @classmethod
    def from_config(cls, cfg):
        pass

    def forward(self, batch):
        x = batch["x"]
        y = batch["y"]
        y_seg = batch["y_seg"]
        seg_mask = batch["seg_mask"]
        enc_features, x = self.unet_encoder(x)
        y_decoder = self.unet_decoder(x, enc_features)
        y_hat = self.predictor(x)
        loss = self._criterion(y_hat, y_decoder[seg_mask], y, y_seg[seg_mask])
        return {"loss": loss, "y_hat": y_hat}

    def _criterion(self, y_hat, y_decoder, y, y_seg):
        loss = {
            "loss_classification": self.cross_entropy(y_hat, y),
            "loss_segmentation": self.lambda_seg * self.cross_entropy(y_seg, y_decoder)
        }
        return loss
