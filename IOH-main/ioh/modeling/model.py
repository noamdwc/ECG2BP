from pytorch_forecasting.models.base_model import BaseModel
from pytorch_forecasting.data import TimeSeriesDataSet
import torch
from torch import nn
import glob
import os
from typing import Dict
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from torchmetrics.classification import Accuracy, F1Score, AUROC, Precision, Recall

MODEL_REGISTRY = {}


def register_model(cls):
    MODEL_REGISTRY[cls.__name__] = cls
    return cls


def create_model(cfg):
    model_type = cfg.MODEL.NAME
    return MODEL_REGISTRY[model_type](**MODEL_REGISTRY[model_type].from_config(cfg))


def load_model_weights(model, cfg, resume=False):
    # check if there are weights to load
    if resume:
        exp_dir = cfg.EXP_DIR
        # get the latest checkpoint
        list_of_files = glob.glob(f"{exp_dir}/checkpoints/*.pth")
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Resuming. Loading model type: {cfg.MODEL.NAME} from: {latest_file}")
        model.load_state_dict(torch.load(latest_file))
    elif cfg.MODEL.WEIGHTS:
        print(f"Loading model type: {cfg.MODEL.NAME} from config file: {cfg.model.weights}")
        model.load_state_dict(torch.load(cfg.model.weights))
    return model


def load_model(cfg, resume=False):
    model = create_model(cfg)
    return load_model_weights(model, cfg, resume)


def create_model_with_time_series(cfg, dataset: TimeSeriesDataSet):
    model_type = cfg.MODEL.NAME
    return MODEL_REGISTRY[model_type].from_config(cfg, dataset)


def load_model_with_time_series(cfg, dataset: TimeSeriesDataSet, resume=False):
    model = create_model_with_time_series(cfg, dataset)
    return load_model_weights(model, cfg, resume)


def load_model_from_checkpoint(cfg, checkpoint_path):
    model = create_model(cfg)
    model = model.load_from_checkpoint(checkpoint_path)
    return model


def load_model_from_checkpoint_with_time_series(cfg, checkpoint_path, dataset: TimeSeriesDataSet):
    model = create_model_with_time_series(cfg, dataset)
    model = model.load_from_checkpoint(checkpoint_path)
    return model


class IOHModel(LightningModule):

    @classmethod
    def from_config(cls, cfg):
        """
        Create a model from a config and pytorch-forecasting dataset. This method must be implemented by the subclass.
        :param cfg:
        :return: A dict containing the arguments to the constructor of the model
        """
        pass


class IOHModelBinaryClassifier(IOHModel):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.thresholds = [0.1 * i for i in range(1, 10)]  # Thresholds from 0.1 to 0.9
        self.thresholds_keys = [f"{t:.1f}" for t in self.thresholds]
        self.val_f1_thresholds = {self.thresholds_keys[i]: F1Score(task="binary", threshold=t) for i, t in
                                  enumerate(self.thresholds)}
        self.val_recall_thresholds = {self.thresholds_keys[i]: Recall(task="binary", threshold=t) for i, t in
                                      enumerate(self.thresholds)}
        self.val_precision_thresholds = {self.thresholds_keys[i]: Precision(task="binary", threshold=t) for i, t in
                                         enumerate(self.thresholds)}

        self.train_f1_thresholds = {self.thresholds_keys[i]: F1Score(task="binary", threshold=t) for i, t in
                                    enumerate(self.thresholds)}
        self.train_recall_thresholds = {self.thresholds_keys[i]: Recall(task="binary", threshold=t) for i, t in
                                        enumerate(self.thresholds)}
        self.train_precision_thresholds = {self.thresholds_keys[i]: Precision(task="binary", threshold=t) for i, t in
                                           enumerate(self.thresholds)}

        self.val_f1 = F1Score(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.val_auc = AUROC(task="binary")
        self.train_accuracy = Accuracy(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.train_auc = AUROC(task="binary")

    def forward(self, batch) -> dict:
        """
        Forward pass of the model. This method must be implemented by the subclass with this signature.
        :param batch: input from the dataset contains "y" the labels and the other inputs
        :return: a dict contains "loss" and "y_hat" logits for the binary classification
        """
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)

        preds = torch.argmax(y_hat, dim=1)
        self.train_accuracy.update(preds, y)
        self.train_f1.update(preds, y)
        self.train_auc.update(y_hat[:, 1], y)
        # Loop through the thresholds to calculate metrics
        for t in self.thresholds_keys:
            self.train_f1_thresholds[t].update(y_hat[:, 1].cpu(), y.cpu())
            self.train_recall_thresholds[t].update(y_hat[:, 1].cpu(), y.cpu())
            self.train_precision_thresholds[t].update(y_hat[:, 1].cpu(), y.cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

        preds = torch.argmax(y_hat, dim=1)
        self.val_accuracy.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_auc.update(y_hat[:, 1], y)

        # Loop through the thresholds to calculate metrics
        for t in self.thresholds_keys:
            self.val_f1_thresholds[t].update(y_hat[:, 1].cpu(), y.cpu())
            self.val_recall_thresholds[t].update(y_hat[:, 1].cpu(), y.cpu())
            self.val_precision_thresholds[t].update(y_hat[:, 1].cpu(), y.cpu())

        return loss

    def training_epoch_end(self, outputs):
        train_acc = self.train_accuracy.compute()
        train_f1 = self.train_f1.compute()
        train_auc = self.train_auc.compute()
        self.log('train_acc', train_acc)
        self.log('train_f1', train_f1)
        self.log('train_auc', train_auc)
        self.train_accuracy.reset()
        self.train_f1.reset()
        self.train_auc.reset()
        val_acc = self.val_accuracy.compute()
        val_f1 = self.val_f1.compute()
        val_auc = self.val_auc.compute()
        self.log('val_acc', val_acc)
        self.log('val_f1', val_f1)
        self.log('val_auc', val_auc)
        self.val_accuracy.reset()
        self.val_f1.reset()
        self.val_auc.reset()
        for t in self.thresholds_keys:
            val_f1 = self.val_f1_thresholds[t].compute()
            val_recall = self.val_recall_thresholds[t].compute()
            val_precision = self.val_precision_thresholds[t].compute()
            self.log(f'val_f1_threshold_{t}', val_f1)
            self.log(f'val_recall_threshold_{t}', val_recall)
            self.log(f'val_precision_threshold_{t}', val_precision)
            self.val_f1_thresholds[t].reset()
            self.val_recall_thresholds[t].reset()

    def configure_optimizers(self):
        # for now we only support Adam
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class NNModel(BaseModel):
    """
    Base class for all models.
    """
    def __init__(self, **kwargs):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        super().__init__(**kwargs)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model. This method must be implemented by the subclass with this signature.
        :param x:
        :return:
        """
        pass

    @classmethod
    def from_config(cls, cfg, dataset: TimeSeriesDataSet = None):
        """
        Create a model from a config and pytorch-forecasting dataset. This method must be implemented by the subclass.
        :param dataset:
        :param cfg:
        :return: A dict containing the arguments to the constructor of the model
        """
        pass


@register_model
class DebugModel(NNModel):
    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers, **kwargs):
        super().__init__(**kwargs)
        # input layer
        module_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        # hidden layers
        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        # output layer
        module_list.append(nn.Linear(hidden_size, output_size))

        self.sequential = nn.Sequential(*module_list)

    @classmethod
    def from_config(cls, cfg, dataset: TimeSeriesDataSet = None):
        kwargs = {
            "loss": torch.nn.MSELoss()
        }
        # convert the config to a dictionary and update the kwargs
        kwargs.update(OmegaConf.to_container(cfg.MODEL.KWARGS, resolve=True, enum_to_str=True))
        return super().from_dataset(dataset=dataset, **kwargs)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        network_input = x["encoder_cont"].squeeze(-1)
        prediction = self.sequential(network_input)

        # rescale predictions into target space
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])

        # We need to return a dictionary that at least contains the prediction
        # The parameter can be directly forwarded from the input.
        # The conversion to a named tuple can be directly achieved with the `to_network_output` function.
        return self.to_network_output(prediction=prediction)
