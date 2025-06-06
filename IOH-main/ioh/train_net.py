import os
from pathlib import Path
import argparse
from config.config import load_config, save_config
from dataset.dataset import get_datasets, get_test_dataset
from modeling.model import load_model_with_time_series, load_model, load_model_from_checkpoint_with_time_series
from pytorch_forecasting import TimeSeriesDataSet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import wandb
import shutil
from torch.utils.data import DataLoader
import json
import time
# make sure all datasets are registered
from __init__ import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="../exp_configs/debug_runs.yaml",
                        help="path to config file")
    parser.add_argument('--resume', default=False, action='store_true', help="resume training from last checkpoint of the experiment")
    parser.add_argument('--test_only', default=False, action='store_true', help="resume training") # TBD
    # Add a catch-all for updating the config dictionary
    parser.add_argument('updates', nargs='*', help='Key-value pairs to update the config file')
    parser.add_argument('--base-config', type=str, default="../exp_configs/base_configs.yaml",
                        help="path to base config file")
    parser.add_argument('--wandb-config', type=str, default="../exp_configs/wnb_configs.yaml",
                        help="path to base config file")
    parser.add_argument('--wandb-proj', type=str, default="default", help="wandb project name")
    parser.add_argument('--wandb-name', type=str, default="default", help="wandb run name")
    return parser.parse_args()


def create_exp(cfg, exp_dir):
    # check if the experiment directory exists
    if Path(exp_dir).exists():
        shutil.rmtree(exp_dir)  # TBD. For now, we will always overwrite the directory
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    # create config.yaml file in the experiment directory
    save_config(cfg, os.path.join(exp_dir, "config.yaml"))


def main():
    args = get_args()
    # read config file using hydra
    cfg = load_config(args.config, args.updates, args.base_config)
    if cfg.CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
    exp_name = f"{cfg.EXP_NAME} {time.strftime('%Y-%m-%d %H:%M:%S')}"
    exp_dir = str(os.path.join(cfg.EXP_BASE_DIR, exp_name))
    # in case we are not resuming, create a new experiment directory
    if not args.resume and cfg.CREATE_EXP_DIR and cfg.WNB.ENABLE:
        create_exp(cfg, exp_dir)
    # log to wandb using API key setting the WANDB_API_KEY environment variable
    if cfg.WNB.ENABLE:
        if not os.path.exists(args.wandb_config):
            raise FileNotFoundError(f"Wandb config file not found at {args.wandb_config}")
        wnb_cfg = OmegaConf.load(args.wandb_config)
        os.environ["WANDB_API_KEY"] = wnb_cfg.KEY
        # initialize wandb server
        wandb.init(project=wnb_cfg.PROJECT,
                   name=exp_name,
                   group=wnb_cfg.GROUP,
                   job_type=wnb_cfg.JOB_TYPE,
                   config=dict(cfg))
        # create a wandb logger for pytorch-lightning
        wandb_logger = WandbLogger(
            log_model='all',
            checkpoint_name=f'nature-{wandb.run.id}',
        )
    else:
        wandb_logger = False
    # run training or testing
    if args.test_only:
        # TBD. This test flow is not implemented in detail yet
        test_set = get_test_dataset(cfg)
        test_dataloader = DataLoader(test_set, batch_size=cfg.EVAL.TEST_BATCH_SIZE, shuffle=False)

        model = load_model_from_checkpoint_with_time_series(cfg, cfg.CHECKPOINT.TEST_CKP, test_set)
        # Create the trainer with WandbLogger and checkpoint callback
        trainer = pl.Trainer(
            accelerator=cfg.DEVICE.ACCELERATOR,
            devices=cfg.DEVICE.DEVICES,
            logger=wandb_logger
        )
        # test the model
        test_results = trainer.test(model, dataloaders=test_dataloader)
        ckp_name = Path(cfg.CHECKPOINT.TEST_CKP).name.split(".")[0]
        test_res_dir = os.path.join(exp_dir, f"test_results_{ckp_name}")
        with open(os.path.join(test_res_dir, "results.json"), 'w') as f:
            json.dump(test_results, f)
    else:
        # load dataset and model
        train_set, val_set = get_datasets(cfg)
        if issubclass(type(train_set), TimeSeriesDataSet):
            model = load_model_with_time_series(cfg, train_set, resume=args.resume)
        else:
            model = load_model(cfg, resume=args.resume)
        # set the seed
        if cfg.SEED:
            pl.seed_everything(cfg.SEED, workers=True)
        # Set up checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(exp_dir, "checkpoints"),
            filename=cfg.CHECKPOINT.FILENAME,
            save_top_k=cfg.CHECKPOINT.SAVE_TOP_K,
            monitor=cfg.CHECKPOINT.MONITOR,
            mode=cfg.CHECKPOINT.MODE
        )

        # Create the trainer with WandbLogger and checkpoint callback
        trainer = pl.Trainer(
            max_epochs=cfg.OPT.MAX_EPOCHS,
            accelerator=cfg.DEVICE.ACCELERATOR,
            strategy=cfg.DEVICE.STRATEGY,
            devices=cfg.DEVICE.DEVICES,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            default_root_dir=exp_dir,
            check_val_every_n_epoch=cfg.EVAL.EVERY_N_EPOCH,
            resume_from_checkpoint=cfg.CHECKPOINT.RESUME,
            # deterministic=bool(cfg.SEED) # TBD. This makes the training slower if set to True, maybe not needed
        )

        train_dataloader = DataLoader(train_set,
                                      batch_size=cfg.OPT.BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=cfg.WORKERS)
        val_dataloader = DataLoader(val_set,
                                    batch_size=cfg.EVAL.TEST_BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=cfg.WORKERS)

        # Train the model
        trainer.fit(model=model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
