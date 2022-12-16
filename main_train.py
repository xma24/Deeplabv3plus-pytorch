import argparse
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
import torch.onnx
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import QuantizationAwareTraining
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from rich import print
from termcolor import colored, cprint

from expr_setting import ExprSetting

cudnn.benchmark = True


matplotlib.use("Agg")

plt.style.use("ggplot")

warnings.filterwarnings("ignore")

from configs.config_v0 import (
    DataConfig,
    NetConfig,
    TestingConfig,
    TrainingConfig,
    ValidationConfig,
)

# def parse_args():
#     parser = argparse.ArgumentParser(description="Train a segmentor")
#     parser.add_argument(
#         "--config", default="config_default.yaml", help="train config file path"
#     )
#     args = parser.parse_args()
#     return args


if __name__ == "__main__":

    # train_args = parse_args()
    expr_setting = ExprSetting()

    lr_logger, model_checkpoint, early_stop, model_class, dataloader_class = (
        expr_setting.lr_logger,
        expr_setting.model_checkpoint,
        expr_setting.early_stop,
        expr_setting.model_class,
        expr_setting.dataloader_class,
    )

    os.makedirs(DataConfig.work_dirs, exist_ok=True)

    seed_everything(DataConfig.random_seed)

    if DataConfig.classes == "none":
        DataConfig.classes = np.arange(DataConfig.num_classes)

    if isinstance(TrainingConfig.num_gpus, int):
        num_gpus = TrainingConfig.num_gpus
    elif TrainingConfig.num_gpus == "autocount":
        TrainingConfig.num_gpus = torch.cuda.device_count()
        num_gpus = TrainingConfig.num_gpus
    else:
        gpu_list = TrainingConfig.num_gpus.split(",")
        num_gpus = len(gpu_list)

    if TrainingConfig.logger_name == "neptune":
        print("Not implemented")
        exit(0)
    elif TrainingConfig.logger_name == "csv":
        own_logger = CSVLogger(DataConfig.logger_root)
    elif TrainingConfig.logger_name == "wandb":
        own_logger = WandbLogger(project=TrainingConfig.wandb_name)
    else:
        own_logger = CSVLogger(DataConfig.logger_root)

    print("num of gpus: {}".format(num_gpus))

    train_dataloader = dataloader_class.get_train_dataloader(num_gpus=num_gpus)
    print("train_dataloader: {}".format(len(train_dataloader)))

    num_train_batches = len(train_dataloader) // num_gpus
    if (len(train_dataloader) // num_gpus) // 6 >= 10:
        num_train_batches = 10
    else:
        num_train_batches = int((len(train_dataloader) // num_gpus) // 6) - 1

    # """
    #     - Get calibration dataloder
    # """
    # cal_dataloader = dataloader_class.get_cal_dataloader()
    # print("cal_dataloader: {}".format(len(cal_dataloader)))
    # config["cal_dataloader"] = cal_dataloader

    # """
    #     - Get sample data
    # """
    # single_batch = next(iter(train_dataloader))
    # config["sample_data"] = single_batch
    # # config["sample_data"] = single_batch[0]
    # # config["sample_labels"] = single_batch[1]
    # # print("sample data: {}, sample lables: {}".format(config["sample_data"].shape, config["sample_labels"].shape))

    val_dataloader = dataloader_class.get_val_dataloader(num_gpus=num_gpus)
    print("val_dataloader: {}".format(len(val_dataloader)))

    # config["val_dataloader"] = val_dataloader

    num_val_batches = len(val_dataloader) // num_gpus
    if len(val_dataloader) % num_gpus == 0:
        num_val_batches = (len(val_dataloader) // num_gpus) - 1
    else:
        num_val_batches = len(val_dataloader) // num_gpus

    model = model_class(TrainingConfig.batch_size, NetConfig.lr, own_logger)
    # model = model_class(own_logger)

    # print(">>> config: {}".format(config))

    if TrainingConfig.resume != "none":
        model = model_class.load_from_checkpoint(
            TrainingConfig.resume, logger=own_logger
        )
        print(">>> Using checkpoint from pretrained models")
        # model = model.load_state_dict(torch.load(config["TRAIN"]["RESUME"]))

    """
        - The setting of pytorch lightning Trainer:
            (https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/trainer/trainer.py)
    """
    if TrainingConfig.cpus:
        print("using CPUs to do experiments ... ")
        trainer = pl.Trainer(
            num_nodes=TrainingConfig.num_nodes,
            # precision=config["TRAIN"]["PRECISION"],
            accelerator="cpu",
            strategy=DDPStrategy(find_unused_parameters=True),
            profiler="pytorch",
            logger=own_logger,
            callbacks=[lr_logger, model_checkpoint, early_stop],
            log_every_n_steps=1,
            # track_grad_norm=1,
            progress_bar_refresh_rate=TrainingConfig.progress_bar_refresh_rate,
            # resume_from_checkpoint=config["TRAIN"]["CKPT_PATH"],
            # sync_batchnorm=True if num_gpus > 1 else False,
            # plugins=DDPPlugin(find_unused_parameters=False),
            check_val_every_n_epoch=ValidationConfig.val_interval,
            auto_scale_batch_size="binsearch",
            replace_sampler_ddp=False,
        )
    elif TrainingConfig.lr_find:
        print("using GPUs and lr_find to do experiments ... ")
        trainer = pl.Trainer(
            # devices=TrainingConfig.num_gpus,
            devices=1,
            # gpus=torch.cuda.device_count(),  ### let the code to detect the number of gpus to use
            num_nodes=TrainingConfig.num_nodes,
            precision=TrainingConfig.precision,
            # accelerator=TrainingConfig.accelerator,
            # strategy=DDPStrategy(find_unused_parameters=True),
            # (strategy="ddp", accelerator="gpu", devices=4);(strategy=DDPStrategy(find_unused_parameters=False), accelerator="gpu", devices=4);
            # (strategy="ddp_spawn", accelerator="auto", devices=4); (strategy="deepspeed", accelerator="gpu", devices="auto"); (strategy="ddp", accelerator="cpu", devices=3);
            # (strategy="ddp_spawn", accelerator="tpu", devices=8); (accelerator="ipu", devices=8);
            strategy="ddp_spawn",
            accelerator="auto",
            # profiler="pytorch",  # "simple", "advanced","pytorch"
            logger=own_logger,
            callbacks=[lr_logger, model_checkpoint, early_stop],
            log_every_n_steps=1,
            # track_grad_norm=1,
            progress_bar_refresh_rate=TrainingConfig.progress_bar_refresh_rate,
            max_epochs=TrainingConfig.max_epochs,
            # resume_from_checkpoint=config["TRAIN"]["CKPT_PATH"],
            # sync_batchnorm=True if num_gpus > 1 else False,
            # plugins=DDPPlugin(find_unused_parameters=False),
            check_val_every_n_epoch=ValidationConfig.val_interval,
            auto_scale_batch_size="binsearch",
            # """>>> C: mmlab如果给出了sampler可以选择使用mmlab的sampler,这里replace_sampler_ddp需要设置成False; """
            replace_sampler_ddp=False,
            auto_lr_find=True,
        )
    else:
        print("using GPUs to do experiments ... ")
        trainer = pl.Trainer(
            devices=TrainingConfig.num_gpus,
            # gpus=torch.cuda.device_count(),  ### let the code to detect the number of gpus to use
            num_nodes=TrainingConfig.num_nodes,
            precision=TrainingConfig.precision,
            accelerator=TrainingConfig.accelerator,
            strategy=DDPStrategy(find_unused_parameters=True),
            # (strategy="ddp", accelerator="gpu", devices=4);(strategy=DDPStrategy(find_unused_parameters=False), accelerator="gpu", devices=4);
            # (strategy="ddp_spawn", accelerator="auto", devices=4); (strategy="deepspeed", accelerator="gpu", devices="auto"); (strategy="ddp", accelerator="cpu", devices=3);
            # (strategy="ddp_spawn", accelerator="tpu", devices=8); (accelerator="ipu", devices=8);
            # profiler="pytorch",  # "simple", "advanced","pytorch"
            logger=own_logger,
            callbacks=[lr_logger, model_checkpoint, early_stop],
            log_every_n_steps=1,
            # track_grad_norm=1,
            progress_bar_refresh_rate=TrainingConfig.progress_bar_refresh_rate,
            max_epochs=TrainingConfig.max_epochs,
            # resume_from_checkpoint=config["TRAIN"]["CKPT_PATH"],
            # sync_batchnorm=True if num_gpus > 1 else False,
            # plugins=DDPPlugin(find_unused_parameters=False),
            check_val_every_n_epoch=ValidationConfig.val_interval,
            auto_scale_batch_size="binsearch",
            # """>>> C: mmlab如果给出了sampler可以选择使用mmlab的sampler,这里replace_sampler_ddp需要设置成False; """
            # replace_sampler_ddp=False,
        )

    # print("==>> model.hparams: ", model.hparams)

    # # lr_Finder = trainer.tuner.lr_find(model)
    # lr_Finder = trainer.lr_find(model, train_dataloader)

    # fig = lr_Finder.plot(suggest=True)
    # plt.savefig("lr_finder.png")

    # model.hparams.learning_rate = lr_Finder.suggestion()

    # trainer.fit(model, train_dataloaders=train_dataloader)

    # trainer.test(model, dataloaders=val_dataloader)
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    # trainer.test(model, dataloaders=val_dataloader)
