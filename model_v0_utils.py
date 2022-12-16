import os
import shutil

import torch
import torchmetrics

from models.models_builder import build_backbone
from utils import csvlogger_start


def get_segmentation_metrics(num_classes, ignore=False):
    if ignore:
        train_iou = torchmetrics.JaccardIndex(
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
        )
        val_iou = torchmetrics.JaccardIndex(
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
        )
        test_iou = torchmetrics.JaccardIndex(
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
        )

        train_precision = torchmetrics.Precision(
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )

        val_precision = torchmetrics.Precision(
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )

        test_precision = torchmetrics.Precision(
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )

        val_recall = torchmetrics.Recall(
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )

        test_recall = torchmetrics.Recall(
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )
    else:
        train_iou = torchmetrics.JaccardIndex(num_classes=num_classes, average="none")
        val_iou = torchmetrics.JaccardIndex(num_classes=num_classes, average="none")

        test_iou = torchmetrics.JaccardIndex(num_classes=num_classes, average="none")

        train_precision = torchmetrics.Precision(
            num_classes=num_classes,
            average="none",
            mdmc_average="global",
        )

        val_precision = torchmetrics.Precision(
            num_classes=num_classes,
            average="none",
            mdmc_average="global",
        )

        test_precision = torchmetrics.Precision(
            num_classes=num_classes,
            average="none",
            mdmc_average="global",
        )

        val_recall = torchmetrics.Recall(
            num_classes=num_classes,
            average="none",
            mdmc_average="global",
        )

        test_recall = torchmetrics.Recall(
            num_classes=num_classes,
            average="none",
            mdmc_average="global",
        )
    return (
        train_iou,
        val_iou,
        test_iou,
        train_precision,
        val_precision,
        test_precision,
        val_recall,
        test_recall,
    )


def get_deeplabv3plus_model():
    cfg = dict(
        type="EncoderDecoder",
        backbone=dict(
            type="ResNetV1c",
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            norm_eval=False,
            style="pytorch",
            contract_dilation=True,
            pretrained="open-mmlab://resnet50_v1c",
        ),
        decode_head=dict(
            type="DepthwiseSeparableASPPHead",
            in_channels=2048,
            # in_index=3,
            channels=256,
            dilations=(1, 12, 24, 36),
            c1_in_channels=2048,
            c1_channels=48,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss",
                loss_name="loss_ce",
                use_sigmoid=False,
                loss_weight=1.0,
            ),
        ),
        auxiliary_head=dict(
            type="FCNHead",
            in_channels=1024,
            # in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
    )

    model = build_backbone(cfg)
    # print("==>> model: ", model)

    return model


def get_deeplabv3plus_backbone(model):
    backbone = model.backbone

    backbone_list = list(backbone.children())
    # print("==>> backbone_list: ", backbone_list)

    state_dict = torch.load("/data/SSD1/data/weights/resnet50_v1c-2cccc1ad.pth")[
        "state_dict"
    ]
    # print("==>> state_dict: ", state_dict.keys())

    backbone_state_dict = {}
    fc_state_dict = {}
    for k in list(state_dict.keys()):
        if k.startswith("fc."):
            fc_state_dict[k[len("fc.") :]] = state_dict[k]
        else:
            backbone_state_dict[k] = state_dict[k]
        del state_dict[k]
    backbone.load_state_dict(backbone_state_dict, strict=True)

    return backbone


def get_deeplabv3plus_heads(model):
    decode_head = model.decode_head
    auxiliary_head = model.auxiliary_head
    return decode_head, auxiliary_head


def init_expr_setup(config, batch_idx, current_epoch):
    if config["TRAIN"]["LOGGER"] == "csv":
        if current_epoch <= 2:
            if batch_idx == 0:
                print("config_path: {}".format(config["config_path"]))
        else:
            if batch_idx == 0:
                print(
                    "current_csv_version: {}, config_path: {}".format(
                        config["current_csv_version"],
                        config["config_path"],
                    )
                )

    if current_epoch == 1 and batch_idx == 0 and config["TRAIN"]["LOGGER"] == "csv":
        csv_logger_folder, config = csvlogger_start(config)

        # copy the config file to the cvs logger folder
        config_filename = config["config_path"].split("/")[-1]
        shutil.copyfile(
            config["config_path"],
            os.path.join(csv_logger_folder, config_filename),
        )
        model_path = config["NET"]["MODEL"] + ".py"
        shutil.copyfile(model_path, os.path.join(csv_logger_folder, model_path))

        dataset_path = config["DATASET"]["DATALOADER_NAME"] + ".py"
        shutil.copyfile(dataset_path, os.path.join(csv_logger_folder, dataset_path))

        middle_results_folder = os.path.join(csv_logger_folder, "middle_results")
        os.makedirs(middle_results_folder, exist_ok=True)
