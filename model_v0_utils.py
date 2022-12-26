import torchmetrics

from configs.config_v0 import DataConfig, DeeplabV3PlusConfig
from models.models_builder import build_backbone


def get_segmentation_metrics(num_classes, ignore=False):
    """>>> C*: https://torchmetrics.readthedocs.io/en/stable/references/modules.html#"""
    if ignore:

        train_iou = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
        )
        val_iou = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
        )
        test_iou = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
        )

        train_precision = torchmetrics.Precision(
            task="multiclass",
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )

        val_precision = torchmetrics.Precision(
            task="multiclass",
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )

        test_precision = torchmetrics.Precision(
            task="multiclass",
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )

        val_recall = torchmetrics.Recall(
            task="multiclass",
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )

        test_recall = torchmetrics.Recall(
            task="multiclass",
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )
    else:
        train_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes, average="none"
        )
        val_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes, average="none"
        )

        test_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes, average="none"
        )

        train_precision = torchmetrics.Precision(
            task="multiclass",
            num_classes=num_classes,
            average="none",
            mdmc_average="global",
        )

        val_precision = torchmetrics.Precision(
            task="multiclass",
            num_classes=num_classes,
            average="none",
            mdmc_average="global",
        )

        test_precision = torchmetrics.Precision(
            task="multiclass",
            num_classes=num_classes,
            average="none",
            mdmc_average="global",
        )

        val_recall = torchmetrics.Recall(
            task="multiclass",
            num_classes=num_classes,
            average="none",
            mdmc_average="global",
        )

        test_recall = torchmetrics.Recall(
            task="multiclass",
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
        # pretrained=DeeplabV3PlusConfig.pretrained,
        backbone=dict(
            type="ResNetV1c",
            depth=DeeplabV3PlusConfig.depth,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            norm_eval=False,
            style="pytorch",
            contract_dilation=True,
            # pretrained=DeeplabV3PlusConfig.pretrained,
            # pretrained=None,
        ),
        decode_head=dict(
            type="DepthwiseSeparableASPPHead",
            in_channels=DeeplabV3PlusConfig.decode_head_in_channels,
            # in_index=3,
            channels=DeeplabV3PlusConfig.decode_head_channels,
            dilations=(1, 12, 24, 36),
            c1_in_channels=DeeplabV3PlusConfig.decode_head_c1_in_channels,
            c1_channels=DeeplabV3PlusConfig.decode_head_c1_channels,
            dropout_ratio=0.1,
            num_classes=DataConfig.num_classes,
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
            in_channels=DeeplabV3PlusConfig.auxiliary_head_in_channels,
            # in_index=2,
            channels=DeeplabV3PlusConfig.auxiliary_head_channels,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=DataConfig.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
    )

    model = build_backbone(cfg)

    return model


def get_deeplabv3plus_backbone(model):
    backbone = model.backbone

    return backbone


def get_deeplabv3plus_heads(model):
    decode_head = model.decode_head
    auxiliary_head = model.auxiliary_head
    return decode_head, auxiliary_head

