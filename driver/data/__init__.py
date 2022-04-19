import logging
import copy
from typing import Dict, Any

from torch.utils.data import DataLoader

import cv_lib.classification.data as cls_data
import cv_lib.distributed.utils as dist_utils
from cv_lib.distributed.sampler import get_train_sampler, get_val_sampler

from .aug import get_data_aug


def build_eval_dataset(data_cfg: Dict[str, Any]):
    # get dataloader
    data_cfg = copy.deepcopy(data_cfg)
    name = data_cfg.pop("name")
    name = name.split("=")[0]
    dataset = cls_data.__REGISTERED_DATASETS__[name]
    root = data_cfg.pop("root")
    val_data_cfg = data_cfg.pop("val")
    val_aug = get_data_aug(name, "val")
    data_cfg.pop("train", None)

    val_dataset: cls_data.ClassificationDataset = dataset(
        root=root,
        augmentations=val_aug,
        **val_data_cfg,
        **data_cfg
    )
    n_classes = val_dataset.n_classes

    logger = logging.getLogger("build_eval_dataset")
    if dist_utils.is_main_process():
        logger.info(
            "Loaded %s dataset with %d val examples, %d classes",
            name, len(val_dataset), n_classes
        )
    dist_utils.barrier()
    img_channels = val_dataset.img_channels
    return val_dataset, n_classes, img_channels


def build_train_dataset(data_cfg: Dict[str, Any]):
    logger = logging.getLogger("build_train_dataset")
    # get dataloader
    train_aug = get_data_aug(data_cfg["name"], "train")
    val_aug = get_data_aug(data_cfg["name"], "val")
    train_dataset, val_dataset, n_classes, img_channels = cls_data.get_dataset(
        data_cfg,
        train_aug,
        val_aug
    )
    if dist_utils.is_main_process():
        logger.info(
            "Loaded %s dataset with %d train examples, %d val examples, %d classes",
            data_cfg["name"], len(train_dataset), len(val_dataset), n_classes
        )
    dist_utils.barrier()
    return train_dataset, val_dataset, n_classes, img_channels


def build_train_dataloader(
    data_cfg: Dict[str, Any],
    num_worker: int,
    train_batch_size: int,
    val_batch_size: int,
    drop_last: bool = True
):
    train_dataset, val_dataset, n_classes, img_channels = build_train_dataset(data_cfg)
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=get_train_sampler(False, train_dataset),
        num_workers=num_worker,
        pin_memory=True,
        batch_size=train_batch_size,
        drop_last=drop_last
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        sampler=get_val_sampler(False, val_dataset),
        num_workers=num_worker,
        pin_memory=True,
        batch_size=val_batch_size,
        drop_last=drop_last
    )

    return train_loader, val_loader, n_classes, img_channels

