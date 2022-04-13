import logging
from typing import Dict, Any

import torch.utils.data as data

import cv_lib.classification.data as cls_data
import cv_lib.distributed.utils as dist_utils

from .aug import get_data_aug


def build_dataset(
    data_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    use_train_set: bool = True
):
    logger = logging.getLogger("build_dataset")
    # get dataloader
    train_aug = get_data_aug(data_cfg["name"], "train")
    train_dataset, val_dataset, n_classes = cls_data.get_dataset(
        data_cfg,
        train_aug,
        train_aug
    )
    if use_train_set:
        dataset = train_dataset
    else:
        dataset = val_dataset
    if dist_utils.is_main_process():
        logger.info(
            "Loaded %s dataset with %d train examples, %d classes",
            data_cfg["name"], len(dataset), n_classes
        )
    sampler = data.SequentialSampler(dataset)
    train_bs = train_cfg["batch_size"]
    train_workers = train_cfg["num_workers"]
    loader = data.DataLoader(
        dataset,
        batch_size=train_bs,
        num_workers=train_workers,
        pin_memory=True,
        sampler=sampler,
    )
    logger.info(
        "Build dataset done\nTraining: %d imgs, %d batchs",
        len(dataset),
        len(loader),
    )
    return loader, n_classes

