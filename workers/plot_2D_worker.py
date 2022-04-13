import os
import logging
from logging.handlers import QueueHandler
from typing import Dict, Any
import yaml
import copy

import torch
import torch.nn as nn
import torch.cuda
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.backends.cudnn

import cv_lib.utils as utils
import cv_lib.distributed.utils as dist_utils

from vit_mutual.models import get_model
from vit_mutual.loss import get_loss_fn
from vit_mutual.eval import Evaluation
import vit_mutual.utils as vit_utils

from loss_landscape.utils import DistLaunchArgs, LogArgs
from loss_landscape.direction import setup_direction
from loss_landscape.surface import setup_coordinates
from loss_landscape.model_setter import set_weights
from workers.plot_contour import plot_contour
from data import build_dataset


class PlotWorker:
    def __init__(
        self,
        coordinate_fp: str,
        direction_fp: str,
        eval_fp: str,
        save_path: str,
        data_loader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        evaluator: Evaluation,
        plot_cfg: Dict[str, Any],
        device: torch.device
    ):
        self.rank = dist_utils.get_rank()
        self.world_size = dist_utils.get_world_size()
        self.logger = logging.getLogger("plot_worker_{}".format(self.rank))

        self.coordinate_fp = coordinate_fp
        coordinates: Dict[str, torch.Tensor] = torch.load(coordinate_fp, map_location=device)
        self.x_coordinate = coordinates["x_coordinate"]
        self.y_coordinate = coordinates["y_coordinate"]
        self.run_groups = len(self.x_coordinate), len(self.y_coordinate)

        self.direction_fp = direction_fp
        directions = torch.load(direction_fp, map_location=device)
        self.x_direction = directions["x_direction"]
        self.y_direction = directions["y_direction"]

        self.save_path = save_path
        self.eval_fp = eval_fp

        self.model = model
        self.model_state_dict = copy.deepcopy(self.model.state_dict())
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.plot_cfg = plot_cfg

        self.loss_map = torch.zeros(size=self.run_groups, device=device)
        self.acc_map = torch.zeros_like(self.loss_map)
        self.device = device

        # assign tasks
        x_steps = torch.arange(self.run_groups[0])
        y_steps = torch.arange(self.run_groups[1])
        self.total_tasks = torch.cartesian_prod(x_steps, y_steps)
        self.assignments = torch.arange(0, self.total_tasks.shape[0]) % self.world_size
        self.tasks = self.total_tasks[self.assignments == self.rank]
        self.logger.info("Total %d tasks, assigned %d tasks", len(self.total_tasks), len(self.tasks))

    def eval(self):
        for task_id, assigned_id in enumerate(self.tasks):
            self.model.load_state_dict(self.model_state_dict)
            x_step = self.x_coordinate[assigned_id[0]]
            y_step = self.y_coordinate[assigned_id[1]]
            self.logger.info("step: (%f, %f)", x_step.item(), y_step.item())
            # modify model weights
            set_weights(self.model, self.x_direction, self.y_direction, x_step, y_step)
            # evaluate
            result = self.evaluator(self.model)
            loss = result["loss"]
            acc1 = result["acc"][1]
            self.logger.info("task: %d|%d, loss: %.5f, acc: %.4f", task_id, len(self.tasks), loss.item(), acc1.item())
            self.loss_map[assigned_id[0], assigned_id[1]] = loss
            self.acc_map[assigned_id[0], assigned_id[1]] = acc1

        self.logger.info("Waiting for all task done...")
        dist_utils.barrier()
        # reduce
        self.loss_map = dist_utils.reduce_tensor(self.loss_map, average=False)
        self.acc_map = dist_utils.reduce_tensor(self.acc_map, average=False)
        res = {
            "loss": self.loss_map,
            "acc": self.acc_map
        }
        if dist_utils.is_main_process():
            torch.save(res, self.eval_fp)
            self.logger.info("Saving eval file done")

    def __call__(self):
        try:
            res = torch.load(self.eval_fp, map_location=self.device)
            assert self.loss_map.shape == res["loss"].shape
            assert self.acc_map.shape == res["acc"].shape
            self.loss_map = res["loss"]
            self.acc_map = res["acc"]
        except Exception as e:
            self.logger.info("Load eval file failed: {}, re-generate eval file".format(e))
            self.eval()
        if dist_utils.is_main_process():
            plot_contour(
                coordinates_fp=self.coordinate_fp,
                eval_fp=self.eval_fp,
                save_path=self.save_path,
                **self.plot_cfg
            )


def plot_2D_worker(
    gpu_id: int,
    launch_args: DistLaunchArgs,
    log_args: LogArgs,
    global_cfg: Dict[str, Any]
):
    ################################################################################
    # Initialization
    # setup process root logger
    if launch_args.distributed:
        root_logger = logging.getLogger()
        handler = QueueHandler(log_args.logger_queue)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False

    # split configs
    data_cfg: Dict[str, Any] = global_cfg["dataset"]
    train_cfg: Dict[str, Any] = global_cfg["training"]
    model_cfg: Dict[str, Any] = global_cfg["model"]
    loss_cfg: Dict[str, Any] = global_cfg["loss"]
    plot_cfg: Dict[str, Any] = global_cfg["plot"]

    distributed = launch_args.distributed
    # get current rank
    current_rank = launch_args.rank
    if distributed:
        if launch_args.multiprocessing:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            current_rank = launch_args.rank * launch_args.ngpus_per_node + gpu_id
        dist.init_process_group(
            backend=launch_args.backend,
            init_method=launch_args.master_url,
            world_size=launch_args.world_size,
            rank=current_rank
        )

    assert dist_utils.get_rank() == current_rank, "code bug"
    # set up process logger
    logger = logging.getLogger("worker_rank_{}".format(current_rank))

    if current_rank == 0:
        logger.info("Starting with configs:\n%s", yaml.dump(global_cfg))

    # make determinstic
    if launch_args.seed is not None:
        seed = launch_args.seed + current_rank
        logger.info("Initial rank %d with seed: %d", current_rank, seed)
        utils.make_deterministic(seed)
    # set cuda
    torch.backends.cudnn.benchmark = True
    logger.info("Use GPU: %d for training", gpu_id)
    device = torch.device("cuda:{}".format(gpu_id))
    # IMPORTANT! for distributed training (reduce_all_object)
    torch.cuda.set_device(device)

    # get dataloader
    logger.info("Building dataset...")

    train_loader, n_classes = build_dataset(data_cfg, train_cfg, use_train_set=train_cfg.get("use_train", True))
    # create model
    logger.info("Building model...")
    model = get_model(model_cfg, n_classes)
    logger.info(
        "Built model with %d parameters, %d trainable parameters",
        utils.count_parameters(model, include_no_grad=True),
        utils.count_parameters(model, include_no_grad=False)
    )
    if train_cfg.get("pre_train", None) is not None:
        vit_utils.load_pretrain_model(
            pretrain_fp=train_cfg["pre_train"],
            model=model,
        )
        logger.info("Loaded pretrain model: %s", train_cfg["pre_train"])
    model.to(device)
    loss_fn = get_loss_fn(loss_cfg).to(device)
    # setup direction
    direction_fp = os.path.join(log_args.logdir, "direction.pth")
    if dist_utils.is_main_process():
        setup_direction(
            direction_fp=direction_fp,
            model=model,
            override=launch_args.override,
            **plot_cfg["direction"]
        )
    # setup surface
    coordinates_fp = os.path.join(log_args.logdir, "coordinates.pth")
    if dist_utils.is_main_process():
        setup_coordinates(
            coordinates_fp=coordinates_fp,
            override=launch_args.override,
            coordinate_cfg=plot_cfg["coordinate"],
            device=device
        )
    eval_fp = os.path.join(log_args.logdir, "eval.pth")
    # wait for the main process writing files
    dist_utils.barrier()
    evaluator = Evaluation(
        loss_fn=loss_fn,
        val_loader=train_loader,
        loss_weights=loss_cfg["weight_dict"],
        device=device
    )
    worker = PlotWorker(
        coordinate_fp=coordinates_fp,
        direction_fp=direction_fp,
        save_path=log_args.logdir,
        eval_fp=eval_fp,
        data_loader=train_loader,
        model=model,
        loss_fn=loss_fn,
        evaluator=evaluator,
        plot_cfg=plot_cfg["cfg"],
        device=device
    )
    worker()
