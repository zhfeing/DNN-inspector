import os
import random
import functools
import itertools
import logging
import signal
import traceback
import argparse
from typing import List, Dict, Any

import torch
import torch.backends.cudnn
import torch.multiprocessing as mp

from cv_lib.logger import MultiProcessLoggerListener
import cv_lib.utils as cv_utils
from cv_lib.config_parsing import get_cfg

from utils import ModelWrapper
from utils.task_scheduler import GPUTaskScheduler
from driver.data import build_train_dataloader
from driver.loss import get_loss_fn
from driver.eval import Evaluation
from dnn_inspector.loss_landscape.direction import setup_direction
from dnn_inspector.loss_landscape.surface import setup_coordinates
from dnn_inspector.loss_landscape.model_setter import set_weights


START_METHOD = "spawn"


def eval_at_coordinate(
    gpu_id: int,
    task_id: int,
    task_args: Dict[str, Any]
):
    ################################################################################
    # Initialization
    # setup process root logger
    logger = logging.getLogger(f"worker_{task_id}")

    data_cfg: Dict[str, Any] = task_args["data_cfg"]
    loss_cfg: Dict[str, Any] = task_args["loss_cfg"]
    x_coordinate: float = task_args["x_coordinate"]
    y_coordinate: float = task_args["y_coordinate"]
    args: bool = task_args["args"]

    # read direction
    direction_fp = os.path.join(args.save_path, "direction.pth")
    directions = torch.load(direction_fp, map_location="cpu")

    eval_fp = os.path.join(args.save_path, "eval", f"{task_id}.pkl")
    if not args.override and os.path.isfile(eval_fp):
        return

    # set cuda
    torch.backends.cudnn.benchmark = True
    logger.info("Use GPU: %d for testing", gpu_id)
    device = torch.device("cuda:{}".format(gpu_id))
    torch.cuda.set_device(device)

    # get model
    model: torch.jit.ScriptModule = torch.jit.load(args.jit, map_location="cpu")
    set_weights(
        model,
        x_step=x_coordinate,
        y_step=y_coordinate,
        **directions
    )
    model = ModelWrapper(model, True)
    model.to(device)
    loss_fn = get_loss_fn(loss_cfg).to(device)

    # get dataloader
    logger.info("Building dataset...")

    train_loader, val_loader, _, _ = build_train_dataloader(
        data_cfg,
        num_worker=args.num_worker,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        drop_last=False
    )
    if args.use_train:
        dataloader = train_loader
    else:
        dataloader = val_loader

    evaluator = Evaluation(
        loss_fn=loss_fn,
        val_loader=dataloader,
        loss_weights=loss_cfg["weight_dict"],
        device=device
    )
    result = evaluator(model)
    torch.save(result, eval_fp)


def logger_constructor(logger_fp: str):
    logger_dir = os.path.dirname(logger_fp)
    os.makedirs(logger_dir, exist_ok=True)
    logger = cv_utils.get_root_logger(
        level=logging.INFO,
        name=None,
        logger_fp=logger_fp
    )
    return logger, logger_fp


def plot(args):
    eval_path = os.path.join(args.save_path, "eval")
    eval_files = os.listdir(eval_path)
    for filename in eval_files:
        fp = os.path.join(eval_path, filename)
        result = torch.load(fp, map_location="cpu")
        print(result)


def main(args):
    # multi-process logger
    logger_fp = os.path.join(args.save_path, "run.log")
    logger_listener = MultiProcessLoggerListener(
        functools.partial(logger_constructor, logger_fp=logger_fp),
        START_METHOD
    )
    logger = logger_listener.get_logger()

    # setup global variables
    # split configs
    data_cfg: Dict[str, Any] = get_cfg(args.data_cfg)
    global_cfg = get_cfg(args.cfg_fp)
    loss_cfg: Dict[str, Any] = global_cfg["loss"]
    plot_cfg: Dict[str, Any] = global_cfg["plot"]

    # make determinstic
    if args.seed is not None:
        cv_utils.make_deterministic(args.seed)

    # create model
    logger.info("Building model...")
    model: torch.jit.ScriptModule = torch.jit.load(args.jit, map_location="cpu")

    # setup direction
    direction_fp = os.path.join(args.save_path, "direction.pth")
    setup_direction(
        direction_fp=direction_fp,
        model=model,
        override=args.override,
        **plot_cfg["direction"]
    )
    # setup surface
    coordinates_fp = os.path.join(args.save_path, "coordinates.pth")
    coordinates = setup_coordinates(
        coordinates_fp=coordinates_fp,
        coordinate_cfg=plot_cfg["coordinate"],
        override=args.override
    )
    # create tasks
    task_base_args = {
        "data_cfg": data_cfg,
        "loss_cfg": loss_cfg,
        "args": args
    }
    iters = list(itertools.product(coordinates["x_coordinate"], coordinates["y_coordinate"]))
    tasks = list()
    for task_id, (x, y) in enumerate(iters):
        task_args = dict(
            x_coordinate=x.item(),
            y_coordinate=y.item(),
            **task_base_args
        )
        task = functools.partial(
            eval_at_coordinate,
            task_id=task_id,
            task_args=task_args,
        )
        tasks.append(task)
    # shuffle tasks for faster visualization
    random.shuffle(tasks)
    # assign tasks
    n_gpus = torch.cuda.device_count()
    logger.info("Detected %d gpus", n_gpus)
    scheduler = GPUTaskScheduler(
        tasks,
        n_gpus=n_gpus,
        logger_queue=logger_listener.queue
    )
    process_pool: List[mp.Process]

    def kill_handler(signum, frame):
        logger.warning("Got kill signal %d, frame:\n%s\nExiting...", signum, frame)
        for process in process_pool:
            try:
                logger.info("Killing subprocess: %d-%s...", process.pid, process.name)
                process.kill()
            except:
                pass
        logger.info("Stopping multiprocess logger...")
        logger_listener.stop()
        exit(1)

    logger.info("Registering kill handler")
    signal.signal(signal.SIGINT, kill_handler)
    signal.signal(signal.SIGHUP, kill_handler)
    signal.signal(signal.SIGTERM, kill_handler)
    logger.info("Registered kill handler")

    try:
        logger.info("Starting evaluating on every coordinate")
        scheduler.start()
        process_pool = scheduler.process_pool
        scheduler.join()
        logger.info("Evaluation Done.")
    except Exception as e:
        tb = traceback.format_exc()
        logger.critical("While running, exception:\n%s\nTraceback:\n%s", str(e), str(tb))
    finally:
        # make sure listener is stopped
        logger_listener.stop()

    plot(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_fp", type=str)
    parser.add_argument("--data_cfg", type=str)
    parser.add_argument("--jit", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_train", action="store_true")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "eval"), exist_ok=True)
    main(args)
