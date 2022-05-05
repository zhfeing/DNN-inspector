from functools import partial
import logging
import signal
import traceback
from typing import List
import random
import time

import torch
import torch.multiprocessing as mp

from cv_lib.logger import MultiProcessLoggerListener
from cv_lib.utils import get_root_logger

from utils.task_scheduler import GPUTaskScheduler


START_METHOD = "spawn"


def task(gpu_id: int, task_id: int):
    logger = logging.getLogger(f"task_{task_id}")
    logger.info(f"task: {task_id} with gpu: {gpu_id}")
    s_time = random.random() * 10
    if task_id == 4:
        raise Exception("Fuck")
    device = torch.device(f"cuda:{gpu_id}")
    x = torch.rand(10, device=device)
    logger.info(str(x.device))
    time.sleep(s_time)


def logger_constructor():
    logger = get_root_logger(
        level=logging.INFO,
        name=None,
        logger_fp=None
    )
    return logger, None


def main():
    # multi-process logger
    logger_listener = MultiProcessLoggerListener(logger_constructor, START_METHOD)
    logger = logger_listener.get_logger()

    n_gpus = torch.cuda.device_count()
    logger.info("Detected %d gpus", n_gpus)
    tasks = list(partial(task, task_id=i) for i in range(10))
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
        scheduler.start()
        process_pool = scheduler.process_pool
        scheduler.join()
    except Exception as e:
        tb = traceback.format_exc()
        logger.critical("While running, exception:\n%s\nTraceback:\n%s", str(e), str(tb))
    finally:
        # make sure listener is stopped
        logger_listener.stop()


if __name__ == "__main__":
    main()
