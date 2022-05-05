import logging
from logging.handlers import QueueHandler
from queue import Empty
from typing import Callable, List, Dict, Any

import torch.multiprocessing as mp


class GPUTaskScheduler:
    def __init__(
        self,
        tasks: List[Callable[[int], Dict[str, Any]]],
        n_gpus: int,
        logger_queue: mp.Queue,
        start_method="spawn"
    ):
        self.tasks = tasks
        self.n_gpus = n_gpus
        self.logger_queue = logger_queue
        self.start_method = start_method

        self.process_pool: List[mp.Process] = list()
        # task queue
        mmp = mp.get_context(start_method)
        self._queue = mmp.Queue()
        for task in self.tasks:
            self._queue.put(task)

    def start(self):
        # launch process
        self.process_context = mp.spawn(
            self.task_worker,
            nprocs=self.n_gpus,
            join=False,
            start_method=self.start_method
        )
        self.process_pool = self.process_context.processes

    def task_worker(self, gpu_id: int):
        root_logger = logging.getLogger()
        handler = QueueHandler(self.logger_queue)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False

        logger = logging.getLogger(f"task_worker_{gpu_id}")
        logger.info(f"Task worker {gpu_id} launched.")
        try:
            while True:
                task_fn = self._queue.get_nowait()
                task_fn(gpu_id)
        except Empty:
            logger.info(f"Task worker {gpu_id} terminated.")

    def join(self):
        # Loop on join until it returns True or raises an exception.
        while not self.process_context.join():
            pass


