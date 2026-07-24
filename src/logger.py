import json
import time
import platform
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import psutil
import torch

class ExperimentLogger:
    def __init__(self, experiment_name, experiment_type ,log_dir, config=None):
        self.experiment_name = experiment_name
        self.experiment_type = experiment_type
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{experiment_type}_{timestamp}"

        self.log_path = self.log_dir / f"{self.run_id}.log"
        self.meta_path = self.log_dir / f"{self.run_id}.json"

        self.logger = self._setup_logger()

        self.config = config or {}
        self.start_wall = None
        self.start_cpu = None
        self.metadata = {
            "run_id": self.run_id,
            "experiment_name": experiment_name,
            "experiment_type": experiment_type,
            "start_time": None,
            "end_time": None,
            "wall_time_sec": None,
            "cpu_time_sec": None,
            "config": self.config,
            "hardware": self.get_hardware_info(),
            "training_history": []
        }

    def _setup_logger(self):
        logger = logging.getLogger(self.run_id)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        fh = logging.FileHandler(self.log_path)
        fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(sh)

        return logger

    def get_hardware_info(self):
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "gpu_memory_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
                ),
            })

        return info

    def start(self):
        self.start_wall = time.perf_counter()
        self.start_cpu = time.process_time()
        self.metadata["start_time"] = datetime.now().isoformat()

        self.logger.info("=" * 80)
        self.logger.info(f"Iniciando experimento: {self.experiment_name}")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Hardware: {self.metadata['hardware']}")
        self.logger.info(f"Config: {self.config}")
        self.logger.info("=" * 80)

    def log_epoch(self, stage, epoch, **metrics):
        row = {
            "stage": stage,
            "epoch": epoch,
            **metrics
        }
        self.metadata["training_history"].append(row)

        msg = f"[{stage}] ep={epoch:04d} " + " | ".join(
            f"{k}={v:.6e}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        self.logger.info(msg)

    def log_message(self, message):
        self.logger.info(message)

    def finish(self, final_metrics=None):
        wall_time = time.perf_counter() - self.start_wall
        cpu_time = time.process_time() - self.start_cpu

        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["wall_time_sec"] = wall_time
        self.metadata["cpu_time_sec"] = cpu_time

        if final_metrics:
            self.metadata["final_metrics"] = final_metrics

        self.logger.info("=" * 80)
        self.logger.info("Experimento finalizado")
        self.logger.info(f"Wall time: {wall_time:.2f} s")
        self.logger.info(f"CPU time: {cpu_time:.2f} s")
        self.logger.info(f"Metadados salvos em: {self.meta_path}")
        self.logger.info("=" * 80)

        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=4)