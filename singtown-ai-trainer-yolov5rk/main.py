from singtown_ai import SingTownAIClient
from singtown_ai import stdout_watcher, file_watcher, error_watcher
from singtown_ai import export_yolo

client = SingTownAIClient()

log_watcher = stdout_watcher(interval=1)

@log_watcher
def on_stdout_write(content: str):
    client.log(content, end="")

@error_watcher()
def on_error(error_message: str):
    client.failed()

from pathlib import Path
import os
import shutil
import torch
import subprocess

YOLOV5_PATH = Path(__file__).parent.parent/"yolov5"
DATASET_PATH = Path("../dataset")
RUNS_PATH = Path("../yolov5")/'runs'
EXP_PATH = RUNS_PATH/'train'/'exp'
METRICS_PATH = EXP_PATH / "results.csv"
shutil.rmtree(RUNS_PATH, ignore_errors=True)

@file_watcher(METRICS_PATH, interval=3)
def file_on_change(content: str):
    import csv
    from io import StringIO

    metrics = list(csv.DictReader(StringIO(content)))
    if not metrics:
        return
    client.update_metrics(metrics)

LABELS = client.task.project.labels
MODEL_NAME = client.task.model_name
EPOCHS = client.task.epochs
BATCH_SIZE = client.task.batch_size
LEARNING_RATE = client.task.learning_rate
EXPORT_WIDTH = client.task.export_width
EXPORT_HEIGHT = client.task.export_height

MODEL_CLASS, IMG_SZ = MODEL_NAME.split("_")

def run_yolov5(command: str):
    """Run a yolov5 script and copy its output into the task log.

    stdout_watcher only replaces this process's sys.stdout/sys.stderr, so the
    output of a child that inherits the real file descriptors never reaches
    the task log: when train.py failed, its error was nowhere to be found.
    """
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=YOLOV5_PATH,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    for raw_line in process.stdout:
        # A progress bar redraws itself with "\r": keep only its final state.
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n").rsplit("\r", 1)[-1]
        # The watcher reads then truncates its buffer; hold its lock so a line
        # written between the two is not lost.
        with log_watcher.lock:
            print(line)
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)

print(f"CUDA available:  {torch.cuda.is_available()}")

print("Download dataset")
export_yolo(client, DATASET_PATH)

print("Training started")
run_yolov5(f"python train.py --data ../dataset/data.yaml --weights ../weights/yolov5s640_coco2017.pt --epochs {EPOCHS} --img {IMG_SZ} --batch-size {BATCH_SIZE}")

print("Export onnx")
run_yolov5(f"python export.py --rknpu rv1103 --weights runs/train/exp/weights/best.pt --img {EXPORT_HEIGHT} {EXPORT_WIDTH} --include onnx")
