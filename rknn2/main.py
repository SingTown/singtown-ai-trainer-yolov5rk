from singtown_ai import SingTownAIClient
from singtown_ai import stdout_watcher, error_watcher

client = SingTownAIClient()

@stdout_watcher(interval=1)
def on_stdout_write(content: str):
    client.log(content, end="")

@error_watcher()
def on_error(error_message: str):
    client.failed()

from pathlib import Path
from rknn.api import RKNN
import os
import tarfile
import torch
import random

RUN_PATH = Path("run")
IMAGES_PATH = Path("../dataset")/"images"/"TRAIN"
RUNS_PATH = Path("../yolov5")/'runs'
EXP_PATH = RUNS_PATH/'train'/'exp'
RUN_PATH.mkdir(parents=True, exist_ok=True)

LABELS = client.task.project.labels
EXPORT_WIDTH = client.task.export_width
EXPORT_HEIGHT = client.task.export_height

print(f"CUDA available:  {torch.cuda.is_available()}")

rknn = RKNN()

rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rv1106")
ret = rknn.load_onnx(str(EXP_PATH / "weights/best.onnx"))
if ret != 0:
    raise Exception("Load model failed!")

with open(RUN_PATH/"dataset.txt", "w") as f:
    imgs = os.listdir(IMAGES_PATH)
    selected_imgs = random.sample(imgs, min(100, len(imgs)))
    for img in selected_imgs:
        f.write(str((IMAGES_PATH/img).absolute()) + "\n")

ret = rknn.build(do_quantization=True, dataset=RUN_PATH/"dataset.txt")
if ret != 0:
    raise Exception("Build model failed!")

ret = rknn.export_rknn(str(RUN_PATH/"best.rknn"))
if ret != 0:
    raise Exception("Export model failed!")

rknn.release()

with open(RUN_PATH/"labels.txt", "wb") as f:
    f.write("\n".join(LABELS).encode("utf-8"))

with tarfile.open(RUN_PATH/"yolov5.tar", "w") as zipf:
    zipf.add(RUN_PATH/"labels.txt", arcname="labels.txt")
    zipf.add(RUN_PATH/"best.rknn", arcname="yolov5.rknn")
    zipf.add("../yolov5/RK_anchors.txt", arcname="anchors.txt")

client.upload_results_zip(RUN_PATH/"yolov5.tar")
client.success()
print("Finished")
