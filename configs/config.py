from confz import BaseConfig
from typing import Tuple


class DetectorArgs(BaseConfig):
    weights: str
    iou: float
    conf: float
    imgsz: Tuple[int, int]
    batch_size: int


class ClassificatorArgs(BaseConfig):
    weights: str
    imgsz: Tuple[int, int]
    batch_size: int


class MainConfig(BaseConfig):
    src_dir: str
    mapping: str
    device: str
    detector: DetectorArgs
    classificator: ClassificatorArgs
