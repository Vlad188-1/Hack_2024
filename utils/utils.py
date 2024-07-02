from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
from torchvision.transforms.functional import normalize
import numpy as np
from confz import BaseConfig
import cv2
from pathlib import Path


MEAN = [123.675, 116.28, 103.535]
STD = [58.395, 57.12, 57.375]


def load_detector(config: BaseConfig):
    # Loading YOLOv8 weights
    model = YOLO(config.weights)
    return model


def load_classificator(config: BaseConfig):
    # Loading timm model
    model = torch.load(config.weights)
    model.eval()
    return model


def open_mapping(path_mapping: str) -> dict[int, str]:
    with open(path_mapping, 'r') as txt_file:
        lines = txt_file.readlines()
        lines = [i.strip() for i in lines]
        dict_map = {k: v for k, v in enumerate(lines)}
    return dict_map


def extract_crops(results: list[Results], config: BaseConfig) -> dict[str, torch.Tensor]:
    dict_crops = {}
    for res_per_img in results:
        if len(res_per_img) > 0:
            crops_per_img = []
            for box in res_per_img.boxes:
                x0, y0, x1, y1 = box.xyxy.cpu().numpy().ravel().astype(np.int32)
                crop = res_per_img.orig_img[y0: y1, x0: x1]

                # Do squared crop
                # crop = letterbox(img=crop, new_shape=config.imgsz, color=(0, 0, 0))
                crop = cv2.resize(crop, config.imgsz, interpolation=cv2.INTER_LINEAR)

                # Convert Array crop to Torch tensor with [batch, channels, height, width] dimensions
                crop = torch.from_numpy(crop.transpose(2, 0, 1))
                crop = crop.unsqueeze(0)
                crop = normalize(crop.float(), mean=MEAN, std=STD)
                crops_per_img.append(crop)

            dict_crops[Path(res_per_img.path).name] = torch.cat(crops_per_img) # if len(crops_per_img) else None
    return dict_crops


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # if not scaleup:  # only scale down, do not scale up (for better test mAP)
    #     r = min(r, 1.0)

    # Compute padding
    # ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # if auto:  # minimum rectangle
    #     dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # elif scaleFill:  # stretch
    #     dw, dh = 0.0, 0.0
    #     new_unpad = (new_shape[1], new_shape[0])
    #     ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img
