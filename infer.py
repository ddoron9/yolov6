import cv2
import torch
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer
from typing import List, Optional
import math
import requests

import numpy as np
import PIL
import os
from glob import glob
from yolov6.data.data_augment import letterbox
from tqdm import tqdm
from typing import List, Optional
import pickle


# Set-up hardware options
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')


# Run YOLOv6 on a video from a URL.
hide_labels: bool = False  # @param {type:"boolean"}
hide_conf: bool = False  # @param {type:"boolean"}

img_size: int = 640  # @param {type:"integer"}

conf_thres: float = 0.25  # @param {type:"number"}
iou_thres: float = 0.45  # @param {type:"number"}
max_det: int = 1000  # @param {type:"integer"}
agnostic_nms: bool = False  # @param {type:"boolean"}
half:bool = False #@param {type:"boolean"}

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32, floor=0):
    def make_divisible(x, divisor):
        return math.ceil(x / divisor) * divisor

    if isinstance(img_size, int):
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size, list) else [new_size] * 2


def process_image(frame, img_size, stride, half):
    try:
        image = letterbox(frame, img_size, stride=stride)[0]
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()
        image /= 255
        return image, frame
    except Exception as e:
        LOGGER.Warning(e)
        return None, None


def inference_and_save_video(video_path, output_path, checkpoint_path):
    model = DetectBackend(checkpoint_path, device=device)
    stride = model.stride
    class_names = load_yaml("./data/dataset.yaml")['names']

    bbox_path = os.path.splitext(output_path)[0]
    os.makedirs(bbox_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    predictions = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img, img_src = process_image(frame, img_size, stride, half)
        if img is not None:
            img = img.to(device)
            if len(img.shape) == 3:
                img = img[None]

            pred_results = model(img)
            classes = None
            det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

            if len(det):
                det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    class_num = int(cls)
                    label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
                    Inferer.plot_box_and_label(img_src, 2, xyxy,
                                               label, color=Inferer.generate_colors(class_num, True))
                    # Inferer.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy,
                    #                            label, color=Inferer.generate_colors(class_num, True))
                    prediction = {
                        "frame_number": idx,
                        "class": class_names[class_num],
                        "probability": conf,
                        "bounding_box": xyxy
                    }
                    predictions.append(prediction)
            out.write(img_src)
            with open(os.path.join(bbox_path, f"{idx:03d}.pickle"), "wb") as f:
                pickle.dump(predictions, f)
            idx += 1

    cap.release()
    out.release()


checkpoint_path = 'runs/train/exp4/weights/best_ckpt.pt'  # 2번 코드의 체크포인트 경로

output_dir = "./output_each_exp4"
os.makedirs(output_dir, exist_ok=True)

for video_path in tqdm(glob("/Users/gimdoi/Downloads/sample_video/*.mp4")):

    output_path = os.path.join(output_dir, os.path.basename(video_path))

    inference_and_save_video(video_path, output_path, checkpoint_path)

# video_path = "/data1/doyi/sample_video/03_left_poka_longer.mp4"

# output_path = os.path.join(output_dir, os.path.basename(video_path))
                            
# inference_and_save_video(video_path, output_path, checkpoint_path)

