import socket
import threading
import re
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
import json
from glob import glob
from yolov6.data.data_augment import letterbox
from tqdm import tqdm
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

hide_labels: bool = False
hide_conf: bool = False
img_size: int = 640
conf_thres: float = 0.25
iou_thres: float = 0.45
max_det: int = 1000
agnostic_nms: bool = False
half: bool = False

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

class HoverBoardInspection:
    def __init__(self, config, checkpoint_path):
        # 여기에서 초기화 작업 수행
        self.config = config
        self.checkpoint_path = checkpoint_path

    def handle_message(self, message):
        # SVT로 시작하고 바로 뒤에 _가 오며, 그 뒤에 숫자가 오는지 확인하는 정규식
        pattern = re.compile(r'^SVT_\d+$')
        if not pattern.match(message):
            # 올바르지 않은 형식일 경우 404 반환
            return -1

        # 정규식에 맞는 경우 숫자 부분을 추출하여 반환
        number = int(message.split('_')[1])
        return number

    def start_server(self, host, port):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"Server listening on {host}:{port}")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"Accepted connection from {addr}")

            client_handler = threading.Thread(target=self.handle_client, args=(client_socket,))
            client_handler.start()

    def connect_server(self, host, port):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.connect((host, port))
        server_socket.send('Hello!'.encode())
        print(f"connected on {host}:{port}")
        client_handler = threading.Thread(target=self.handle_client, args=(client_socket,))
        client_handler.start()

    def handle_client(self, client_socket):
        try:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break

                message = data.decode("utf-8")
                num = self.handle_message(message)

                if num < 0:
                    return f"{message}:invalid input message received!"

                result = self.execute_inspections(num)
                client_socket.sendall(result.encode("utf-8"))

        except Exception as e:
            print(f"Error handling client: {e}")

        finally:
            client_socket.close()
    
    def execute_inspections(self, num, img_path="path/to/imgs"):
        parts_info = self.config.get(num, [])
        inspection_results = {"success": False, "details": []}
        mask = self.config.get("mask", {3: "right", 9: "left"})
        total_predictions = self.inference_and_save_images(img_path, mask = mask.get(num, None))

        ## prediction = {"frame_number": idx, "class": class_names[class_num], "probability": conf, "bounding_box": xyxy}
        for predictions in total_predictions: # predictions 는 현재 프레임의 모든 boundingbox 결과를 포함
            print(f"predictions: {predictions}")
            for part_info in parts_info:
                part_name = part_info.get("class", None)
                required_quantity = part_info.get("quantity", 0)
                location = part_info.get("location", None)

                message = ""
                for prediction in predictions:
                    class_name = prediction["class"]
                    bounding_box = prediction["bounding_box"]
                    w,h,_ = prediction["img_size"]

                    if required_quantity < 0:
                        break

                    if location == "left":
                        coordinate = [0, 0, w // 2, h]
                    elif location == "right":
                        coordinate = [w // 2, w // 2, w, h]
                    else:
                        coordinate = self._get_location_from_class_name(predictions, location)

                    if class_name == part_name:
                        if coordinate is not None:
                            is_within_threshold, bbox_centroid = self.is_bounding_box_within_coordinate(bounding_box, coordinate) # TODO 그냥 quantity를 줄이지 않고 이 값을 통과할 때만 줄이도록
                            message += f"({bbox_centroid}) detected at location {location} {is_within_threshold}\n"
                            print(f"{part_name} ({is_within_threshold}/{bbox_centroid}) detected at location {location} {is_within_threshold}")
                            if is_within_threshold:
                                required_quantity -= 1
                        else:
                            required_quantity -= 1  # 처리한 개수를 감소시킴

                if required_quantity == 0:
                    print(f"All required {part_name} inspected.")
                    inspection_results["details"].append({"frame" : prediction["frame_number"], "part_name": part_name, "success": True, "message": message})
                else:
                    print(f"Inspection of {part_name} incomplete. {required_quantity} remaining.")
                    inspection_results["details"].append({"frame" : prediction["frame_number"], "part_name": part_name, "success": False, "message": message})

            # 모든 부품이 성공적으로 검사되었는지 확인
            if all(detail["success"] for detail in inspection_results["details"]):
                inspection_results["success"] = True


        # 클라이언트에게 결과를 전송
        return json.dumps(inspection_results)
        
    def _get_location_from_class_name(self, predictions, location):

        if location is None:
            return

        if "-" in location:
            left_or_right, location = location.split("-")
        cnt = 0
        bounding_box = None
        for prediction in predictions:
            if prediction["class"] == location:
                if left_or_right == "left" and prediction["bounding_box"][2] <= prediction["img_size"][1] // 2:
                    bounding_box = prediction["bounding_box"]
                    cnt += 1
                elif left_or_right == "right" and prediction["bounding_box"][0] >= prediction["img_size"][1] // 2:
                    bounding_box = prediction["bounding_box"]
                    cnt += 1
                elif left_or_right is None:
                    bounding_box = prediction["bounding_box"]
                    cnt += 1
        return bounding_box if cnt == 1 else None

    def is_bounding_box_within_coordinate(self, bounding_box, coordinate, threshold=0.5):
        
        x_min, y_min, x_max, y_max = bounding_box
        left, top, right, bottom = coordinate

        bbox_center_x = (x_min + x_max) / 2
        bbox_center_y = (y_min + y_max) / 2
        
        center_within_coordinate = (
            left - threshold <= bbox_center_x <= right + threshold and
            top - threshold <= bbox_center_y <= bottom + threshold
        )
        
        # bounding_box의 네 꼭지점이 좌표 내에 있고, threshold 이상 겹치는 경우 True 반환
        corners_within_coordinate = (
            left - threshold <= x_min <= right + threshold and
            left - threshold <= x_max <= right + threshold and
            top - threshold <= y_min <= bottom + threshold and
            top - threshold <= y_max <= bottom + threshold
        )
    
        
        # 두 조건 중 적어도 하나를 만족하면 True 반환
        return center_within_coordinate or corners_within_coordinate, (bbox_center_x, bbox_center_y)
    
    def inference_and_save_images(self, image_paths, mask=None):
        model = DetectBackend(self.checkpoint_path, device=device)
        stride = model.stride
        class_names = load_yaml("./data/dataset.yaml")['names']

        if os.path.isfile(image_paths):
            image_paths = [image_paths]
        elif os.path.isdir(image_paths):
            image_paths = glob(os.path.join(image_paths, "*"))

        total_predictions = []
        for idx, image_path in enumerate(image_paths):
            frame = cv2.imread(image_path)
            img, img_src = process_image(frame, img_size, stride, half)
            if img is not None:
                img = img.to(device)
                if len(img.shape) == 3:
                    img = img[None]
                
                if mask == "left":
                    img[:, :, :, :img.size(3) // 2] = 0
                elif mask == "right":
                    img[:, :, :, img.size(3) // 2:] = 0

                pred_results = model(img)
                classes = None
                det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

                if len(det):
                    det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
                    predictions = []
                    for *xyxy, conf, cls in reversed(det):
                        class_num = int(cls)
                        label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
                        Inferer.plot_box_and_label(img_src, 2, xyxy,
                                                   label, color=Inferer.generate_colors(class_num, True))
                        os.makedirs("./tmp", exist_ok=True)
                        save_path = os.path.join("./tmp", f"{os.path.basename(image_path)}.png")
                        cv2.imwrite(save_path, img_src)
                        prediction = {
                            "frame_number": idx,
                            "class": class_names[class_num],
                            "probability": conf,
                            "bounding_box": xyxy,
                            "img_size": img_src.shape
                        }
                        predictions.append(prediction)
                    total_predictions.append(predictions)
        return total_predictions


if __name__ == "__main__":

    checkpoint_path = 'runs/train/exp4/weights/best_ckpt.pt'
    config = {
        "mask" : {
            # 3 : "right",
            8 : "right",
            9 : "left",
        },
        1: [{"part_name": "hoverboard_base", "quantity": 1, "class": "hoverboard_base"}],
        2: [
            {"part_name": "wheel", "quantity": 1, "class": "wheel", "location": "right"},
            {"part_name": "screw_board", "quantity": 1, "class": "screw_board", "location": "right"},
            {"part_name": "screw", "quantity": 4, "class": "screw", "location": "right-screw_board"}
        ],
        3: [
            {"part_name": "wheel", "quantity": 1, "class": "wheel", "location": "left"},
            {"part_name": "screw_board", "quantity": 1, "class": "screw_board", "location": "left"},
            {"part_name": "screw", "quantity": 4, "class": "screw", "location": "left-screw_board"}
        ],
        4: [
            {"part_name": "electric_board", "quantity": 1, "class": "electric_board", "location": "left"},
            {"part_name": "screw", "quantity": 2, "class": "screw", "location": "left-electric_board"}
        ],
        5: [{"part_name": "battery", "quantity": 1, "class": "battery"}],
        8: [
            {"part_name": "batterygard", "quantity": 1, "class": "batterygard", "location": "left"},
            {"part_name": "screw", "quantity": 2, "class": "screw", "location": "left-batterygard"}
        ],
        9: [
            {"part_name": "electric_board", "quantity": 1, "class": "electric_board", "location": "right"},
            {"part_name": "screw", "quantity": 2, "class": "screw", "location": "right-electric_board"}
        ],
        11: [
            {"part_name": "hoverboard_cover_inside", "quantity": 1, "class": "hoverboard_cover_inside"},
        ],
        12: [
            {"part_name": "hoverboard_cover_outside", "quantity": 1, "class": "hoverboard_cover_inside"},
        ],
    }

    my_instance = HoverBoardInspection(config=config, checkpoint_path=checkpoint_path)
    my_instance.connect_server("192.168.37.1", 30000)
    my_instance.execute_inspections(11, "/Users/gimdoi/Downloads/가이드영상 및 증강 현실 프로그램 제작 자료/frames/11 커버 좌 우 지그 이동/11_382.png")