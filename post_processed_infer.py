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
from glob import glob
from yolov6.data.data_augment import letterbox
from tqdm import tqdm
import pickle

# Set-up hardware options
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

# Run YOLOv6 on a video from a URL.
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

                self.execute_inspections(num)

        except Exception as e:
            print(f"Error handling client: {e}")

        finally:
            client_socket.close()
    
    def execute_inspections(self, num):
        # num에 따라 필요한 부품들의 정보 가져오기
        parts_info = self.config.get(num, [])

        if parts_info:
            for part_info in parts_info:
                part_name = part_info["part_name"]
                quantity = part_info["quantity"]

                # 여기에서 필요한 부품에 대한 처리 수행
                print(f"Inspecting {quantity} {part_name}(s).")

                # YOLOv6 실행 및 결과 분석
                predictions = self.inference_and_save_video(file_path="/default/file/path/frame.jpg")
                
                # 필요한 클래스와 개수 확인
                required_classes = [part_info["class"]]  # 여기에 필요한 클래스 이름을 넣어주세요
                required_quantity = quantity  # 여기에 필요한 개수를 넣어주세요
                
                # 결과 분석
                for prediction in predictions:
                    class_name = prediction["class"]
                    bounding_box = prediction["bounding_box"]

                    if class_name in required_classes:
                        # 필요한 클래스에 대한 처리 수행
                        # 여기에 추가적인 로직을 작성하세요
                        required_quantity -= 1  # 처리한 개수를 감소시킴

                        # bounding_box를 이용한 추가적인 처리 등 수행
                        # ...

                if required_quantity == 0:
                    print(f"All required {part_name} inspected.")
                else:
                    print(f"Inspection of {part_name} incomplete. {required_quantity} remaining.")

        else:
            print(f"No information found for process with number {num}.")

    def inference_and_save_video(self, file_path="/default/file/path/frame.jpg"):
        model = DetectBackend(self.checkpoint_path, device=device)
        stride = model.stride
        class_names = load_yaml("./data/dataset.yaml")['names']

        cap = cv2.VideoCapture(file_path)
        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))
        # out = cv2.VideoWriter(os.path.join(output_path, f"{part_name}_output.mp4"), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

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
                        prediction = {
                            "frame_number": idx,
                            "class": class_names[class_num],
                            "probability": conf,
                            "bounding_box": xyxy
                        }
                        predictions.append(prediction)
                # out.write(img_src)
                idx += 1

        cap.release()
        # out.release()
        return predictions

if __name__ == "__main__":
    # 여기에서 YOLOv6 체크포인트 경로와 config 정보를 설정
    checkpoint_path = 'runs/train/exp4/weights/best_ckpt.pt'
    config = {
        1: [{"part_name": "hoverboard_base", "quantity": 1, "class": "hoverboard_base", "location": "right"}],
        2: [
            {"part_name": "wheel", "quantity": 1, "class": "wheel", "location": "right"},
            {"part_name": "screw_board", "quantity": 1, "class": "screw_board", "location": "right"},
            {"part_name": "screw", "quantity": 4, "class": "screw", "location": "screw_board"}
        ],
        3: [
            {"part_name": "wheel", "quantity": 1, "class": "wheel", "location": "left"},
            {"part_name": "screw_board", "quantity": 1, "class": "screw_board", "location": "left"},
            {"part_name": "screw", "quantity": 4, "class": "screw", "location": "screw_board"}
        ],
        4: [
            {"part_name": "electric_board", "quantity": 1, "class": "electric_board", "location": "left"},
            {"part_name": "screw", "quantity": 2, "class": "screw", "location": "electric_board"}
        ],
        5: [{"part_name": "battery", "quantity": 1, "class": "battery", "location": "left"}],
        8: [
            {"part_name": "batterygard", "quantity": 1, "class": "batterygard", "location": "left"},
            {"part_name": "screw", "quantity": 2, "class": "screw", "location": "batterygard"}
        ],
        9: [
            {"part_name": "electric_board", "quantity": 1, "class": "electric_board", "location": "right"},
            {"part_name": "screw", "quantity": 2, "class": "screw", "location": "electric_board"}
        ],
        11: [
            {"part_name": "hoverboard_inside", "quantity": 1, "class": "hoverboard_inside"},
            {"part_name": "hoverboard_inside", "quantity": 2, "class": "hoverboard_inside"}
        ],
        12: [
            {"part_name": "hoverboard_outside", "quantity": 1, "class": "hoverboard_outside"},
            {"part_name": "hoverboard_base", "quantity": 0, "class": "hoverboard_base"}
        ],
        # 다른 공정에 대한 정보도 추가 가능
    }


    my_instance = HoverBoardInspection(config=config, checkpoint_path=checkpoint_path)
    my_instance.start_server("127.0.0.1", 8080)
