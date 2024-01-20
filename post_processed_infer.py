import socket
import threading
import re
import cv2
import torch
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer
import numpy as np
import random
import time
import PIL
import os
import json
from glob import glob
from yolov6.data.data_augment import letterbox

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
        LOGGER.warning(e)
        return None, None

class HoverBoardInspection:
    def __init__(self, config, checkpoint_path, label_path = "./data/dataset.yaml", camera_num = 0, success_threshold = 0.2, width = 640, height = 480, inspection_batch = 150, max_duration_seconds = 300, sample_rate = 0.05, **kwargs):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.class_names = load_yaml(label_path)['names']
        self.camera_num = camera_num
        self.success_threshold = success_threshold
        self.width = width
        self.height = height
        self.inspection_batch = inspection_batch
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = sample_rate
        self.screw_boxes = {
            2 : "screw_board",
            3 : "screw_board",
            4 : "electric_board",
            9 : "electric_board", 
            8 : "batterygard"
        }

    def handle_message(self, message):
        pattern = re.compile(r'^SVT_\d+$')
        if not pattern.match(message):
            return -1

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
        client_handler = threading.Thread(target=self.handle_client, args=(server_socket,))
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

                result = self.execute_inspections_with_webcam(num, 
                                                              camera_num=self.camera_num, 
                                                              success_threshold=self.success_threshold, 
                                                              width = self.width, 
                                                              height = self.height, 
                                                              inspection_batch = self.inspection_batch, 
                                                              max_duration_seconds = self.max_duration_seconds,
                                                              sample_rate = self.sample_rate
                                                            )
                client_socket.sendall(result.encode("utf-8"))

        except Exception as e:
            print(f"Error handling client: {e}")

        finally:
            client_socket.close()
    
    def execute_inspections(self, num, img_path="path/to/imgs", success_threshold = 0.2):
        parts_info = self.config.get(num, [])
        total_predictions = self.inference_and_save_images(img_path)

        inspection_results = self._execute_inspections(total_predictions, parts_info, num, success_threshold)

        return json.dumps(inspection_results)
    
    def _execute_inspections(self, total_predictions, parts_info, num, success_threshold = 0.2):
        inspection_results = {"success": False, "details": []}

        success_count = 0
        screw_board = None
        ## prediction = {"frame_number": idx, "class": self.class_names[class_num], "probability": conf, "bounding_box": xyxy}
        for idx, predictions in enumerate(total_predictions):
            for part_info in parts_info:
                part_name = part_info.get("class", None)
                required_quantity = part_info.get("quantity", 0)
                location = part_info.get("location", None)

                message = ""
                screw_positions = []
                for prediction in predictions:
                    class_name = prediction["class"]
                    bounding_box = prediction["bounding_box"]
                    w, h, _ = prediction["img_size"]

                    if required_quantity < 0:
                        break

                    if location == "left":
                        coordinate = [0, 0, (w * 2) // 3, h]
                    elif location == "right":
                        coordinate = [w // 3, 0, w, h]
                    else:
                        coordinate = self._get_location_from_class_name(predictions, location)

                    if class_name == part_name:
                        if coordinate is not None:
                            is_within_threshold, bbox_centroid = self.is_bounding_box_within_coordinate(bounding_box, coordinate)
                            if is_within_threshold:
                                required_quantity -= 1
                                if part_name == "screw":
                                    screw_positions.append(bbox_centroid)
                                else:
                                    message += f"({bbox_centroid}) detected! this belongs to {location}\n"
                                    if class_name == self.screw_boxes[num]:
                                        screw_board = bounding_box
                            else:
                                LOGGER.info(f"({bbox_centroid}) detected! but {class_name} doesn't belong to current target {location}!")
                        else:
                            required_quantity -= 1
                        
                
                if screw_positions:
                    screw_info = self.classify_screw_positions(screw_positions, screw_board)
                    if screw_info is not None:
                        message += self._get_messages_from_screw_coordinate(screw_info, part_info["quantity"], num)
                    screw_board = None
                screw_positions = []

                if required_quantity <= 0:
                    print(f"All required {part_name} inspected.")
                    inspection_results["details"].append({"frame" : prediction["frame_number"], "part_name": part_name, "success": True, "message": message})
                else:
                    inspection_results["details"].append({"frame" : prediction["frame_number"], "part_name": part_name, "success": False, "message": message})

            
            success = all(detail["success"] for detail in inspection_results["details"])
            if success:
                success_count += 1

        if success_count / len(total_predictions) >= success_threshold:
            inspection_results["success"] = True

        return inspection_results
        
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

    def is_bounding_box_within_coordinate(self, bounding_box, coordinate, threshold=5):
        
        x_min, y_min, x_max, y_max = bounding_box
        left, top, right, bottom = coordinate

        bbox_center_x = (x_min + x_max) / 2
        bbox_center_y = (y_min + y_max) / 2
        
        center_within_coordinate = (
            left - threshold <= bbox_center_x <= right + threshold and
            top - threshold <= bbox_center_y <= bottom + threshold
        )
        
        corners_within_coordinate = (
            left - threshold <= x_min <= right + threshold and
            left - threshold <= x_max <= right + threshold and
            top - threshold <= y_min <= bottom + threshold and
            top - threshold <= y_max <= bottom + threshold
        )
        return center_within_coordinate or corners_within_coordinate, (bbox_center_x, bbox_center_y)
    
    def inference_and_save_images(self, image_paths):
        model = DetectBackend(self.checkpoint_path, device=device)
        stride = model.stride

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

                pred_results = model(img)
                det = non_max_suppression(pred_results, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)[0]

                if len(det):
                    det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
                    predictions = []
                    for *xyxy, conf, cls in reversed(det):
                        class_num = int(cls)
                        label = None if hide_labels else (self.class_names[class_num] if hide_conf else f'{self.class_names[class_num]} {conf:.2f}')
                        Inferer.plot_box_and_label(img_src, 2, xyxy,
                                                   label, color=Inferer.generate_colors(class_num, True))
                        os.makedirs("./tmp", exist_ok=True)
                        save_path = os.path.join("./tmp", f"{os.path.basename(image_path)}.png")
                        cv2.imwrite(save_path, img_src)
                        prediction = {
                            "frame_number": idx,
                            "class": self.class_names[class_num],
                            "probability": conf,
                            "bounding_box": xyxy,
                            "img_size": img_src.shape
                        }
                        predictions.append(prediction)
                    total_predictions.append(predictions)
        return total_predictions

    def execute_inspections_with_webcam(self, num, camera_num = 0, success_threshold = 0.2, width = 640, height = 480, sample_rate = 0.05, inspection_batch = 150, max_duration_seconds = 300):
        parts_info = self.config.get(num, [])
        model = DetectBackend(self.checkpoint_path, device=device)
        stride = model.stride
        
        cap = cv2.VideoCapture(camera_num)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        total_predictions = []
        frame_number = 0

        start_time = time.time()
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
                det = non_max_suppression(pred_results, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)[0]

                if len(det):
                    det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
                    predictions = []
                    for *xyxy, conf, cls in reversed(det):
                        class_num = int(cls)
                        label = None if hide_labels else (self.class_names[class_num] if hide_conf else f'{self.class_names[class_num]} {conf:.2f}')
                        Inferer.plot_box_and_label(img_src, 2, xyxy,
                                                   label, color=Inferer.generate_colors(class_num, True))
                        prediction = {
                            "frame_number": frame_number,
                            "class": self.class_names[class_num],
                            "probability": conf,
                            "bounding_box": xyxy,
                            "img_size": img_src.shape
                        }
                        predictions.append(prediction)
                    total_predictions.append(predictions)

                cv2.imshow('Webcam Detection', img_src)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if len(total_predictions) % inspection_batch == 0:
                total_predictions = random.sample(total_predictions, int(inspection_batch * sample_rate))
                inspection_results = self._execute_inspections(total_predictions, parts_info, success_threshold)
                total_predictions = []

            if inspection_results["success"]:
                break
            if (time.time() - start_time) > max_duration_seconds:
                inspection_results["details"].append({"frame" : frame_number, "part_name": "", "success": False, "message": f"time limit {max_duration_seconds} second exceeded. increase max_duration_seconds to wait longer."})
                break
            frame_number += 1

        cap.release()
        cv2.destroyAllWindows()
        return inspection_results

    @staticmethod
    def classify_screw_positions(screw_positions, screw_board):

        if not isinstance(screw_board, list) and len(screw_board) != 4:
            return None

        classifications = set()

        for x, y in screw_positions:
            lr = "left" if x < (screw_board[0] + screw_board[2]) / 2 else "right"
            tb = "top" if y < (screw_board[1] + screw_board[3]) / 2 else "bottom"
            classifications.add(f"{lr} {tb}")

        return classifications


    def _get_messages_from_screw_coordinate(self, classifications, required_quantity, num):
        if len(classifications) == required_quantity:
            return f"all {required_quantity} screws are inside {self.screw_boxes[num]}"
        
        if required_quantity == 4:
            screw_positions = {"left top", "right top", "left bottom", "right bottom"}
        elif required_quantity == 2:
            screw_positions = {"left top", "right top"}

        m = ""
        for p in list(screw_positions - classifications):
            m += f"{p}, "
        m = m.strip(", ") + f" {required_quantity - len(classifications)} screws are not in {self.screw_boxes[num]}"
        return m

if __name__ == "__main__":

    checkpoint_path = 'runs/train/exp9/weights/best_ckpt.pt'
    config = {
        1: [{"quantity": 1, "class": "hoverboard_base"}],
        2: [
            {"quantity": 1, "class": "wheel", "location": "right"},
            {"quantity": 1, "class": "screw_board", "location": "right"},
            {"quantity": 4, "class": "screw", "location": "right-screw_board"}
        ],
        3: [
            {"quantity": 1, "class": "wheel", "location": "left"},
            {"quantity": 1, "class": "screw_board", "location": "left"},
            {"quantity": 4, "class": "screw", "location": "left-screw_board"}
        ],
        4: [
            {"quantity": 1, "class": "electric_board", "location": "left"},
            {"quantity": 2, "class": "screw", "location": "left-electric_board"}
        ],
        5: [{"quantity": 1, "class": "battery"}],
        8: [
            {"quantity": 1, "class": "batterygard", "location": "left"},
            {"quantity": 2, "class": "screw", "location": "left-batterygard"}
        ],
        9: [
            {"quantity": 1, "class": "electric_board", "location": "right"},
            {"quantity": 2, "class": "screw", "location": "right-electric_board"}
        ],
        11: [
            {"quantity": 1, "class": "hoverboard_cover_inside"},
        ],
        12: [
            {"quantity": 1, "class": "hoverboard_cover_inside"},
        ],
    }

    my_instance = HoverBoardInspection(config=config, checkpoint_path=checkpoint_path)
    # my_instance.connect_server("192.168.37.1", 30000)
    response = my_instance.execute_inspections(8, "/Users/gimdoi/Downloads/가이드영상 및 증강 현실 프로그램 제작 자료/frame_rate10/8/v5_002638.png")

    print(response)