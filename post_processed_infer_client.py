import socket
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
import os
import json
from glob import glob
from yolov6.data.data_augment import letterbox
from collections import defaultdict

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
    def __init__(self, config, checkpoint_path, label_path = "./data/dataset.yaml", camera_num = 0, success_threshold = 0.3, width = 640, height = 480, inspection_batch = 150, max_duration_seconds = 300, sample_rate = 0.03, **kwargs):
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
        pattern = re.compile(r'SVT_\d+,?')
        match = pattern.search(message)
        filtered_message = match.group() if match else None

        if filtered_message is None:
            return -1
        
        filtered_message = filtered_message.strip(",")

        number = int(filtered_message.split('_')[1])
        return number

    def test_server(self, host, port, process_message="SVT_3,"):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"Server listening on {host}:{port}")

        client_socket, addr = server_socket.accept()
        print(f"Accepted connection from {addr}")

        client_socket.send(process_message.encode("utf-8"))
        msg = client_socket.recv(1024)
        server_socket.close()
        return msg

    def connect_server(self, host, port):
        self.handle_client(host, port)

    def handle_client(self, host, port):
        while True:
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((host, port))
                print(f"connected on {host}:{port}")
                data = client_socket.recv(1024)
                if not data:
                    continue
                message = data.decode("utf-8")
                num = self.handle_message(message)

                if num < 0:
                    break

                result = self.execute_inspections_with_webcam(num, 
                                                              camera_num=self.camera_num, 
                                                              success_threshold=self.success_threshold, 
                                                              width = self.width, 
                                                              height = self.height, 
                                                              inspection_batch = self.inspection_batch, 
                                                              max_duration_seconds = self.max_duration_seconds,
                                                              sample_rate = self.sample_rate
                                                            )

                client_socket.send(result.encode("utf-8"))
                data = False

            except Exception as e:
                if "Connection refused" not in str(e):
                    print(f"Error handling client: {e}")
        client_socket.close()
        
    
    def execute_inspections(self, num, img_path="path/to/imgs", success_threshold = 0.3):
        parts_info = self.config.get(num, [])
        total_predictions = self.inference_from_file_system(img_path)

        inspection_results = self._execute_inspections(total_predictions, parts_info, num, success_threshold)

        return json.dumps(inspection_results)
    
    def _execute_inspections(self, total_predictions, parts_info, num, success_threshold = 0.3):
        inspection_results = {"success": False, "details": str()}

        message_detail = defaultdict(str)
        success_count = 0
        screw_board = None
        screw_flag = not any(p["class"] == "screw" for p in parts_info)
        ## prediction = {"frame_number": idx, "class": self.class_names[class_num], "probability": conf, "bounding_box": xyxy}
        for idx, predictions in enumerate(total_predictions): # frame
            for part_info in parts_info: # classes to check
                part_name = part_info.get("class", None)
                required_quantity = part_info.get("quantity", 0)
                location = part_info.get("location", None)
                strict_label = part_info.get("strict", False)

                message = ""
                screw_positions = []
                for prediction in predictions:
                    class_name = prediction["class"]
                    bounding_box = prediction["bounding_box"]
                    frame_number = prediction["frame_number"]
                    h, w, _ = prediction["img_size"]

                    # if required_quantity < 0:
                    #     break

                    if location == "left":
                        coordinate = [0, 0, w // 2, h]
                    elif location == "right":
                        coordinate = [w // 2, 0, w, h]
                    else:
                        coordinate = self._get_location_from_class_name(predictions, location)
                    if class_name == part_name:
                        if coordinate is not None:
                            is_within_threshold, bbox_centroid = self.is_bounding_box_within_coordinate(bounding_box, coordinate)
                            if is_within_threshold:
                                required_quantity -= 1
                                LOGGER.info(f"{part_name} ({bbox_centroid}) detected! this belongs to {location}")
                                if part_name == "screw":
                                    screw_positions.append(bbox_centroid)
                                elif class_name == self.screw_boxes[num]:
                                    screw_board = bounding_box
                            else:
                                LOGGER.info(f"{part_name} ({bbox_centroid}) detected! but {class_name} doesn't belong to current target {location}!")
                        else:
                            required_quantity -= 1
                        
                
                if screw_positions:
                    screw_info = self.classify_screw_positions(screw_positions, screw_board)
                    if screw_info is not None:
                        message += self._get_messages_from_screw_coordinate(screw_info, part_info["quantity"], num)
                        if not message:
                            screw_flag = True
                    # screw_board = None

                if required_quantity <= 0:
                    if required_quantity == 0:
                        print(f"All required {part_name} inspected.")
                    elif strict_label:
                        message_detail[frame_number] +=  f"{part_name},"
                    # message_detail[prediction["frame_number"]] +=  f"success:{part_name},"
                else:
                    if part_name == "screw":
                        message_detail[frame_number] += message
                    else:
                        message_detail[frame_number] +=  f"{part_name},"
            

            if not message_detail[frame_number] and screw_flag:
                success_index = frame_number
                success_count += 1
            else:
                last_fail = frame_number

        if success_count / len(total_predictions) >= success_threshold:
            inspection_results["success"] = True
            inspection_results["details"] = message_detail[success_index]
        else:
            inspection_results["details"] = f"fail:" + message_detail[last_fail]

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
    
    def _inference_frame(self, frame, model, stride, frame_number, save_path=None):
        predictions = []
        img, img_src = process_image(frame, img_size, stride, half)

        if img is None:
            return

        img = img.to(device)
        if len(img.shape) == 3:
            img = img[None]

        pred_results = model(img)
        det = non_max_suppression(pred_results, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)[0]

        if len(det):
            det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)
                label = None if hide_labels else (self.class_names[class_num] if hide_conf else f'{self.class_names[class_num]} {conf:.2f}')
                Inferer.plot_box_and_label(img_src, 2, xyxy,
                                            label, color=Inferer.generate_colors(class_num, True))
                if save_path is not None:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    save_path = os.path.join(save_path)
                    cv2.imwrite(save_path, img_src)
        
                prediction = {
                    "frame_number": frame_number,
                    "class": self.class_names[class_num],
                    "probability": conf,
                    "bounding_box": xyxy,
                    "img_size": img_src.shape
                }
                predictions.append(prediction)
        return predictions, img_src

    def inference_from_file_system(self, image_paths):
        model = DetectBackend(self.checkpoint_path, device=device)
        stride = model.stride

        if os.path.isfile(image_paths):
            image_paths = [image_paths]
        elif os.path.isdir(image_paths):
            image_paths = glob(os.path.join(image_paths, "*"))

        total_predictions = []
        for idx, image_path in enumerate(image_paths):
            frame = cv2.imread(image_path)
            predictions, _ = self._inference_frame(frame, model, stride, idx, save_path = f"./tmp/{os.path.basename(image_path)}")
            if predictions is not None:
                total_predictions.append(predictions)
        return total_predictions

    def execute_inspections_with_webcam(self, num, camera_num = 0, success_threshold = 0.3, width = 640, height = 480, sample_rate = 0.03, inspection_batch = 150, max_duration_seconds = 300):
        parts_info = self.config.get(num, [])
        model = DetectBackend(self.checkpoint_path, device=device)
        stride = model.stride
        
        cap = cv2.VideoCapture(camera_num)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        total_predictions = []
        frame_number = 0
        inspection_results = {"success": False, "details": ""}
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_number > 0:
                    frame_number -= 1
                break

            predictions, img_src = self._inference_frame(frame, model, stride, frame_number)
            total_predictions.append(predictions)

            cv2.imshow('Webcam Detection', img_src)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"break while loop")
                break

            if len(total_predictions) % inspection_batch == 0 and len(total_predictions) > 0:
                total_predictions = random.sample(total_predictions[:-2], int(inspection_batch * sample_rate)) + total_predictions[:-2]
                inspection_results = self._execute_inspections(total_predictions, parts_info, num, success_threshold)

                total_predictions = []

            if inspection_results["success"]:
                inspection_results["details"] = "success,"
                break
            if (time.time() - start_time) > max_duration_seconds:
                inspection_results["details"][frame_number] += f"fail:time limit {max_duration_seconds} second exceeded. increase max_duration_seconds to wait longer."
                break
            frame_number += 1

        cap.release()
        cv2.destroyAllWindows()
        print(f"inspection_results : {inspection_results}")
        return inspection_results["details"]

    @staticmethod
    def classify_screw_positions(screw_positions, screw_board):

        if not isinstance(screw_board, list) and len(screw_board) != 4:
            return None

        classifications = set()

        for x, y in screw_positions:
            lr = "l" if x < (screw_board[0] + screw_board[2]) / 2 else "r"
            tb = "t" if y < (screw_board[1] + screw_board[3]) / 2 else "b"
            classifications.add(f"{lr}{tb}m")

        return classifications

    def _get_messages_from_screw_coordinate(self, classifications, required_quantity, num):
        if len(classifications) == required_quantity:
            LOGGER.info(f"all {required_quantity} screws are inside {self.screw_boxes[num]}")
            return ""

        if required_quantity == 4:
            screw_positions = {"ltm", "rtm", "lbm", "rbm"}
        elif required_quantity == 2:
            screw_positions = {"ltm", "rtm"}

        m = ""
        for p in list(screw_positions - classifications):
            m += f"{p},"
        # m = m.strip(",") + f" {required_quantity - len(classifications)} screws are not in {self.screw_boxes[num]}"
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
        14: [
            {"quantity": 0, "class": "hoverboard_cover_inside", "strict" : True},
            {"quantity": 1, "class": "hoverboard_cover_outside"},
        ],
    }

    my_instance = HoverBoardInspection(config=config, checkpoint_path=checkpoint_path, camera_num = "/Users/gimdoi/Downloads/가이드영상 및 증강 현실 프로그램 제작 자료/2 챕터 영상 정리/5 배터리 배치.mp4")
    msg = my_instance.connect_server("127.0.0.1", 10000)
    print(msg)
    # response = my_instance.execute_inspections_with_webcam(3, )

    # pprint(response)