import numpy as np
from ultralytics import YOLO

from settings import get_yolo_model_path

def load_yolo_model():
    global yolo_model
    yolo_model = YOLO(get_yolo_model_path())

def parse_face(raw_image: np.ndarray):
    pred_result = yolo_model.predict(raw_image, conf=0.4)

    y_size, x_size = raw_image.shape[0:2]
    truth_matrix = np.ones(raw_image.shape[0:2])
    if pred_result:
        if pred_result[0].boxes:
            x1, y1, x2, y2 = map(int, list(pred_result[0].boxes[0].xyxy[0]))
            truth_matrix = np.zeros(raw_image.shape[0:2])
            truth_matrix[y1:y2, x1:x2] = 1
            y_size = y2 - y1
            x_size = x2 - x1
            print("FACE DETECTED!")

    return truth_matrix.astype(np.bool), x_size, y_size