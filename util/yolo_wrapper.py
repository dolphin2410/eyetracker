#     Copyright (C) 2025 dolphin2410
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

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