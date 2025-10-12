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
from enum import Enum

import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image

from settings import get_model_path
from face_parsing.models.bisenet import BiSeNet

class KeypointType(Enum):
    BACKGROUND = 0
    FACE_BACKGROUND = 1
    RIGHT_EYEBROW = 2
    LEFT_EYEBROW = 3
    RIGHT_EYE = 4
    LEFT_EYE = 5
    UNKNOWN_6 = 6
    RIGHT_EAR = 7
    LEFT_EAR = 8
    UNKNOWN_9 = 9
    NOSE = 10
    UNKNOWN_11 = 11
    TOP_LIP = 12
    BOTTOM_LIP = 13
    NECK = 14
    UNKNOWN_15 = 15
    BODY = 16
    HAIR = 17
    UNKNOWN_18 = 18


class Keypoint:
    def __init__(self, keypoint_type: KeypointType, matrix: np.ndarray):
        self.keypoint_type = keypoint_type
        self.matrix = matrix

KEYPOINT_INDEX_TO_NAME = [
    'background',
    'face_background',
    'right_eyebrow',
    'left_eyebrow',
    'right_eye',
    'left_eye',
    'unknown_6',
    'right_ear',
    'left_ear',
    'unknown_9',
    'nose',
    'unknown_11',
    'top_lip',
    'bottom_lip',
    'neck',
    'unknown_15',
    'body',
    'hair',
    'unknown_18'
]

KEYPOINT_NAME_TO_INDEX = {
    'background': 0,
    'face_background': 1,
    'right_eyebrow': 2,
    'left_eyebrow': 3,
    'right_eye': 4,
    'left_eye': 5,
    'unknown_6': 6,
    'right_ear': 7,
    'left_ear': 8,
    'unknown_9': 9,
    'nose': 10,
    'unknown_11': 11,
    'top_lip': 12,
    'bottom_lip': 13,
    'neck': 14,
    'unknown_15': 15,
    'body': 16,
    'hair': 17,
    'unknown_18': 18
}


def filter_not(list_keypoints):
    """Returns an index list of keypoints that weren't passed to list_keypoints

    Args:
        list_keypoints: List of keypoint names

    Returns:
        Index list of keypoints that isn't contained in list_keypoints
    """

    index_list = np.array(range(19))
    index_list = np.delete(
        index_list,
        list(map(lambda keypoint_name: KEYPOINT_NAME_TO_INDEX[keypoint_name], list_keypoints)))

    return index_list


def load_faceparser_model(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Loads FaceParser Model"""
    global faceparser_model

    faceparser_model = BiSeNet(len(KEYPOINT_INDEX_TO_NAME), backbone_name="resnet34")
    faceparser_model.to(device)
    faceparser_model.load_state_dict(torch.load(get_model_path(), map_location=device))
    faceparser_model.eval()

def objectify_keypoints(keypoints: np.ndarray[np.ndarray]) -> np.ndarray[Keypoint]:
    """
    objectify raw keypoints returned from FaceParser. Converts a list of 17 keypoints (y, x, confidence)
    """

    keypoint_list = [Keypoint(KeypointType(idx), kp[0]) for idx, kp in enumerate(keypoints)]

    return np.array(keypoint_list)


def parse_keypoints(image: np.ndarray, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Returns a list of keypoints parsed from the given image.

    FaceParser model should be loaded before call of this function

    Args:
        image: Image to parse keypoints from

    Returns:
        A list of raw keypoints data, referred as "frame" in this project
    """

    output = faceparser_model(torch.from_numpy(image).to(device))[0]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only
    predicted_mask = output.squeeze(0).cpu().detach().numpy().argmax(0)

    keypoints = []

    for i in range(len(KEYPOINT_INDEX_TO_NAME)):
        keypoints.append(predicted_mask == i)

    return keypoints

def resize_for_faceparser(image: np.ndarray):
    """Preprocesses image returned by cv2

    Args:
        image: Image to process
        input_size: Input size in pixels

    Returns:
        The processed image

    """

    resized_image = Image.fromarray(image).resize((512, 512), resample=Image.BILINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image_tensor = transform(resized_image)
    image_batch = image_tensor.unsqueeze(0)

    return image_batch.detach().numpy()[0]