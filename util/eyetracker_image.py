import cv2
import numpy as np

from util import faceparser_wrapper
from util.camera_context import CameraContext
from util.yolo_wrapper import parse_face


class EyetrackerImage:
    """
    Wrapper class for images

    Attributes:
        raw_image: Raw image data in numpy array
        camera_context: CameraContext instance
        keypoints: FaceParser keypoints data
    """
    
    def __init__(self, original_image: np.ndarray, camera_context: CameraContext):
        """Initializes a LabeledImage object

        Args:
            image: Raw image in numpy array format
            camera_context: CameraContext instance
        """

        self.raw_image = original_image
        self.camera_context = camera_context

        parsed_face, x_size, y_size = parse_face(original_image)
        self.yolo_parsed_image = cv2.resize(original_image[parsed_face].reshape(y_size, x_size, 3), original_image.shape[0:2])

        image_for_faceparser = np.array([faceparser_wrapper.resize_for_faceparser(self.yolo_parsed_image)])
        self.keypoints = faceparser_wrapper.parse_keypoints(image_for_faceparser)
