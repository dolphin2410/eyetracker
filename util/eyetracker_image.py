import cv2
import numpy as np

from util import faceparser_wrapper
from util.camera_context import CameraContext


class EyetrackerImage:
    """
    Wrapper class for images

    Attributes:
        raw_image: Raw image data in numpy array
        camera_context: CameraContext instance
        keypoints: FaceParser keypoints data
    """
    
    def __init__(self, original_image: np.ndarray, image: np.ndarray, camera_context: CameraContext):
        """Initializes a LabeledImage object

        Args:
            image: Raw image in numpy array format
            camera_context: CameraContext instance
        """

        self.raw_image = original_image
        self.camera_context = camera_context
        self.keypoints = faceparser_wrapper.parse_keypoints(np.array([image]))
