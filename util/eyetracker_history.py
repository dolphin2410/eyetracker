class EyetrackerHistoryFrame():
    """A wrapper class for faceparser keypoints"""

    def __init__(self, keypoints):
        self.keypoints = keypoints

class EyetrackerHistory():
    """This class stores a time series of faceparser keypoints"""
    
    def __init__(self):
        self.frames = []

    def append_frame(self, frame: EyetrackerHistoryFrame):
        self.frames.append(frame)

    def save_history(self):
        pass # todo implement
