import numpy as np
from core.application import EyetrackerApplication
from util.camera_context import CameraContext
from util.eyetracker_history import EyetrackerHistoryFrame
from util.eyetracker_image import EyetrackerImage
from util.faceparser_wrapper import KEYPOINT_INDEX_TO_NAME

def application_callback(image: EyetrackerImage, camera_context: CameraContext):
    image.raw_image = camera_context.timer.display_timer(image.raw_image)

    if "start_record" in camera_context.settings:
        if camera_context.settings["start_record"]:
            frame = EyetrackerHistoryFrame(image.keypoints)
            camera_context.history_manager.append_frame(frame)
        else:
            camera_context.history_manager.save_history()
            del camera_context.settings["start_record"]

    matrix = np.zeros((512, 512))
    for keypoint_index, keypoint_data in enumerate(image.keypoints):
        keypoint_name = KEYPOINT_INDEX_TO_NAME[keypoint_index]
        if keypoint_name == "left_eye" or keypoint_name == "right_eye":
            matrix[keypoint_data] = 1

    image.raw_image = image.yolo_parsed_image
    
    if matrix.sum() == 0:
        image.raw_image = np.zeros((512, 512, 3)).astype(np.uint8)
        return

    x_list = np.where(np.any(matrix, axis=0))[0]
    y_list = np.where(np.any(matrix, axis=1))[0]

    new_mask = matrix[y_list[0]:y_list[-1], x_list[0]:x_list[-1]]
    new = image.raw_image[y_list[0]:y_list[-1], x_list[0]:x_list[-1]]
    new[new_mask == 0, :] = 0

    # top right: 64 20 56 18 -> 오른쪽일수록 흰색이 오른쪽으로
    # top left: 42 16 59 13
    # bottom left 50 19 63 17 -> 아래볼수록
    # bottom right 83 12 57 15

    white_part = np.logical_and(new.mean(axis=2) >= 50, new_mask == 1)  # TODO: Fix this hardcoded value
    black_part = np.logical_and(new.mean(axis=2) < 50, new_mask == 1)

    if white_part.sum() == 0 or black_part.sum() == 0:
        return

    x_pos_mask = np.arange(new_mask.shape[1])
    y_pos_mask = np.arange(new_mask.shape[0]).reshape(-1, 1).repeat(new_mask.shape[1], axis=1)
    print("흰자 x좌표\t", (x_pos_mask * white_part).sum() / white_part.sum())
    print("흰자 y좌표\t", (y_pos_mask * white_part).sum() / white_part.sum())
    print("검은자 x좌표\t", (x_pos_mask * black_part).sum() / black_part.sum())
    print("검은자 y좌표\t", (y_pos_mask * black_part).sum() / black_part.sum())
    print("----")

    new[white_part] = 255
    new[black_part] = 100

    image.raw_image = np.zeros((512, 512, 3)).astype(np.uint8)
    image.raw_image[:new.shape[0], :new.shape[1]] = new

        
def exit_callback(application: EyetrackerApplication):
    camera_context = application.camera_context

    if "start_record" in camera_context.settings and camera_context.settings["start_record"]:
        camera_context.history_manager.save_history()
        del camera_context.settings["start_record"]

class LiveDataCollector():
    def __init__(self):
        self.application = EyetrackerApplication(0)

    def start_collector(self):
        camera_context = self.application.camera_context

        def start_record():
            camera_context.settings["start_record"] = True
            print("recording start!")

        def end_record():
            camera_context.settings["start_record"] = False
            print("recording ended!!!")

        camera_context.timer.register_action(3, start_record)
        camera_context.timer.register_action(5, end_record)

        self.application.start_application(application_callback, exit_callback)

class VideoDataCollector():
    def __init__(self, video_path):
        self.video_path = video_path
        self.application = EyetrackerApplication(video_path)

    def start_collector(self):
        self.application.camera_context.settings["start_record"] = True
        self.application.start_application(application_callback, exit_callback)
