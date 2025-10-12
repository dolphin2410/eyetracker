import cv2

from util.camera_context import CameraContext
from util.eyetracker_image import EyetrackerImage


class EyetrackerApplication:
    def __init__(self, video_source):
        self.video_source = video_source
        self.camera_context = CameraContext()

    def start_application(self, application_callback = None, exit_callback = None):
        """
        application_callback: EyetrackerImage -> None
        exit_callback: EyetrackerApplication -> None
        """
        
        video_capture = cv2.VideoCapture(self.video_source)

        while video_capture.isOpened():
            try:
                _, frame = video_capture.read()

                labeled_image = EyetrackerImage(cv2.resize(frame, (512, 512)), self.camera_context)
                
                if application_callback is not None:
                    application_callback(labeled_image)

                self.camera_context.timer.increment_tick_lazy()  # increment timer tick

                cv2.imshow('frame', labeled_image.raw_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as error:
                print(error)
                break
        
        if exit_callback is not None:
            exit_callback(self)

        video_capture.release()
        cv2.destroyAllWindows()
    