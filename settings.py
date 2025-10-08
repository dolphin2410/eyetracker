#     Copyright (C) 2024 dolphin2410
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

import os

def eyetracker_configuration(configuration_name):
    configuration_value = str(os.environ.get(configuration_name))
    print(f"{configuration_name}: {configuration_value}")
    return lambda x: x

@eyetracker_configuration("eyetracker_dbg")
def is_debug():
    """Returns whether this project is run in debug mode

    Returns:
        Whether this project is run in debug mode, defaulting to True
    """
    return bool(os.environ.get("eyetracker_dbg", "True"))


@eyetracker_configuration("eyetracker_weights_path")
def get_model_path():
    """Returns weights path

    Returns:
        Weight Path specified in the environment variable, or a default weights location
    """
    return str(os.environ.get("eyetracker_weights_path", "./face_parsing/weights/resnet34.pt"))


@eyetracker_configuration("eyetracker_mjpeg_port")
def get_mjpeg_port():
    """Returns mjpeg server port

    Returns:
        Set MJPEG server port, or a default port
    """
    return int(os.environ.get("eyetracker_mjpeg_port", "8080"))


@eyetracker_configuration("eyetracker_mjpeg_channel_name")
def get_mjpeg_channel_name():
    """Returns mjpeg server port

    Returns:
        Set MJPEG server port, or a default port
    """
    return str(os.environ.get("eyetracker_mjpeg_channel_name", "my_camera"))


@eyetracker_configuration("eyetracker_target_model_path")
def get_target_model_path():
    """Returns target model path to save after training

    Returns:
        Set target model path, or a default path
    """

    return str(os.environ.get("eyetracker_target_model_path", "./model/eyetracker_model.keras"))


@eyetracker_configuration("eyetracker_frames_per_sample")
def get_frames_per_sample():
    """Frames per sample, defaults to 25"""

    return int(os.environ.get("eyetracker_frames_per_sample", "32"))


@eyetracker_configuration("eyetracker_max_frames_in_history")
def get_max_frames_in_history():
    """Max frames in history, defaults to 100"""

    return int(os.environ.get("eyetracker_max_frames_in_history", "32"))
