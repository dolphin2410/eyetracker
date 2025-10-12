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
