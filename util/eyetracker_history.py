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
import json

def parse_offset_from_vector(v):
    v = np.array(v)

    mean = np.mean(v)

    return (v[1] - mean) / mean

class EyetrackerHistoryFrame():
    """A wrapper class for faceparser keypoints"""

    def __init__(self, x_w, y_w, x_b, y_b):
        self.raw_points_x = [x_w, x_b]
        self.raw_points_y = [y_w, y_b]

        self.norm_x = parse_offset_from_vector(self.raw_points_x)
        self.norm_y = parse_offset_from_vector(self.raw_points_y)

class EyetrackerHistory():
    """This class stores a time series of faceparser keypoints"""
    
    def __init__(self):
        self.curr_frame = None
        self.frames = []

    def append_frame(self, frame: EyetrackerHistoryFrame):
        self.frames.append(frame)

    def save_history(self, target_file):
        with open(target_file, 'w') as wf:
            json.dump(list(map(lambda x: x.__dict__, self.frames)), wf)
