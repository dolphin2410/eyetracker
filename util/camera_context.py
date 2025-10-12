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

from util.eyetracker_history import EyetrackerHistory
from util.timer import Timer


class CameraContext:
    """Shared data for labeled image

    Since a new LabeledImage object is initialized per frame, shared data needs to be stored in a separate object.
    CameraContext stores shared data.

    Attributes:
        timer:
            Timer instance
        settings:
            Dictionary to save shared variables
    """

    def __init__(self):
        """Initializes a new CameraContext object

        Initializes a new timer, a history timer and an empty dictionary named "settings", which is used to
        save extra shared data.
        """

        self.timer = Timer()
        self.settings = {}
        self.history_manager = EyetrackerHistory()