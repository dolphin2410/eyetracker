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

from recording.data_collector import LiveDataCollector
from util.faceparser_wrapper import load_faceparser_model
from util.yolo_wrapper import load_yolo_model
import tkinter as tk
import numpy as np

load_faceparser_model()
load_yolo_model()
live_data_collector = LiveDataCollector()
live_data_collector.start_collector()

def get_eyeball_loc():
    curr_frame = live_data_collector.application.camera_context.eyetracker_history.curr_frame
    return curr_frame.norm_x, curr_frame.norm_y

def generate_random_targets(width, height, n_points=20, margin_ratio=0.08, min_dist_ratio=0.12):
    margin = int(min(width, height) * margin_ratio)
    min_dist = int(min(width, height) * min_dist_ratio)

    targets = []
    while len(targets) < n_points:
        x = np.random.uniform(margin, width - margin)
        y = np.random.uniform(margin, height - margin)

        if all((x - tx) ** 2 + (y - ty) ** 2 > min_dist ** 2 for tx, ty in targets):
            targets.append((x, y))
    return targets


class EyeTrackerCalibration:
    def __init__(self, root):
        self.root = root

        root.attributes("-fullscreen", True)
        root.configure(bg="black")
        root.bind("<Escape>", lambda e: root.destroy())

        self.W = root.winfo_screenwidth()
        self.H = root.winfo_screenheight()

        self.canvas = tk.Canvas(
            root,
            width=self.W,
            height=self.H,
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        self.targets = generate_random_targets(self.W, self.H, n_points=20)
        self.samples_per_target = 4
        self.sample_rest_ms = 500
        self.initial_stabilize_ms = 700

        self.eye_samples = []
        self.screen_samples = []

        self.current_target_idx = 0
        self.current_eye_buffer = []

        self.A = None
        self.b = None

        self.prev_screen_point = None

        self.root.after(1000, self.show_target)

    def show_target(self):
        if self.current_target_idx >= len(self.targets):
            self.finish_calibration()
            return

        self.canvas.delete("all")

        x, y = self.targets[self.current_target_idx]
        r = 12

        self.canvas.create_oval(
            x - r, y - r,
            x + r, y + r,
            fill="red"
        )

        self.current_eye_buffer = []

        self.root.after(self.initial_stabilize_ms, self.collect_sample)

    def collect_sample(self):
        ex, ey = get_eyeball_loc()
        self.current_eye_buffer.append([ex, ey])

        if len(self.current_eye_buffer) < self.samples_per_target:
            self.root.after(self.sample_rest_ms, self.collect_sample)
        else:
            mean_eye = np.mean(self.current_eye_buffer, axis=0)

            self.eye_samples.append(mean_eye)
            self.screen_samples.append(self.targets[self.current_target_idx])

            self.current_target_idx += 1
            self.show_target()

    def finish_calibration(self):
        print("Calibration finished")
        self.fit_affine_model()
        self.run_tracking()

    def fit_affine_model(self):
        E = np.array(self.eye_samples)
        S = np.array(self.screen_samples)

        ones = np.ones((E.shape[0], 1))
        X = np.hstack([E, ones])

        W, _, _, _ = np.linalg.lstsq(X, S, rcond=None)

        self.A = W[:2, :].T
        self.b = W[2, :]

        print("A =\n", self.A)
        print("b =", self.b)

    def eye_to_screen(self, ex, ey):
        return self.A @ np.array([ex, ey]) + self.b

    def run_tracking(self):
        self.canvas.delete("all")
        self.prev_screen_point = None
        self.track_loop()

    def track_loop(self):
        ex, ey = get_eyeball_loc()
        sx, sy = self.eye_to_screen(ex, ey)

        if self.prev_screen_point is not None:
            px, py = self.prev_screen_point
            self.canvas.create_line(
                px, py, sx, sy,
                fill="lime",
                width=2
            )

        self.prev_screen_point = (sx, sy)
        self.root.after(16, self.track_loop)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Eye Tracker Calibration (with Rest Time)")

    app = EyeTrackerCalibration(root)
    root.mainloop()
