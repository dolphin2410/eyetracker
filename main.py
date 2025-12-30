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

import math
import random
import time
from recording.data_collector import LiveDataCollector
from util.faceparser_wrapper import load_faceparser_model
from util.yolo_wrapper import load_yolo_model
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

load_faceparser_model()
load_yolo_model()
live_data_collector = LiveDataCollector()
live_data_collector.start_collector()

def get_eyeball_loc():
    curr_frame = live_data_collector.application.camera_context.eyetracker_history.curr_frame
    return curr_frame.norm_x, curr_frame.norm_y

WINDOW_W = 800
WINDOW_H = 600

GRID_X = 5
GRID_Y = 4
JITTER = 15
SAMPLES_PER_POINT = 4
REST_TIME_MS = 500
SAMPLE_INTERVAL_MS = 50

CHAR_SIZE = 100
HIT_RADIUS = 60
RESPAWN_TIME = 1500
FPS = 60

root = tk.Tk()
root.title("Eye Tracking Calibration + Game")
root.update_idletasks()

screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()

x = (screen_w - WINDOW_W) // 2
y = (screen_h - WINDOW_H) // 2

root.geometry(f"{WINDOW_W}x{WINDOW_H}+{x}+{y}")

canvas = tk.Canvas(root, width=WINDOW_W, height=WINDOW_H, bg="black")
canvas.pack()

xs = np.linspace(0.1, 0.9, GRID_X) * WINDOW_W
ys = np.linspace(0.1, 0.9, GRID_Y) * WINDOW_H

calib_targets = []
for yy in ys:
    for xx in xs:
        calib_targets.append((
            xx + random.randint(-JITTER, JITTER),
            yy + random.randint(-JITTER, JITTER)
        ))

random.shuffle(calib_targets)

eye_samples = []
screen_samples = []

current_target_idx = 0
current_sample_idx = 0
target_dot_id = None

def draw_red_dot(x, y):
    global target_dot_id
    if target_dot_id:
        canvas.delete(target_dot_id)
    target_dot_id = canvas.create_oval(
        x - 20, y - 20,
        x + 20, y + 20,
        fill="red", outline=""
    )

def start_calibration():
    root.after(500, show_next_target)

def show_next_target():
    global current_sample_idx

    if current_target_idx >= len(calib_targets):
        finish_calibration()
        return

    tx, ty = calib_targets[current_target_idx]
    draw_red_dot(tx, ty)

    current_sample_idx = 0
    root.after(REST_TIME_MS, collect_one_sample)

def collect_one_sample():
    global current_sample_idx, current_target_idx

    ex, ey = get_eyeball_loc()
    tx, ty = calib_targets[current_target_idx]

    eye_samples.append([ex, ey, 1.0])
    screen_samples.append([tx, ty])

    current_sample_idx += 1

    if current_sample_idx < SAMPLES_PER_POINT:
        root.after(SAMPLE_INTERVAL_MS, collect_one_sample)
    else:
        current_target_idx += 1
        root.after(REST_TIME_MS, show_next_target)

def finish_calibration():
    global A

    canvas.delete("all")

    X = np.array(eye_samples)
    Y = np.array(screen_samples)

    A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    print("Calibration completed.")
    start_game()

def eye_to_screen(ex, ey):
    v = np.array([ex, ey, 1.0])
    sx, sy = v @ A
    return sx, sy

santa_img = Image.open("assets/santa.png").resize((CHAR_SIZE, CHAR_SIZE))
rudolph_img = Image.open("assets/rudolph.png").resize((CHAR_SIZE, CHAR_SIZE))

santa_tk = ImageTk.PhotoImage(santa_img)
rudolph_tk = ImageTk.PhotoImage(rudolph_img)

class Character:
    def __init__(self, image):
        self.image = image
        self.id = None
        self.x = 0
        self.y = 0
        self.alive = False

    def spawn(self):
        self.x = random.randint(CHAR_SIZE, WINDOW_W - CHAR_SIZE)
        self.y = random.randint(CHAR_SIZE, WINDOW_H - CHAR_SIZE)
        self.id = canvas.create_image(self.x, self.y, image=self.image)
        self.alive = True

    def kill(self):
        if self.id:
            canvas.delete(self.id)
        self.alive = False
        root.after(RESPAWN_TIME, self.spawn)

    def check_hit(self, gx, gy):
        if not self.alive:
            return
        if math.hypot(self.x - gx, self.y - gy) < HIT_RADIUS:
            self.kill()

characters = []
gaze_dot_id = None

def draw_gaze_dot(x, y):
    global gaze_dot_id
    if gaze_dot_id:
        canvas.delete(gaze_dot_id)
    gaze_dot_id = canvas.create_oval(
        x - 4, y - 4,
        x + 4, y + 4,
        fill="red", outline=""
    )

def game_loop():
    ex, ey = get_eyeball_loc()
    gx, gy = eye_to_screen(ex, ey)

    draw_gaze_dot(gx, gy)

    for c in characters:
        c.check_hit(gx, gy)

    root.after(int(1000 / FPS), game_loop)

def start_game():
    global characters
    characters = [
        Character(santa_tk),
        Character(rudolph_tk)
    ]
    for c in characters:
        c.spawn()
    game_loop()

start_calibration()
root.mainloop()