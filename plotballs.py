import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks
from tqdm import tqdm

from gridmodel import GridModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--n_balls", type=int, nargs="?", default=3)
    args = parser.parse_args()

    model = GridModel(
        "../grid_models/grid_model_submovavg_64x64_light.h5",
        nBalls=args.n_balls,
        preprocessType="SUBMOVAVG",
        flip=False,
        postprocess=True,
    )

    cap = cv2.VideoCapture(args.path)
    ball_ys = np.empty((0, args.n_balls))
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            balls_and_hands = model.predict(frame)
            balls = balls_and_hands["balls"]
            ball_ys = np.vstack((ball_ys, balls[:, 1]))
            pbar.update()
    cap.release()

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    xs = np.arange(ball_ys.shape[0])
    ball_ys_blurred = gaussian_filter(ball_ys, sigma=(2, 0))

    ax1.set_title("Original ball positions")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("y position")
    for i in range(args.n_balls):
        ax1.plot(xs, ball_ys[:, i], "-o", label=f"Ball {i}")
    ax1.legend()

    ax2.set_title("Blurred ball positions")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("y position")
    for i in range(args.n_balls):
        ax2.plot(xs, ball_ys_blurred[:, i], "-o", label=f"Ball {i}")
    ax2.legend()

    ax3.set_title("Ball troughs")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("y position")
    for i in range(args.n_balls):
        ax3.plot(xs, ball_ys_blurred[:, i], "--", label=f"Ball {i}")
    for i in range(args.n_balls):
        troughs, _ = find_peaks(-ball_ys_blurred[:, i])
        ax3.plot(
            xs[troughs],
            ball_ys_blurred[troughs, i],
            "o",
            label=f"Ball {i} troughs",
            color=f"C{i}",
        )
    ax3.legend()

    plt.show()

