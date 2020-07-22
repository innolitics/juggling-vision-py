import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
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

    fig, ax = plt.subplots()
    ax.set_xlabel("frame")
    ax.set_ylabel("y position")
    xs = np.arange(ball_ys.shape[0])
    for i in range(ball_ys.shape[1]):
        ax.plot(xs, ball_ys[:, i], "-o", label=f"Ball {i}")
    ax.legend()
    plt.show()

