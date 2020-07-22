#!/usr/bin/env python3
import time
import argparse

import numpy as np
import cv2
from keras.models import load_model

from utils import handleTensorflowSession
from drawingutils import drawBallsAndHands
from gridmodel import GridModel
from frameratechecker import FramerateChecker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--balls", type=int, default=3)
    args = parser.parse_args()

    handleTensorflowSession(memoryLimit=0.2)

    gridModel = GridModel("../grid_models/grid_model_submovavg_64x64.h5", args.balls)
    cap = cv2.VideoCapture(0)
    framerateChecker = FramerateChecker(expected_fps=30)


    while True:
        framerateChecker.check()
        ret, original_img = cap.read()
        if not ret:
            print("Couldn't get frame from camera.")
            break
        else:
            height, width, channels = original_img.shape
            tocrop = int((width - height) / 2)
            original_img = original_img[:,tocrop:-tocrop]
            ballsAndHands = gridModel.predict(original_img.copy())

            img = cv2.resize(original_img, (256,256), cv2.INTER_CUBIC)
            drawBallsAndHands(img, ballsAndHands)

            cv2.imshow('Webcam', img)
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
