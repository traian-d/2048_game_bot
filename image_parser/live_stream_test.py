from __future__ import division

import time

import cv2
import picamera
from picamera.array import PiRGBArray

import find_contours as fc
import main

with picamera.PiCamera() as camera:
    avg = None
    i = 0
    camera.resolution = (1024, 768)
    camera.framerate = 5
    rawCapture = PiRGBArray(camera, size=camera.resolution)
    camera.start_preview()
    time.sleep(2)
    for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image
        frame = f.array
        print(i)
        i += 1
        # resize the frame, convert it to grayscale, and blur it
        frame_res = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame_res, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        # if the average frame is None, initialize it
        if avg is None:
            avg = blur.copy().astype("float")
            rawCapture.truncate(0)
            continue
        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(blur, avg, 0.5)
        frameDelta = cv2.absdiff(blur, cv2.convertScaleAbs(avg))
        if sum(sum(frameDelta)) < 60000:
            grid_contour = fc.extract_game_grid(gray)
            for o in camera.overlays:
                camera.remove_overlay(o)
            if grid_contour is not None:
                capture = gray
                original_points, predictions = main.predict(capture)
                empty_img = main.make_empty_image_duplicate(capture)
                main.add_points_to_image(empty_img, original_points, predictions)
                cv2.drawContours(empty_img, [grid_contour], 0, (0, 255, 0), 3)
                o = camera.add_overlay(empty_img.tostring(), size=(1024, 768))
                o.alpha = 80
                o.layer = 3
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
    camera.stop_preview()
