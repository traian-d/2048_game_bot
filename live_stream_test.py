# # import the necessary packages
# import datetime
# import time
#
# import cv2
# from picamera import PiCamera
# from picamera.array import PiRGBArray
#
# # {
# # 	"show_video": true,
# # 	"use_dropbox": false,
# # 	"dropbox_key": "YOUR_DROPBOX_KEY",
# # 	"dropbox_secret": "YOUR_DROPBOX_SECRET",
# # 	"dropbox_base_path": "YOUR_DROPBOX_APP_PATH",
# #     "use_email": false,
# #     "email_address": ["your email address"],
# # 	"min_upload_seconds": 0.5,
# # 	"min_motion_frames": 8,
# # 	"camera_warmup_time": 10,
# # 	"delta_thresh": 5,
# # 	"blur_size": [21, 21],
# # 	"resolution": [640, 480],
# # 	"fps": 16,
# # 	"min_area": 5000
# # }
#
# # initialize the camera and grab a reference to the raw camera capture
# camera = PiCamera()
# camera.resolution = (1024, 768)
# camera.framerate = 16
# rawCapture = PiRGBArray(camera, size=camera.resolution)
# # camera.start_preview()
# # capture frames from the camera
# for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#     # grab the raw NumPy array representing the image and initialize
#     # the timestamp and occupied/unoccupied text
#     frame = f.array
#     camera.a =
#     # draw the text and timestamp on the frame
#     cv2.putText(frame, "hello world !", (10, 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#     # clear the stream in preparation for the next frame
#     # cv2.imshow("", frame)
#     # cv2.waitKey(0)
#     rawCapture.truncate(0)

from __future__ import division

import picamera
import numpy as np
import cv2
import time
from picamera.array import PiRGBArray
import find_contours as fc
import main
import extract_digits as ed

motion_dtype = np.dtype([
    ('x', 'i1'),
    ('y', 'i1'),
    ('sad', 'u2'),
    ])

class MyMotionDetector(object):
    def __init__(self, camera):
        width, height = camera.resolution
        self.cols = (width + 15) // 16
        self.cols += 1 # there's always an extra column
        self.rows = (height + 15) // 16

    def write(self, s):
        # Load the motion data from the string to a numpy array
        data = np.fromstring(s, dtype=motion_dtype)
        # Re-shape it and calculate the magnitude of each vector
        data = data.reshape((self.rows, self.cols))
        data = np.sqrt(
            np.square(data['x'].astype(np.float)) +
            np.square(data['y'].astype(np.float))
            ).clip(0, 255).astype(np.uint8)
        # If there're more than 10 vectors with a magnitude greater
        # than 60, then say we've detected motion
        if (data > 60).sum() > 10:
            print('Motion detected!')
        # Pretend we wrote all the bytes of s
        return len(s)


# img = cv2.imread('test_photos/image265.jpg')
# (height, width, channels) = img.shape
# blank_image = np.zeros((height, width, 3), np.uint8)
# cv2.putText(blank_image, "hello world!",
#             (300, 300), cv2.FONT_HERSHEY_SIMPLEX,
#             fontScale=1, color=(255, 255, 255), lineType=2, thickness=2)

with picamera.PiCamera() as camera:
    avg = None
    i = 0
    camera.resolution = (1024, 768)
    camera.framerate = 5
    rawCapture = PiRGBArray(camera, size=camera.resolution)
    camera.start_preview()
    time.sleep(2)
    for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image and initialize
        # the timestamp and occupied/unoccupied text
        frame = f.array
        print(i)
        i += 1

        # resize the frame, convert it to grayscale, and blur it
        frame_res = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame_res, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the average frame is None, initialize it
        if avg is None:
            avg = gray.copy().astype("float")
            rawCapture.truncate(0)
            continue

        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        if sum(sum(frameDelta)) < 60000:
            # if i < 10:
            grid_contour = fc.contains_game_grid(frame)
            for o in camera.overlays:
                camera.remove_overlay(o)
            if grid_contour is not None:
                # cv2.imshow('frame', ed.threshold_image(frame))
                # camera.capture('capture.png')
                # capture = cv2.imread('capture.png')
                # cv2.imshow('image from file', ed.threshold_image(capture))
                # cv2.waitKey(0)
                # cv2.imshow('frame', ed.threshold_image(frame))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                capture = frame
                original_points, predictions = main.predict(capture)
                # if sum(predictions) == 0:
                #     cv2.imwrite('post_mortem.png', frame)
                empty_img = main.make_empty_image_duplicate(capture)
                main.add_points_to_image(empty_img, original_points, predictions)
                cv2.drawContours(empty_img, [grid_contour], 0, (0, 255, 0), 3)
                o = camera.add_overlay(empty_img.tostring(), size=(1024, 768))
                o.alpha = 80
                o.layer = 3
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
    time.sleep(150)
    camera.stop_preview()
