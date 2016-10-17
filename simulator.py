import numpy as np
import cv2
from math import sin, cos, acos, pi

import pygame


STEER_RATIO = 14.8
SPEED_CONVERT = 17.6
WHEEL_BASE = 112.5
ERROR_LIMIT = pi/4


def initialize_pygame(w, h):
    pygame.init()
    font = pygame.font.Font(None,30)

    size = (w, h)
    pygame.display.set_caption("Brahms - Udacity Self Driving Car Simulator")
    screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
    camera_surface = pygame.surface.Surface(size, 0, 24).convert()
    return screen, camera_surface, font


def rotate(im, theta, isdeg=True):
    if isdeg:
        theta = np.deg2rad(theta)
    f = 2.
    h, w, _ = im.shape

    cx = cz = cos(0)
    sx = sz = sin(0)
    cy = cos(theta)
    sy = sin(theta)

    R = np.array([
        [cz * cy, cz * sy * sx - sz * cx],
        [sz * cy, sz * sy * sx + cz * cx],
        [ -sy, cy * sx]
    ], np.float32)

    pts1 = [
        [-w/2, -h/2],
        [w/2, -h/2],
        [w/2, h/2],
        [-w/2, h/2]
    ]

    pts2 = []
    mx, my = 0, 0
    for i in range(4):
        pz = pts1[i][0] * R[2][0] + pts1[i][1] * R[2][1];
        px = w / 2 + (pts1[i][0] * R[0][0] + pts1[i][1] * R[0][1]) * f * h / (f * h + pz);
        py = h / 2 + (pts1[i][0] * R[1][0] + pts1[i][1] * R[1][1]) * f * h / (f * h + pz);
        pts2.append([px, py])


    pts2 = np.array(pts2, np.float32)
    pts1 = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], np.float32)

    x1, x2 = int(min(pts2[0][0], pts2[3][0])), int(max(pts2[1][0], pts2[2][0]))
    y1, y2 = int(max(pts2[0][1], pts2[1][1])),  int(min(pts2[2][1], pts2[3][1]))

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(im, M, (w, h), cv2.INTER_NEAREST | cv2.INTER_NEAREST)

    x1 = np.clip(x1, 0, w)
    x2 = np.clip(x2, 0, w)
    y1 = np.clip(y1, 0, h)
    y2 = np.clip(y2, 0, h)
    z = dst[y1:y2, x1:x2]
    x, y, _ = z.shape
    if x == 0 or y == 0:

        return
    return cv2.resize(z, (w, h), interpolation = cv2.INTER_AREA)


def calculate_heading(steering, speed, delta_time, steer_ratio, speed_convert, wheel_base):
    wheel_angle = steering/steer_ratio
    if abs(wheel_angle) < 1e-8:
        return 0

    yspeed = speed * steer_ratio * sin(wheel_angle)
    turning_radius = abs(wheel_base/sin(wheel_angle))
    y_disp = abs(yspeed) * float(delta_time)/(10**9)
    theta = acos((turning_radius-y_disp)/turning_radius)
    return np.sign(steering) * abs(theta)


def simulate(model, data_iter, width, height, steer_ratio=STEER_RATIO, speed_convert=SPEED_CONVERT,
             wheel_base=WHEEL_BASE, error_limit=ERROR_LIMIT):
    '''
    data_iter should produce elements of the following form:
        (image_pred, image_disp, speed, steering, ts)
        where
            speed in miles/second
            steering in radians
            ts in nanoseconds
            speed_convert converts the speed to inches/sec
    '''

    screen, camera_surface, font = initialize_pygame(width, height)
    last_ts = st = 0
    theta_error = 0
    last_steering = last_prediction = last_speed = 0

    for image_pred, image_disp, speed, steering, ts in data_iter:
        if last_ts:
            delta_time = ts - last_ts
            theta_actual = calculate_heading(last_steering,
                                             last_speed,
                                             delta_time,
                                             steer_ratio,
                                             speed_convert,
                                             wheel_base)
            theta_prediction = calculate_heading(last_prediction,
                                                 last_speed,
                                                 delta_time,
                                                 steer_ratio,
                                                 speed_convert,
                                                 wheel_base)
            theta_error = theta_error + (theta_actual - theta_prediction)

        if abs(theta_error) > error_limit:
            print "Error limit reached"
            return float(ts - st)/10**9

        if st == 0:
            st = ts

        image_pred = rotate(image_pred, theta_error, False)
        if image_pred is None:
            print "Too much error", theta_error
            return float(ts - st)/10**9

        image_disp = rotate(image_disp, theta_error, False)
        predicted_steering = model.predict(image_pred[None, :, :, :])[0][0]

        last_steering = steering
        last_prediction = predicted_steering
        last_ts = ts
        last_speed = speed

        pygame.surfarray.blit_array(camera_surface, image_disp.swapaxes(0,1))
        screen.blit(camera_surface, (0,0))
        label = font.render("Actual = %.3f, Predicted = %.3f, Error = %.3f" % (steering, predicted_steering, theta_error),
                            1, (255,255,255))
        screen.blit(label, (10, 440))
        pygame.display.flip()

    return float(ts - st)/10**9
