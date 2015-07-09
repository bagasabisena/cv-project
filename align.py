from __future__ import division

import math
from PIL import Image
import cv2
import numpy as np

def align_image(image, eye_left, eye_right, dest_size):

    offset = [0.3, 0.3]

    offset_h = math.floor(float(offset[0]))*dest_size[0]
    offset_v = math.floor(float(offset[1]))*dest_size[1]

    # check if the image is a bit rotated by looking at the eye
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    rotation_angle = eye_direction[0] / eye_direction[1]
    
    # calculate distance of the left and right eye
    dx = eye_right[0] - eye_left[0]
    dy = eye_right[1] - eye_left[1]
    dist = math.sqrt(dx*dx + dy*dy)

    # compare the actual eye width with the intended width
    # this is used for scaling the image
    reference = dest_size[0] - 2.0*offset_h
    scale = dist / reference

    # rotate and translate the image
    x = eye_left[0]
    y = eye_left[1]
    a = math.cos(rotation_angle) / scale
    b = math.sin(rotation_angle) / scale
    c = x - x*a - y*b
    d = -1 * math.sin(rotation_angle) / scale
    e = math.cos(rotation_angle) / scale
    f = y - x*d - y*e
    transformed_image = image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=Image.BICUBIC)

    # crop image
    crop_pos = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
    crop_size = (dest_size[0] * scale, dest_size[1] * scale)
    cropped_image = transformed_image.crop((int(crop_pos[0]), int(crop_pos[1]), int(crop_pos[0]+crop_size[0]), int(crop_pos[1]+crop_size[1])))
    cropped_image = cropped_image.resize(dest_size, Image.ANTIALIAS)

    return cropped_image

def eye_detector(image):

    np_image = np.array(image.convert('L'), dtype='uint8')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(np_image)

    # estimate eye coordinate as center of the eye rectangle returned by Viola Jones
    # detector can detect false eyes
    # currently only set two best detected image
    eyes = eyes[0:1,:]
    eye_center = np.zeros((2, 2))
    for i, (x, y, width, height) in enumerate(eyes):
        x_center = width / 2 + x
        y_center = height / 2 + y
        eye_center[i, :] = [x_center, y_center]

    return eye_center



