#! /usr/bin/env /usr/bin/python3.13

import numpy as np
import cv2
# import imutils
from imutils.object_detection import non_max_suppression
from imutils import rotate_bound

layer_names = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]


confidence_cutoff = 0.5
image_rescale_factor = 1


def east_detect(image):
    original_image = image.copy()

    height = image.shape[0]
    width = image.shape[1]

    # If the width and height are multiples of 32, add 10 pixels so it isn't.  Probably could be better written 😅
    height_temp = height
    width_temp = width

    if height_temp % 32 == 0:
        height_temp += 10

    if width_temp % 32 == 0:
        width_temp += 10

    # for both the width and height: Should be multiple of 32
    new_height = int((height_temp % 32) * image_rescale_factor) * 32
    new_width = int((width_temp % 32) * image_rescale_factor) * 32

    # resize scaling
    resize_width = width / float(new_width)
    resize_height = height / float(new_height)

    image = cv2.resize(image, (new_width, new_height))

    # update after resize
    height = image.shape[0]
    width = image.shape[1]

    EAST = cv2.dnn.readNet("model/frozen_east_text_detection.pb")

    image_blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (width, height),
        swapRB=True,
        crop=False
    )

    EAST.setInput(image_blob)

    (scores, geometry) = EAST.forward(layer_names)

    (number_of_rows, number_of_columns) = scores.shape[2:4]

    rectangles = []
    confidences = []

    for y in range(0, number_of_rows):
        scores_data = scores[0, 0, y]

        tops = geometry[0, 0, y]
        lefts = geometry[0, 1, y]
        bottoms = geometry[0, 2, y]
        rights = geometry[0, 3, y]

        angles = geometry[0, 4, y]

        for x in range(0, number_of_columns):
            if scores_data[x] < confidence_cutoff:
                continue

            (offset_x, offset_y) = (x * 4.0, y * 4.0)

            angle = angles[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            bounding_box_height = tops[x] + bottoms[x]
            bounding_box_width = lefts[x] + rights[x]

            end_x = int(
                offset_x + (cos * lefts[x]) + (sin * bottoms[x])
            )

            end_y = int(
                offset_y - (sin * lefts[x]) + (cos * bottoms[x])
            )

            start_x = int(end_x - bounding_box_width)
            start_y = int(end_y - bounding_box_height)

            rectangles.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    # boxes = imutils.object_detection.non_max_suppression(
    boxes = non_max_suppression(
        np.array(rectangles), probs=confidences)

    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * resize_width)
        start_y = int(start_y * resize_height)
        end_x = int(end_x * resize_width)
        end_y = int(end_y * resize_height)

        cv2.rectangle(
            original_image,
            (start_x, start_y),
            (end_x, end_y),
            (0, 255, 0),
            5
        )

    return original_image


image = cv2.imread("remove_before_flight.jpeg")

increments = 45
for angle in range(0, int(360 / increments)):
    out_image = east_detect(rotate_bound(image, angle * increments))

    cv2.imwrite(f"output/output-{angle}.jpg", out_image)
