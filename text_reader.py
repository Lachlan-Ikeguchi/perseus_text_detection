#! /usr/bin/python3.13

import pytesseract as tes
import cv2
from PIL import Image

tes.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

reference_image = Image.open('./reference_images/remove_before_flight.jpeg')

print(tes.imae_to_string(reference_image))
