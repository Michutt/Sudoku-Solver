import cv2
import numpy as np


def img_preparation(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

    return img


img_in = cv2.imread("img.jpg")
img = img_preparation(img_in)

cv2.imshow('bin', img)
cv2.waitKey(0)