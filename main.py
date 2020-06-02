import cv2
import numpy as np


def img_preparation(img):
    # kernel = np.ones((10,10), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return img

def digits_detection(img):
    output = cv2.connectedComponentsWithStats(img)
    stats = output[2]
    mask = []

    stats = np.delete(stats, 0, 0)

    for i in range(len(stats)):
        if stats[i, cv2.CC_STAT_AREA] > img.shape[1] * img.shape[1] / 140:
            mask.append(True)
        else:
            mask.append(False)

    mask = np.where(mask)

    return stats[mask]



img_in = cv2.imread("img.jpg")

img = img_preparation(img_in)
stats = digits_detection(img)


cv2.imshow('bin', img)
cv2.waitKey(0)
