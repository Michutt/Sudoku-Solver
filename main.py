import cv2
import numpy as np
from tensorflow import keras


def img_binarise(img):
    # kernel = np.ones((3, 3), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return img


def img_color_inverse(img):
    return cv2.bitwise_not(img)


def create_digit_img(img, stat):
    digit_img = img[stat[0]:stat[0] + stat[2], stat[1]:stat[1] + stat[3]]
    return digit_img


def digits_detection(img):
    output = cv2.connectedComponentsWithStats(img)
    stats = output[2]
    mask = []

    for i in range(len(stats)):
        if stats[i, cv2.CC_STAT_AREA] > img.shape[0] * img.shape[1] / 81:
            mask.append(False)
        elif stats[i, cv2.CC_STAT_AREA] > img.shape[0] * img.shape[1] / 180:
            mask.append(True)
        else:
            mask.append(False)

    mask = np.where(mask)

    return stats[mask]


def sudoku_arr_create(model, digit_images, stats):
    sudoku_array = np.zeros((81, 1), int)
    for i in range(len(digit_images)):
        if stats[i, cv2.CC_STAT_AREA] < stats[i, cv2.CC_STAT_WIDTH]**2 - stats[i, cv2.CC_STAT_AREA] / 20:
            digit_images[i] = digit_images[i].astype('float32')
            digit_images[i] /= 255
            p = model.predict(digit_images[i]).argmax()
            sudoku_array[i] = p


            # digit_images[i] = digit_images[i].reshape(28, 28)
            # print(p)
            # cv2.imshow('bin', digit_images[i])
            # cv2.waitKey(0)
        else:
            sudoku_array[i] = 0

    return sudoku_array.reshape(9,9)


img_in = cv2.imread("img2.jpg")

img = img_binarise(img_in)
stats = digits_detection(img)
print(len(stats))
digit_images = []

for i in range(len(stats)):
    digit_images.append(create_digit_img(img, stats[i]))
    digit_images[i] = cv2.bitwise_not(digit_images[i])
    digit_images[i] = cv2.resize(digit_images[i], (28, 28), interpolation=cv2.INTER_AREA)
    digit_images[i] = digit_images[i].reshape(1, 784)


reconstructed_model = keras.models.load_model('model')

sudoku_array = sudoku_arr_create(reconstructed_model, digit_images, stats)

print(sudoku_array)

cv2.imshow('bin', img)
cv2.waitKey(0)



# img1 = cv2.imread("beep1.jpg")
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img1 = cv2.bitwise_not(img1).reshape(1, 784)
# prediction = reconstructed_model.predict(img1).argmax()
# print(prediction)