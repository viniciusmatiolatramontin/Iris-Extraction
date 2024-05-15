#Vinícius Matiola Tramontin

import cv2
import numpy as np
from matplotlib import pyplot as plt


def hough(img, cont, window_num):
    fig = plt.figure(window_num, figsize=(8, 8))

    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 2, 160, param1=20, param2=20, minRadius=100, maxRadius=180)
    circles = np.uint16(np.around(circles))

    x_iris = circles[0][0][0]
    y_iris = circles[0][0][1]
    radius_iris = circles[0][0][2]

    mask_iris = np.zeros_like(img)
    mask_iris = cv2.circle(mask_iris, (x_iris,y_iris), radius_iris, (255,255,255), -1)

    res = cv2.bitwise_and(img, img, mask = mask_iris)

    circles_pupil = cv2.HoughCircles(res, cv2.HOUGH_GRADIENT, 2, 160, param1=20, param2=20, minRadius=54, maxRadius=80)
    circles_pupil = np.uint16(np.around(circles_pupil))

    x_pupil = circles_pupil[0][0][0]
    y_pupil = circles_pupil[0][0][1]
    radius_pupil = circles_pupil[0][0][2]

    mask_pupil = np.zeros_like(img)
    mask_pupil = cv2.circle(mask_pupil, (x_pupil,y_pupil), radius_pupil, (255,255,255), -1)

    mask = cv2.subtract(mask_iris, mask_pupil)

    res = cv2.bitwise_and(res, res, mask = mask)

    plt.gray()

    fig.add_subplot(2, 2, cont)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    fig.add_subplot(2, 2, cont+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(res)




window_num = 0

for i in range(1, 11):
    for j in range(1, 6):
        imgR = None
        imgL = None

        caminhoR = ''
        caminhoL = ''

        if i < 10:
            caminhoR = './Amostras/00' + str(i) + '_R_0' + str(j) + '.JPG'
            caminhoL = './Amostras/00' + str(i) + '_L_0' + str(j) + '.JPG'
        else:
            caminhoR = './Amostras/0' + str(i) + '_R_0' + str(j) + '.JPG'
            caminhoL = './Amostras/0' + str(i) + '_L_0' + str(j) + '.JPG'

        imgR = cv2.imread(caminhoR)
        imgL = cv2.imread(caminhoL)

        croppedR = imgR[500:500+900, 700:700+1200].copy()
        croppedL = imgL[500:500+900, 700:700+1200].copy()

        grayR = cv2.cvtColor(croppedR, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(croppedL, cv2.COLOR_BGR2GRAY)

        hough(grayR, 1, window_num)
        hough(grayL, 3, window_num)

        plt.show()

        window_num += 1
