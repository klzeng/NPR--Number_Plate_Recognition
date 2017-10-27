# coding: utf-8
import numpy
import cv2

img = cv2.imread('NP_image2.jpg',0)
(thresh, img_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("win1", img_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()


