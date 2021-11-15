import cv2
import numpy as np
from imutils import contours

image = cv2.imread('aaaa.png',-1)

#transform background
sp=image.shape
width=sp[0]
height=sp[1]
for yh in range(height):
    for xw in range(width):
        color_d=image[xw,yh]
        if(color_d[3]==0):
            image[xw,yh]=[255,255,255,255]

#split num
mask = np.zeros(image.shape, dtype=np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
(cnts, _) = contours.sort_contours(cnts, method="left-to-right")
ROI_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    x,y,w,h = cv2.boundingRect(c)
    ROI = 255 - thresh[y:y+h, x:x+w]
    cv2.drawContours(mask, [c], -1, (255,255,255), -1)
    cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
    ROI_number += 1

for i in range(ROI_number):
    if(i < 2):
        img1 = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)