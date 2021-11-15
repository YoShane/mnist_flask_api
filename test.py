import copy
import cv2
import numpy as np
from imutils import contours
from keras.models import load_model
import matplotlib.pyplot as plt


def process_image(img,min_side):
    size = img.shape
    h, w = size[0], size[1]
    #縮放到min_side 
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # 填充min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    else:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=[255,255,255]) 
    #print pad_img.shape
    #cv2.imwrite("after-" + os.path.basename(filename), pad_img)
    return pad_img


# Load Model
model = load_model('CNN_Mnist.h5')    

# method 1
image = cv2.imread('aaaa.png',-1)

#transform background process
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

resultNum = ""
for i in range(ROI_number):
    if(i < 2):

        # method 2
        image = cv2.imread("ROI_"+str(i)+".png",-1)
        img1 = process_image(image,20)
        pad_img = cv2.copyMakeBorder(img1,4,4,4,4, cv2.BORDER_CONSTANT, value=[255,255,255]) 
        #print(pad_img.shape)

        cover=copy.deepcopy(pad_img)
        rows=cover.shape[0]
        cols=cover.shape[1]

        for i in range(rows):
            for j in range(cols):
                cover[i][j]=255-cover[i][j]

        # view result
        cv2.imshow("result", cover)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        image = np.array(cover).reshape(1,28,28)/255
        #aa = image.astype("float32")
        # Make sure images have shape (28, 28, 1)

        ans = np.expand_dims(image, -1)
        print("shape:", ans.shape)

        predict = model.predict(ans)
        y_classes = predict.argmax(axis=-1)
        print('Prediction:', predict)
        print(y_classes)
        resultNum = resultNum + str(y_classes)[1]

print(resultNum)