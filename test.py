import copy
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Load Model
model = load_model('CNN_Mnist.h5')    

# method 1
image = cv2.imread("aaaa.png",-1)
img1 = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
sp=img1.shape
width=sp[0]
height=sp[1]
for yh in range(height):
    for xw in range(width):
        color_d=img1[xw,yh]
        if(color_d[3]==0):
            img1[xw,yh]=[255,255,255,255]


img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

cover=copy.deepcopy(img1)
rows=cover.shape[0]
cols=cover.shape[1]

for i in range(rows):
    for j in range(cols):
        cover[i][j]=255-cover[i][j]

image = np.array(cover).reshape(1,28,28)/255
#aa = image.astype("float32")
# Make sure images have shape (28, 28, 1)

ans = np.expand_dims(image, -1)
print("shape:", ans.shape)

predict = model.predict(ans)
y_classes = predict.argmax(axis=-1)
print('Prediction:', predict)
print(y_classes)