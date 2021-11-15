import os
from flask import Flask, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import copy
import cv2
import numpy as np
from imutils import contours
from keras.models import load_model
import matplotlib.pyplot as plt
import requests # to get image from the web
import shutil # to save it locally
from flask_cors import cross_origin


app = Flask(__name__)

@app.route('/upload_file', methods=["POST"])
@cross_origin()
def upload_file():
    if request.method == "POST":
	
        json_data = request.json
        print(json_data)
        image_url = json_data["url"]
        filename = json_data["url"].split("/")[-1]
        print(image_url)
        print(filename)

        # Open the url image, set stream to True, this will return the stream content.
        r = requests.get(image_url, stream = True)

        # Check if the image was retrieved successfully
        if r.status_code == 200:

            resultNum = ""

            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True
            
            # Open a local file with wb ( write binary ) permission.
            with open(filename,'wb') as f:
                shutil.copyfileobj(r.raw, f)
                
            shutil.move(filename,r'imgs/'+filename)
            print('Image sucessfully Downloaded: ',filename)


            # Load Model
            model = load_model('CNN_Mnist.h5')    

            # method 1
            #transform background process
            image = cv2.imread(r'imgs/'+filename,-1)
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
                

            num = int(resultNum)
            if(num>23):
                num = int(resultNum[0]) #只取第一個
 
            print(num)

            return jsonify({'msg': 'success','result':str(num)})


        else:
            return jsonify({'msg': 'Image Couldn\'t be retreived'})
    return jsonify({'msg': 'Only suppoet the POST msg'})

#等比縮放圖片用
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

if __name__=="__main__":
    app.run(host="163.18.42.231", port=4870)