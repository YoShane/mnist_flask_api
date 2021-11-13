import os
from flask import Flask, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import copy
import cv2
import numpy as np
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
            image = cv2.imread(r'imgs/'+filename,-1)
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
            # Make sure images have shape (28, 28, 1)

            ans = np.expand_dims(image, -1)
            print("shape:", ans.shape)
            predict = model.predict(ans)
            y_classes = predict.argmax(axis=-1)
            print('Prediction:', predict)
            print(y_classes)

            return jsonify({'msg': 'success','result':str(y_classes)[1]})


        else:
            return jsonify({'msg': 'Image Couldn\'t be retreived'})
    return jsonify({'msg': 'Only suppoet the POST msg'})

if __name__=="__main__":
    app.run(host="127.0.0.1", port=4870)