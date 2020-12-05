# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from gevent.pywsgi import WSGIServer

# deep learning utilities
from util import base64_to_pil
import keras
import numpy as np
import argparse
import cv2
from skimage.feature import hog
import pickle
import imutils
#from enhance import *
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
# Declare a flask app
app = Flask(__name__)

#Declaring the classes


# Model saved with Keras model.save()
MODEL_PATH = ''

print('Model loaded. Start serving at http://127.0.0.1:5000/')


#routes

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("./uploads/image.jpg")

        ########
       #print("uploading of image works")
        #######
        #pre-processing
        img = cv2.imread("./uploads/image.jpg")
       # print("Reading of image works")
        img = imutils.resize(img, height=300)
        height = len(img)
        width = len(img[0])

       # print("checkpoint1")
        xp = [0, 64, 128, 192, 255]
        fp = [0, 16, 128, 240, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        img2 = cv2.LUT(img, table)
        
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # gray = cv2.blur(gray, (3, 3))
        hog_features = []
        y_train_labels = []
        clahe = cv2.createCLAHE(clipLimit=40)
        #print("checkpoint 2")
        clf8 = joblib.load('filename3.pkl')
       # print("checkpoint 3")
        sum1=0
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50)
        if circles is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.blur(gray, (3, 3))
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 70)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                left = max(0, x-r)
                top = max(0, y-r)
                right = min(x+r, width)
                bottom = min(y+r, height)
                img_res = gray[top: bottom, left:right]
                img_res = cv2.equalizeHist(img_res)
                img_res = clahe.apply(img_res)
                im = cv2.resize(img_res, (100, 100))
                a = []
                fd, hog_imge = hog(im, orientations=9, pixels_per_cell=(8, 8),
                                   cells_per_block=(1, 1), visualize=True)
                pc = [fd]
                y_pred = clf8.predict_proba(pc)
                #print(max(y_pred[0]))
                y_pred = clf8.predict(pc)
                #res=y_pred[0]
                #print(type(res))
                print(y_pred[0])
                sum1+=int(y_pred[0].split("_")[0])
                #cv2.imshow("", im)
                #cv2.waitKey(0)
            #cv2.imshow("output", np.hstack([img, output]))
            #cv2.waitKey(0)
        else:
            print("No coins recognized")
            #cv2.imshow("Output", img)
            #cv2.waitKey(0)
    
        
        # Make prediction
        #classe = model.predict_classes(images)
        #print(str(classe[0]))
        res="Value is "+str(sum1)
        # Serialize the result, you can add additional fields
        return jsonify(result=res)
     

        

    return render_template('predict.html')


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
