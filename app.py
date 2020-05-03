import os
from flask import Flask, render_template, request
from flask import send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
MODEL_FOLDER = 'model'

# Load model
cnn_model = tf.keras.models.load_model(MODEL_FOLDER + '/' + 'catdog_vgg16.h5')

IMAGE_SIZE = 224
# Preprocess an image

# Read the image from path and preprocess
def load_and_preprocess_image(path):
    # load the image
    img = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

# Predict & classify image
def classify(model, image_path):

    preprocessed_imgage = load_and_preprocess_image(image_path)
    # preprocessed_imgage = tf.reshape(preprocessed_imgage, (1,IMAGE_SIZE ,IMAGE_SIZE ,3))

    prob = cnn_model.predict(preprocessed_imgage)
    label = "Cat" if prob <= 0.5 else "Dog"
    classified_prob = prob if prob >= 0.5 else 1 - prob

    
    return label, classified_prob

# home page
@app.route('/')
def home():
   return render_template('home.html')

@app.route('/classify', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('home.html')


    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify(cnn_model, upload_image_path)

        prob = round((prob[0][0] * 100), 5)

    return render_template('classify.html', image_file_name = file.filename, label = label, prob = prob)

@app.route('/classify/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True