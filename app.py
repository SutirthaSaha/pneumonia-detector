import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Keras
from tensorflow.keras.models import load_model
from skimage import transform
from PIL import Image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/91.5_model.h5'

# # Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def model_predict(img_path, model):
    img_dims = 256
    np_image = Image.open(img_path)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (img_dims, img_dims, 3))
    np_image = np.expand_dims(np_image, axis=0)
    preds = model.predict(np_image)
    return preds

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        pred_class = preds[0][0]
        pred_class = int(np.round(pred_class))
        if pred_class==1:
            result ='This is a Pneumonia case'
        else:
            result='This is a Normal Case'
        os.remove(file_path)
        return result
    return None


if __name__ == '__main__':
    app.run(port=5002, debug=True)
