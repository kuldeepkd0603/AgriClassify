from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


model = load_model('model/fruit_veg_classifier.h5')  


def preprocess_image(image_path):
    
    image = load_img(image_path, target_size=(224, 224))  
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            
            processed_image = preprocess_image(file_path)
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)

            
            class_names = [ "apple", "banana", "beetroot", "bell pepper", "cabbage", "capsicum", 
    "carrot", "cauliflower", "chilli pepper", "corn", "cucumber", 
    "eggplant", "garlic", "ginger", "grapes", "jalepeno", "kiwi", 
    "lemon", "lettuce", "mango", "onion", "orange", "paprika", "pear", 
    "peas", "pineapple", "pomegranate", "potato", "raddish", "soy beans", 
    "spinach", "sweetcorn", "sweetpotato", "tomato", "turnip", "watermelon"]  
            predicted_label = class_names[predicted_class[0]]

            return render_template('result.html', label=predicted_label, image=file.filename)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
