from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

saved_model_path = 'output/final_model.keras'
model = tf.keras.models.load_model(saved_model_path)

@app.route('/upload', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Convert the file storage to PIL Image and ensure it's in RGB
        img = Image.open(io.BytesIO(file.read())).convert('RGB')  # Added .convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        predictions = model.predict(img_array)
        result = "Real Image" if predictions[0, 0] > 0.45 else "Fake Image"
        print(predictions)
        
        return jsonify({'prediction': result})
        # return jsonify({'rate': predictions})

if __name__ == '__main__':
    app.run(debug=True)
