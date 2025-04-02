import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from ultralytics import YOLO
import numpy as np
import cv2
from pdf2image import convert_from_bytes

app = Flask(__name__)

# Allow requests from your deployed domain (adjust if needed)
CORS(app, resources={r"/*": {"origins": "https://ai4tb-be-gnangzcmfhbcbrhf.francecentral-01.azurewebsites.net"}})

# Load your YOLO model
model_weights = 'last.pt'
model = YOLO(model_weights)

@app.route('/')
def home():
    return "Server is running"

@app.route('/detect', methods=['POST'])
def detect():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filename = file.filename.lower()

    # Read confidence from form data (default 0.5)
    confidence = request.form.get('confidence', 0.5)
    try:
        confidence = float(confidence)
    except ValueError:
        return jsonify({'error': 'Invalid confidence value'}), 400

    # Handle PDF files: convert first page to image
    if filename.endswith('.pdf'):
        pdf_bytes = file.read()
        try:
            images = convert_from_bytes(pdf_bytes)
            pil_img = images[0]  # just the first page
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return jsonify({'error': f'Error processing PDF: {str(e)}'}), 400
    else:
        # Assume an image (PNG/JPG)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

    # Run YOLO detection
    results = model.predict(source=img, conf=confidence)

    # Count bacilli (assume class 0) and draw bounding boxes
    bacilli_count = 0
    for result in results:
        for detection in result.boxes.data:
            if int(detection[5]) == 0:  # class index
                bacilli_count += 1
                x1, y1, x2, y2 = map(int, detection[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Encode processed image to base64
    _, buffer = cv2.imencode('.jpg', img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'bacilli_count': bacilli_count,
        'image': encoded_image
    })


if __name__ == '__main__':
    # On Azure, 'PORT' is provided via environment variable
    port = int(os.environ.get('PORT', 8000))
    # Run built-in Flask server on 0.0.0.0 so Azure can route requests
    app.run(host='0.0.0.0', port=port, debug=False)
