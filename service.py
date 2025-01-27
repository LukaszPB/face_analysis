from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io
from PIL import Image
import dlib

# Ładowanie modelu do detekcji punktów twarzy
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

app = Flask(__name__)
CORS(app)

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return {"error": "No image file provided"}, 400

    # Pobranie obrazu z żądania
    file = request.files['image']
    image = Image.open(file.stream)

    # Konwersja obrazu do formatu OpenCV
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale for face detection
    image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Wykrywanie twarzy
    faces = detector(image_gray)

    if len(faces) == 0:
        return {"error": "Face not detected"}, 400
    if len(faces) > 1:
        return {"error": f"Too many faces detected: {len(faces)}"}, 400

    for face in faces:
        landmarks = predictor(image_gray, face)
        # Rysowanie punktów kluczowych
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image_cv, (x, y), 5, (255, 0, 0), -1)

    # Konwersja obrazu z powrotem do formatu PIL
    result_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # Przygotowanie obrazu do zwrócenia jako odpowiedź
    img_io = io.BytesIO()
    result_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()