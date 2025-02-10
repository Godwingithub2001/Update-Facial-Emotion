import os

file_path = "C:/DATASCIENCE/MyProjects/FER.CK_emotions/Project/emotion_recognition_model.h5"
if os.path.exists(file_path):
    print("File exists")
else:
    print("File NOT found!")

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the trained emotion recognition model
model = load_model('emotion_recognition_model.h5')

# Load Xception for feature extraction
base_model = tf.keras.applications.Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
feature_extractor_model = tf.keras.Model(inputs=base_model.input, outputs=x)

# Emotion labels
emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("video_frame")
def handle_video_frame(data):
    try:
        # Decode base64 image
        frame_data = base64.b64decode(data["frame"])
        npimg = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            emit("emotion", {"emotion": "No Face Detected"})
            return

        # Process the first detected face
        x, y, w, h = faces[0]
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48)).astype("float32") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Resize for Xception model and extract features
        roi_rgb = tf.image.resize(roi_gray, (299, 299))
        roi_rgb = np.repeat(roi_rgb, 3, axis=-1)
        features = feature_extractor_model.predict(roi_rgb)

        # Predict emotion
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)
        emotion_label = emotion_labels.get(predicted_class, "Unknown")

        # Send emotion prediction to frontend
        emit("emotion", {"emotion": emotion_label})

    except Exception as e:
        emit("error", {"error": str(e)})


@app.route("/predict", methods=["POST"])
def predict_emotion():
    try:
        file = request.files["file"]
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Convert to grayscale and detect face
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({"emotion": "No Face Detected"})

        # Process the first detected face
        x, y, w, h = faces[0]
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48)).astype("float32") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Resize for Xception model and extract features
        roi_rgb = tf.image.resize(roi_gray, (299, 299))
        roi_rgb = np.repeat(roi_rgb, 3, axis=-1)
        features = feature_extractor_model.predict(roi_rgb)

        # Predict emotion
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)
        emotion_label = emotion_labels.get(predicted_class, "Unknown")

        return jsonify({"emotion": emotion_label})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    socketio.run(app, debug=True)


    
