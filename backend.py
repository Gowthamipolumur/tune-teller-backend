import os
from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment
from flask_cors import CORS
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load the trained CNN (Conv1D) model
model = tf.keras.models.load_model("cnn_model.h5")

# Class labels - update if your model uses different labels
classes = [
    "Baby Baby",
    "baby shark",
    "Cheri cheri lady",
    "Jingle bell rock",
    "mary had a little lamb",
    "Merry christmas",
    "old mcd had a farm",
    "Senorita",
    "twinkle twinkle",
    "we will rock you",
    "wheels on the bus"
]

# Feature extraction for Conv1D model
def extract_features_for_cnn(audio_bytes):
    try:
        sound = AudioSegment.from_file(BytesIO(audio_bytes))
        mp3_io = BytesIO()
        sound.export(mp3_io, format="mp3")
        mp3_io.seek(0)

        y, sr = librosa.load(mp3_io, sr=22050, duration=5.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60)
        mfcc_scaled = np.mean(mfcc.T, axis=0)

        desired_length = 200
        if mfcc_scaled.shape[0] < desired_length:
            mfcc_scaled = np.pad(mfcc_scaled, (0, desired_length - mfcc_scaled.shape[0]), mode='constant')
        else:
            mfcc_scaled = mfcc_scaled[:desired_length]

        mfcc_scaled = np.expand_dims(mfcc_scaled, axis=-1)  # (200, 1)
        return mfcc_scaled

    except Exception as e:
        print(f"❌ Feature Extraction Error: {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict_song():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        audio_bytes = file.read()
        features = extract_features_for_cnn(audio_bytes)
        features = np.expand_dims(features, axis=0)  # (1, 200, 1)

        predictions = model.predict(features)
        predicted_index = np.argmax(predictions)
        predicted_label = classes[predicted_index]
        confidence = float(np.max(predictions))

        return jsonify({
            'prediction': predicted_label,
            'confidence': f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'CNN Model Backend is running ✅'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
