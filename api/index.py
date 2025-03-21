import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning logs

import tensorflow as tf

# No GPU-specific memory settings (since Render uses CPU-only)
print("✅ Running TensorFlow on CPU with optimized settings")

from flask import Flask, request, jsonify
import numpy as np
import librosa
import joblib
from pydub import AudioSegment
from flask_cors import CORS
import imageio_ffmpeg

from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Lazy Load Model to Conserve Memory
cnn_model = None  # Initialize model as None

def get_model():
    global cnn_model
    if cnn_model is None:
        cnn_model = load_model("cnn_model.h5")
        print("✅ Model Loaded Successfully")
    return cnn_model

# Load scaler
scaler = joblib.load("scaler.pkl")

# Class labels
classes = ["baby shark", "mary had a little lamb", "Merry Christmas",
           "Old McDonald Had a Farm", "Twinkle Twinkle", "Wheels on the Bus"]

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'aac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert any audio file to MP3
def convert_to_mp3(file_path):
    try:
        extension = file_path.split('.')[-1].lower()
        if extension != 'mp3':
            sound = AudioSegment.from_file(file_path)
            mp3_path = file_path.rsplit('.', 1)[0] + ".mp3"
            sound.export(mp3_path, format="mp3")
            return mp3_path
        return file_path
    except Exception as e:
        raise RuntimeError(f"Error converting to MP3: {e}")

# Feature Extraction
def extract_features(file_path):
    try:
        print(f"📂 Processing file: {file_path}")
        file_path = convert_to_mp3(file_path)

        audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
        print(f"✅ Audio loaded: Sample Rate - {sample_rate}, Duration - {len(audio) / sample_rate}s")

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)

        features = np.hstack((
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0)
        ))

        return features
    except Exception as e:
        print(f"🔥 Error extracting features: {e}")
        raise RuntimeError(f"Error extracting features: {e}")

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict_song():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or unsupported file type. Please upload MP3, WAV, M4A, or AAC files only.'}), 400

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    file.save(temp_file_path)

    try:
        # Extract Features
        features = extract_features(temp_file_path)
        features_scaled = scaler.transform([features])

        features_scaled_cnn = features_scaled.reshape(1, features_scaled.shape[1], 1)
        cnn_model = get_model()
        cnn_prediction = np.argmax(cnn_model.predict(features_scaled_cnn), axis=1)[0]

        prediction = classes[cnn_prediction]

        return jsonify({"song_name": prediction})

    except Exception as e:
        print(f"❌ Backend Error: {str(e)}")
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

    finally:
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"🧹 Successfully deleted temporary file: {temp_file_path}")
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup error: {cleanup_error}")

# Root Route to Confirm Deployment Status
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Tune Teller Backend is Running Successfully!'}), 200

# Run the Server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render expects port 10000
    app.run(host='0.0.0.0', port=port)
