from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from deepface import DeepFace
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import sqlite3

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize DB
def init_db():
    conn = sqlite3.connect('user_logs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (id INTEGER PRIMARY KEY, timestamp TEXT, user_name TEXT, image_path TEXT, emotion TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    user_name = request.form.get('user_name', 'Anonymous')
    image_data = request.form.get('image')

    # Decode base64 image
    import base64
    img_data = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Save image
    filename = secure_filename(f"{user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(filepath, img)

    # Use DeepFace (built-in model!)
    try:
        result = DeepFace.analyze(
            img,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        emotion = result[0]['dominant_emotion']
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Log to DB
    conn = sqlite3.connect('user_logs.db')
    c = conn.cursor()
    c.execute("INSERT INTO logs (timestamp, user_name, image_path, emotion) VALUES (?, ?, ?, ?)",
              (datetime.now().isoformat(), user_name, filepath, emotion))
    conn.commit()
    conn.close()

    return jsonify({'emotion': emotion.capitalize()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)