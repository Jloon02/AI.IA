from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import subprocess
from werkzeug.utils import secure_filename
import re
import json
from datetime import datetime
import csv

app = Flask(__name__)
CORS(app)

# Configuration
IMAGE_FOLDER = 'images'
RESULTS_FILE = 'emotion_results.json'
EMOTION_SCRIPT = "python emonet/demo.py --nclass 8 --image_path"
EMOTION_MAPPING_CSV = "emotion_mapping.csv"

# Default emotion mapping (UTF-8 encoded)
DEFAULT_EMOTION_MAPPING = [
    {"valence_min": -1.0, "valence_max": -0.6, "arousal_min": 0.4, "arousal_max": 1.0, "emotion": "angry", "emoji": "😡"},
    {"valence_min": -1.0, "valence_max": -0.6, "arousal_min": 0.0, "arousal_max": 0.4, "emotion": "sad", "emoji": "😭"},
    {"valence_min": -0.6, "valence_max": -0.2, "arousal_min": 0.6, "arousal_max": 1.0, "emotion": "nervous", "emoji": "😅"},
    {"valence_min": -0.6, "valence_max": -0.2, "arousal_min": 0.0, "arousal_max": 0.6, "emotion": "skeptical", "emoji": "🙄"},
    {"valence_min": -0.2, "valence_max": 0.2, "arousal_min": 0.0, "arousal_max": 1.0, "emotion": "neutral", "emoji": "🙂"},
    {"valence_min": 0.2, "valence_max": 0.6, "arousal_min": 0.0, "arousal_max": 0.6, "emotion": "thoughtful", "emoji": "🤔"},
    {"valence_min": 0.2, "valence_max": 0.6, "arousal_min": 0.6, "arousal_max": 1.0, "emotion": "happy", "emoji": "😁"},
    {"valence_min": 0.6, "valence_max": 1.0, "arousal_min": 0.0, "arousal_max": 0.6, "emotion": "loving", "emoji": "😍"},
    {"valence_min": 0.6, "valence_max": 1.0, "arousal_min": 0.6, "arousal_max": 1.0, "emotion": "excited", "emoji": "🤣"},
    {"valence_min": -0.2, "valence_max": 0.2, "arousal_min": 0.6, "arousal_max": 1.0, "emotion": "surprised", "emoji": "😮"},
    {"valence_min": 0.2, "valence_max": 0.6, "arousal_min": 0.4, "arousal_max": 0.8, "emotion": "confident", "emoji": "😎"}
]

def load_emotion_mapping():
    """Load emotion mapping from CSV with UTF-8 encoding or use defaults"""
    try:
        with open(EMOTION_MAPPING_CSV, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)
    except Exception as e:
        print(f"Using default emotion mapping. Could not read CSV: {str(e)}")
        return DEFAULT_EMOTION_MAPPING

def map_to_custom_emotion(valence, arousal):
    """Map valence/arousal to custom emotion/emoji with proper type conversion"""
    emotion_mapping = load_emotion_mapping()
    
    for mapping in emotion_mapping:
        try:
            # Ensure all values are floats
            v_min = float(mapping.get('valence_min', -1.0))
            v_max = float(mapping.get('valence_max', 1.0))
            a_min = float(mapping.get('arousal_min', 0.0))
            a_max = float(mapping.get('arousal_max', 1.0))
            
            if (v_min <= valence <= v_max) and (a_min <= arousal <= a_max):
                return {
                    'emotion': mapping.get('emotion', 'neutral'),
                    'emoji': mapping.get('emoji', '🙂')
                }
        except (ValueError, KeyError) as e:
            print(f"Error processing mapping entry: {e}")
            continue
    
    return {
        'emotion': 'neutral',
        'emoji': '🙂'
    }

def save_results(emotion_data):
    """Save results to JSON file with error handling"""
    result_data = {
        'timestamp': datetime.now().isoformat(),
        'emotion': emotion_data.get('emotion', 'neutral'),
        'emoji': emotion_data.get('emoji', '🙂'),
        'valence': emotion_data.get('valence', 0.0),
        'arousal': emotion_data.get('arousal', 0.5)
    }
    
    try:
        os.makedirs('results', exist_ok=True)
        filepath = os.path.join('results', RESULTS_FILE)
        
        existing_data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
            except Exception as e:
                print(f"Error reading existing results: {e}")
        
        existing_data.append(result_data)
        existing_data = existing_data[-100:]  # Keep only last 100 entries
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error saving results: {str(e)}", file=sys.stderr)

@app.route('/save-image', methods=['POST'])
def save_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        file = request.files['image']
        filename = 'current_frame.jpg'
        filepath = os.path.join(IMAGE_FOLDER, filename)
        file.save(filepath)
        absolute_path = os.path.abspath(filepath)

        result = subprocess.run(
            [sys.executable, "emonet/demo.py", "--nclass", "8", "--image_path", absolute_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )

        print("Script output:", result.stdout)
        
        match = re.search(
            r"Predicted Emotion (\w+) - valence (-?\d+\.\d+) - arousal (\d+\.\d+)",
            result.stdout
        )
        
        if not match:
            raise ValueError("Could not parse emotion output")
            
        valence = float(match.group(2))
        arousal = float(match.group(3))
        
        custom_emotion = map_to_custom_emotion(valence, arousal)
        emotion_data = {
            'valence': valence,
            'arousal': arousal,
            'emotion': custom_emotion['emotion'],
            'emoji': custom_emotion['emoji']
        }
        
        save_results(emotion_data)
        
        return jsonify({
            'status': 'success',
            **emotion_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Script failed: {e.stderr}"
        print(error_msg, file=sys.stderr)
        return jsonify({'status': 'error', 'message': error_msg}), 500
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(error_msg, file=sys.stderr)
        return jsonify({'status': 'error', 'message': error_msg}), 500

if __name__ == '__main__':
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    app.run(debug=True, port=5000, host='0.0.0.0')