from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import subprocess
import re
import json
import csv
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
IMAGE_FOLDER = 'images'
RESULTS_FILE = 'emotion_results.json'
EMOTION_SCRIPT = "python emonet/demo.py --nclass 8 --image_path"

def load_emotion_mapping_from_csv(filepath='emotion_mapping.csv'):
    mapping = []
    try:
        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                mapping.append({
                    "emotion": row["emotion"],
                    "valence_range": (float(row["valence_min"]), float(row["valence_max"])),
                    "arousal_range": (float(row["arousal_min"]), float(row["arousal_max"]))
                })
    except Exception as e:
        print(f"Error loading emotion mapping: {e}")
    return mapping

# Emotion mapping for custom mode
EMOTION_MAPPING = load_emotion_mapping_from_csv()


def map_to_custom_emotion(valence, arousal):
    for mapping in EMOTION_MAPPING:
        v_min, v_max = mapping["valence_range"]
        a_min, a_max = mapping["arousal_range"]
        if v_min <= valence <= v_max and a_min <= arousal <= a_max:
            return mapping["emotion"]
    return "unknown"



def save_results(data):
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', RESULTS_FILE)

    existing_data = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:  # ✅ Make sure it's not empty
                existing_data = json.loads(content)
            else:
                print("Warning: results file is empty. Starting fresh.")

    existing_data.append(data)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(existing_data[-100:], f, indent=2)


@app.route('/save-image', methods=['POST'])
def save_image():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400

    file = request.files['image']
    use_custom = request.form.get('use_custom', 'false').lower() == 'true'

    filename = 'current_frame.jpg'
    filepath = os.path.join(IMAGE_FOLDER, filename)
    file.save(filepath)

    try:
        result = subprocess.run(
            [sys.executable, "emonet/demo.py", "--nclass", "8", "--image_path", os.path.abspath(filepath)],
            capture_output=True,
            text=True,
            check=True
        )

        # Optional: Print output for debugging
        print("EMOnet output:", result.stdout)

        match = re.search(
            r"Predicted Emotion (\w+) - valence (-?\d+\.\d+) - arousal (\d+\.\d+)",
            result.stdout
        )
        if not match:
            raise ValueError("Could not parse emotion output")

        valence = float(match.group(2))
        arousal = float(match.group(3))

        if use_custom:
            emotion = map_to_custom_emotion(valence, arousal)
            mode = 'custom'
        else:
            emotion = match.group(1)
            mode = 'original'

        result_data = {
            'timestamp': datetime.now().isoformat(),
            'emotion': emotion,
            'valence': valence,
            'arousal': arousal,
            'mode': mode
        }


        save_results(result_data)
        return jsonify({'status': 'success', **result_data})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)