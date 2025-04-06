from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import subprocess
from werkzeug.utils import secure_filename
import re  # Added for more robust output parsing
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
IMAGE_FOLDER = 'images'
RESULTS_FILE = 'emotion_results.json'
EMOTION_SCRIPT = "python emonet/demo.py --nclass 8 --image_path"

def save_results(emotion, valence, arousal):
    """Save results to JSON file with timestamp"""
    result_data = {
        'timestamp': datetime.now().isoformat(),
        'emotion': emotion,
        'valence': valence,
        'arousal': arousal
    }
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    filepath = os.path.join('results', RESULTS_FILE)
    
    try:
        # Read existing data if file exists
        existing_data = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
        
        # Add new result and keep only last 100 entries
        existing_data.append(result_data)
        if len(existing_data) > 100:
            existing_data = existing_data[-100:]
        
        # Write back to file
        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
    except Exception as e:
        print(f"Error saving results: {str(e)}", file=sys.stderr)

@app.route('/save-image', methods=['POST'])
def save_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    filename = 'current_frame.jpg'
    filepath = os.path.join(IMAGE_FOLDER, filename)
    file.save(filepath)

    # Use absolute path to avoid any path issues
    absolute_path = os.path.abspath(filepath)
    
    try:
        # Run emotion analysis with full path
        result = subprocess.run(
            [sys.executable, "emonet/demo.py", "--nclass", "8", "--image_path", absolute_path],
            capture_output=True,
            text=True,
            check=True
        )

        print("Script output:", result.stdout)  # Debugging
        
        # Robust parsing using regular expression
        match = re.search(
            r"Predicted Emotion (\w+) - valence (-?\d+\.\d+) - arousal (\d+\.\d+)",
            result.stdout
        )
        
        if not match:
            raise ValueError("Could not parse emotion output")
            
        emotion = match.group(1)
        valence = float(match.group(2))
        arousal = float(match.group(3))

        # Save the results to JSON file
        save_results(emotion, valence, arousal)
        
        return jsonify({
            'status': 'success',
            'emotion': emotion,
            'valence': valence,
            'arousal': arousal,
            'timestamp': datetime.now().isoformat()
        })
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Script failed with code {e.returncode}: {e.stderr}"
        print(error_msg, file=sys.stderr)
        return jsonify({'status': 'error', 'message': error_msg}), 500
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(error_msg, file=sys.stderr)
        return jsonify({'status': 'error', 'message': error_msg}), 500

@app.route('/get-results', methods=['GET'])
def get_results():
    """Endpoint to retrieve saved results"""
    filepath = os.path.join('results', RESULTS_FILE)
    if not os.path.exists(filepath):
        return jsonify({'status': 'error', 'message': 'No results available'}), 404
    
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        return jsonify({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Failed to read results: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Create required directories
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    app.run(debug=True, port=5000, host='0.0.0.0')