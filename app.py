from flask import Flask, request, jsonify
import os
import tempfile
from werkzeug.utils import secure_filename
import numpy as np

# Import your TB detection class
try:
    from TBCoughDetectorPrediction import TBCoughDetectorPrediction
except ImportError:
    print("Warning: TBCoughDetectorPrediction not found")
    TBCoughDetectorPrediction = None

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'UploadedAudio')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize the detector (will load model when needed)
detector = None

def get_detector():
    global detector
    if detector is None and TBCoughDetectorPrediction is not None:
        detector = TBCoughDetectorPrediction()
        
        # Try to load the model
        if detector.load_model('tb_cough_detector'):
            print("✅ Model loaded successfully")
        else:
            print("❌ Failed to load model - model files may not be present")
            return None
    return detector

@app.route('/api/health', methods=['GET'])
def health_check():
    detector_instance = get_detector()
    model_status = "loaded" if detector_instance and detector_instance.model is not None else "not_loaded"
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'upload_folder': UPLOAD_FOLDER,
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'environment': 'Azure App Service'
    }), 200

@app.route('/api/audio/upload', methods=['POST'])
def upload_audio():
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']

    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'File type not supported. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        # Get detector instance
        detector_instance = get_detector()
        if detector_instance is None or detector_instance.model is None:
            return jsonify({'error': 'Model not available. Please check if model files exist.'}), 500

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        file.save(file_path)
        print(f"📁 File saved to: {file_path}")
        
        try:
            result = detector_instance.predict(file_path)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'file_path': file_path,
                'prediction': result['predicted_class'],
                'confidence': round(float(result['confidence']), 4),
                'probabilities': {k: round(float(v), 4) for k, v in result['probabilities'].items()},
                'message': f'Prediction: {result["predicted_class"]} with {result["confidence"]:.2%} confidence'
            }), 200

        except Exception as e:
            # Clean up file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': f'File processing failed: {str(e)}'}), 500

@app.route('/api/predict/file', methods=['POST'])
def predict_from_path():
    """Predict from a file path (for files already on server)"""
    data = request.get_json()
    
    if not data or 'file_path' not in data:
        return jsonify({'error': 'file_path is required in JSON body'}), 400
    
    file_path = data['file_path']
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        detector_instance = get_detector()
        if detector_instance is None or detector_instance.model is None:
            return jsonify({'error': 'Model not available'}), 500
        
        result = detector_instance.predict(file_path)
        print(f"🔍 Prediction result: {result}")

        return jsonify({
            'success': True,
            'file_path': file_path,
            'prediction': result['predicted_class'],
            'confidence': round(float(result['confidence']), 4),
            'probabilities': {k: round(float(v), 4) for k, v in result['probabilities'].items()}
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "TB Cough Detection API",
        "version": "1.0.2",
        "status": "Running on Azure App Service",
        "endpoints": {
            "health": "/api/health",
            "upload_predict": "/api/audio/upload",
            "predict_file": "/api/predict/file",
        },
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size": "16MB"
    }), 200

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# This is the key part for Azure App Service
if __name__ == "__main__":
    # Development mode
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
