#!/usr/bin/env python3
"""
main.py - Entry point for Azure App Service
TB Cough Detection API
"""

import os
import sys
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Configure logging for Azure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'UploadedAudio')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Try to import the detector
detector = None
TBCoughDetectorPrediction = None

try:
    from TBCoughDetectorPrediction import TBCoughDetectorPrediction
    logger.info("✅ Successfully imported TBCoughDetectorPrediction")
except ImportError as e:
    logger.warning(f"⚠️ Could not import TBCoughDetectorPrediction: {e}")
    TBCoughDetectorPrediction = None

def get_detector():
    global detector
    if detector is None and TBCoughDetectorPrediction is not None:
        try:
            detector = TBCoughDetectorPrediction()
            if detector.load_model('tb_cough_detector'):
                logger.info("✅ Model loaded successfully")
            else:
                logger.warning("⚠️ Model files not found")
                return None
        except Exception as e:
            logger.error(f"❌ Error creating detector: {e}")
            return None
    return detector

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    
    # Get current directory and files for debugging
    cwd = os.getcwd()
    files = []
    try:
        files = [f for f in os.listdir(cwd) if os.path.isfile(f)][:10]
    except Exception as e:
        files = [f"Error reading directory: {e}"]
    
    return jsonify({
        "message": "TB Cough Detection API",
        "version": "1.0.4",
        "status": "✅ Running on Azure App Service",
        "python_version": sys.version,
        "current_directory": cwd,
        "files_in_directory": files,
        "detector_available": TBCoughDetectorPrediction is not None,
        "endpoints": {
            "health": "/api/health",
            "debug": "/api/debug",
            "upload": "/api/audio/upload"
        }
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    
    try:
        detector_instance = get_detector()
        model_status = "loaded" if detector_instance and detector_instance.model is not None else "not_loaded"
        
        health_info = {
            'status': 'healthy',
            'model_status': model_status,
            'upload_folder': UPLOAD_FOLDER,
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'environment': 'Azure App Service',
            'detector_available': TBCoughDetectorPrediction is not None
        }
        
        return jsonify(health_info), 200
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check environment"""
    try:
        info = {
            'current_directory': os.getcwd(),
            'python_path': sys.path[:5],  # First 5 entries
            'environment_vars': {k: v for k, v in os.environ.items() if 'WEBSITE' in k},
            'directory_contents': {},
            'model_files_check': {}
        }
        
        # Check directory contents
        try:
            for item in os.listdir('.'):
                if os.path.isfile(item):
                    info['directory_contents'][item] = f'file ({os.path.getsize(item)} bytes)'
                else:
                    info['directory_contents'][item] = 'directory'
        except Exception as e:
            info['directory_error'] = str(e)
        
        # Check for model files
        model_files = [
            'tb_cough_detector_model.h5',
            'tb_cough_detector_scaler.pkl',
            'tb_cough_detector_encoder.pkl'
        ]
        
        for model_file in model_files:
            info['model_files_check'][model_file] = {
                'exists': os.path.exists(model_file),
                'size': os.path.getsize(model_file) if os.path.exists(model_file) else 0
            }
        
        return jsonify(info), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/audio/upload', methods=['POST'])
def upload_audio():
    """Upload and analyze audio file"""
    logger.info("Audio upload requested")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'File type not supported. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        detector_instance = get_detector()
        if detector_instance is None or detector_instance.model is None:
            return jsonify({
                'error': 'Model not available. Check /api/debug for details.'
            }), 500

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(file_path)
        logger.info(f"File saved: {filename}")
        
        try:
            result = detector_instance.predict(file_path)
            logger.info(f"Prediction successful: {result['predicted_class']}")
            
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'prediction': result['predicted_class'],
                'confidence': round(float(result['confidence']), 4),
                'probabilities': {k: round(float(v), 4) for k, v in result['probabilities'].items()},
                'message': f'Prediction: {result["predicted_class"]} with {result["confidence"]:.2%} confidence'
            }), 200

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"File processing failed: {e}")
        return jsonify({'error': f'File processing failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

# This is crucial for Azure App Service
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting TB Cough Detection API on port {port}")
    logger.info(f"📁 Current directory: {os.getcwd()}")
    logger.info(f"📋 Files: {os.listdir('.')[:10]}")
    
    app.run(host="0.0.0.0", port=port, debug=False)
