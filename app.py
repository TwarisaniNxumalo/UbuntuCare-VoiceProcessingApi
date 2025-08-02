import os
import sys
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np

# Configure logging for Azure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'UploadedAudio')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Try to import the detector with better error handling
detector = None
TBCoughDetectorPrediction = None

try:
    from TBCoughDetectorPrediction import TBCoughDetectorPrediction
    logger.info("✅ Successfully imported TBCoughDetectorPrediction")
except ImportError as e:
    logger.error(f"❌ Failed to import TBCoughDetectorPrediction: {e}")
    TBCoughDetectorPrediction = None

def get_detector():
    global detector
    if detector is None and TBCoughDetectorPrediction is not None:
        try:
            detector = TBCoughDetectorPrediction()
            logger.info("Created detector instance")
            
            # Try to load the model
            if detector.load_model('tb_cough_detector'):
                logger.info("✅ Model loaded successfully")
            else:
                logger.warning("❌ Failed to load model - model files may not be present")
                return None
        except Exception as e:
            logger.error(f"Error creating detector: {e}")
            return None
    return detector

@app.route('/api/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    
    try:
        detector_instance = get_detector()
        model_status = "loaded" if detector_instance and detector_instance.model is not None else "not_loaded"
        
        # Check current working directory and files
        cwd = os.getcwd()
        files_in_cwd = os.listdir(cwd)
        
        health_info = {
            'status': 'healthy',
            'model_status': model_status,
            'upload_folder': UPLOAD_FOLDER,
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'environment': 'Azure App Service',
            'current_directory': cwd,
            'python_version': sys.version,
            'files_in_directory': files_in_cwd[:10],  # Show first 10 files
            'detector_available': TBCoughDetectorPrediction is not None
        }
        
        logger.info(f"Health check response: {health_info}")
        return jsonify(health_info), 200
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'environment': 'Azure App Service'
        }), 500

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check environment and files"""
    try:
        debug_info = {
            'python_path': sys.path,
            'environment_variables': dict(os.environ),
            'current_directory': os.getcwd(),
            'directory_contents': {},
            'installed_packages': {}
        }
        
        # Check directory contents
        try:
            for item in os.listdir('.'):
                if os.path.isdir(item):
                    debug_info['directory_contents'][item] = 'directory'
                else:
                    debug_info['directory_contents'][item] = f'file ({os.path.getsize(item)} bytes)'
        except Exception as e:
            debug_info['directory_error'] = str(e)
        
        # Check for model files
        model_files = [
            'tb_cough_detector_model.h5',
            'tb_cough_detector_scaler.pkl', 
            'tb_cough_detector_encoder.pkl'
        ]
        
        debug_info['model_files'] = {}
        for model_file in model_files:
            debug_info['model_files'][model_file] = os.path.exists(model_file)
        
        return jsonify(debug_info), 200
        
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/audio/upload', methods=['POST'])
def upload_audio():
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
            return jsonify({'error': 'Model not available. Check debug endpoint for details.'}), 500

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(file_path)
        logger.info(f"File saved to: {file_path}")
        
        try:
            result = detector_instance.predict(file_path)
            logger.info(f"Prediction successful: {result['predicted_class']}")
            
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

@app.route("/", methods=["GET"])
def root():
    logger.info("Root endpoint accessed")
    return jsonify({
        "message": "TB Cough Detection API",
        "version": "1.0.3",
        "status": "Running on Azure App Service with Enhanced Logging",
        "endpoints": {
            "health": "/api/health",
            "debug": "/api/debug",
            "upload_predict": "/api/audio/upload"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS)
    }), 200

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

# Azure App Service entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting TB Cough Detection API on port {port}")
    logger.info(f"📁 Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"🎵 Supported formats: {ALLOWED_EXTENSIONS}")
    logger.info(f"🔍 Current directory: {os.getcwd()}")
    logger.info(f"📋 Files in directory: {os.listdir('.')}")
    
    app.run(host="0.0.0.0", port=port, debug=False)
