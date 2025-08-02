from flask import Flask, request, jsonify
import os
import tempfile
from werkzeug.utils import secure_filename
import numpy as np

# Option 1: Use the lightweight prediction-only class (RECOMMENDED for API)
from TBCoughDetectorPrediction import TBCoughDetectorPrediction

# Option 2: Use the full training class (if you need training capabilities)
# from TBCoughDetector import TBCoughDetector

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
    if detector is None:
        # Using the lightweight prediction class
        detector = TBCoughDetectorPrediction()
        
        # Load the model
        if detector.load_model('tb_cough_detector'):
            print("✅ Model loaded successfully")
        else:
            print("❌ Failed to load model")
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
        'supported_formats': list(ALLOWED_EXTENSIONS)
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

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    detector_instance = get_detector()
    
    if detector_instance is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    info = {
        'model_loaded': detector_instance.model is not None,
        'sample_rate': detector_instance.sample_rate,
        'duration': detector_instance.duration,
        'n_mels': detector_instance.n_mels,
        'n_mfcc': detector_instance.n_mfcc,
    }
    
    if detector_instance.model is not None:
        info['model_summary'] = {
            'input_shape': detector_instance.model.input_shape,
            'output_shape': detector_instance.model.output_shape,
            'total_params': detector_instance.model.count_params()
        }
        
        if hasattr(detector_instance, 'label_encoder') and detector_instance.label_encoder is not None:
            info['classes'] = detector_instance.label_encoder.classes_.tolist()
    
    return jsonify(info), 200

@app.route('/api/batch/predict', methods=['POST'])
def batch_predict():
    """Predict multiple files at once"""
    data = request.get_json()
    
    if not data or 'file_paths' not in data:
        return jsonify({'error': 'file_paths array is required in JSON body'}), 400
    
    file_paths = data['file_paths']
    
    if not isinstance(file_paths, list):
        return jsonify({'error': 'file_paths must be an array'}), 400
    
    detector_instance = get_detector()
    if detector_instance is None or detector_instance.model is None:
        return jsonify({'error': 'Model not available'}), 500
    
    results = []
    errors = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            errors.append(f"File not found: {file_path}")
            continue
        
        try:
            result = detector_instance.predict(file_path)
            results.append({
                'file_path': file_path,
                'prediction': result['predicted_class'],
                'confidence': round(float(result['confidence']), 4),
                'probabilities': {k: round(float(v), 4) for k, v in result['probabilities'].items()}
            })
        except Exception as e:
            errors.append(f"Error predicting {file_path}: {str(e)}")
    
    return jsonify({
        'success': True,
        'results': results,
        'errors': errors,
        'total_files': len(file_paths),
        'successful_predictions': len(results),
        'failed_predictions': len(errors)
    }), 200

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "TB Cough Detection API",
        "version": "1.0.1",
        "status": "Fixed hardcoded path bug",
        "endpoints": {
            "health": "/api/health",
            "upload_predict": "/api/audio/upload",
            "predict_file": "/api/predict/file",
            "batch_predict": "/api/batch/predict",
            "model_info": "/api/model/info"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size": "16MB",
        "features": [
            "Audio file upload and prediction",
            "File path prediction",
            "Batch file prediction",
            "Model health checking",
            "Real-time predictions"
        ]
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

if __name__ == "__main__":
    print("🚀 Starting TB Cough Detection API...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🎵 Supported formats: {ALLOWED_EXTENSIONS}")
    
    # Test model loading on startup
    test_detector = get_detector()
    if test_detector and test_detector.model is not None:
        print("✅ Model loaded successfully - API ready!")
    else:
        print("❌ Warning: Model failed to load. Check if model files exist.")
        print("   Make sure you've run the training script first.")
    
    app.run(debug=True, host="0.0.0.0", port=5000)
