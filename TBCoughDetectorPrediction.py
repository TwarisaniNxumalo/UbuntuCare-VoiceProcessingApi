import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

class TBCoughDetectorPrediction:
    def __init__(self, sample_rate=22050, duration=5.0, n_mels=128, n_mfcc=13):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def extract_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            if len(y) < self.n_samples:
                y = np.pad(y, (0, self.n_samples - len(y)))
            else:
                y = y[:self.n_samples]
            features = {}
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_mean'] = np.mean(mel_spec_db, axis=1)
            features['mel_std'] = np.std(mel_spec_db, axis=1)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            features['mfcc_delta'] = np.mean(librosa.feature.delta(mfcc), axis=1)
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            
            
            harmonic, percussive = librosa.effects.hpss(y)
            features['harmonic_ratio'] = np.mean(harmonic**2) / (np.mean(y**2) + 1e-10)
            
           
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            
            
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = tempo
            except:
                features['tempo'] = 120.0
            
           
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            features['onset_rate'] = len(onset_frames) / self.duration
            
           
            rms = librosa.feature.rms(y=y)
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
           
            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500)
                f0_clean = f0[~np.isnan(f0)]
                if len(f0_clean) > 0:
                    features['f0_mean'] = np.mean(f0_clean)
                    features['f0_std'] = np.std(f0_clean)
                    features['voiced_ratio'] = np.mean(voiced_probs)
                else:
                    features['f0_mean'] = 0
                    features['f0_std'] = 0
                    features['voiced_ratio'] = 0
            except:
                features['f0_mean'] = 0
                features['f0_std'] = 0
                features['voiced_ratio'] = 0
            
           
            feature_vector = []
            for key in sorted(features.keys()):
                if isinstance(features[key], np.ndarray):
                    feature_vector.extend(features[key])
                else:
                    feature_vector.append(features[key])
            
            return np.array(feature_vector)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros(self.n_mels * 2 + self.n_mfcc * 3 + 12 + 12 + 9)
    
    def load_model(self, filepath):
        try:
            self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
            self.label_encoder = joblib.load(f"{filepath}_encoder.pkl")
            return True
        except Exception as e:
            return False
    
    def predict(self, audio_path):
        if self.model is None:
            raise ValueError("Model not loaded! Use load_model() first.")
        
        features = self.extract_features(audio_path)
        features = features.reshape(1, -1)
        features = self.scaler.transform(features)    
    
        probabilities = self.model.predict(features, verbose=0)[0]
        predicted_class = self.label_encoder.classes_[np.argmax(probabilities)]
        
        result = {
            'predicted_class': predicted_class,
            'confidence': np.max(probabilities),
            'probabilities': {
                class_name: prob 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            }
        }
        
        return result



def test_all_files():
    detector = TBCoughDetectorPrediction()
    if not detector.load_model('tb_cough_detector'):
        return
    
    audio_files = [
        'Whooping Cough in an Adult  NEJM.wav',
        'Whooping Cough in an Adult  NEJM 1.wav',
        'Whooping Cough in an Adult  NEJM 2.wav',
        'Whooping Cough in an Adult  NEJM 3.wav',
        'Whooping Cough in an Adult  NEJM 4.wav',
        'Whooping Cough in an Adult  NEJM 5.wav'
    ]
    
    results = []
    
    print(f"\n🎵 Testing {len(audio_files)} audio files...")
    print("=" * 60)
    
    for i, audio_file in enumerate(audio_files):
        print(f"\n[{i+1}/{len(audio_files)}] Testing: {audio_file}")
        
        if not os.path.exists(audio_file):
            continue
        
        try:
            result = detector.predict(audio_file)
            results.append({
                'file': audio_file,
                'prediction': result['predicted_class'],
                'confidence': result['confidence'],
                'tb_prob': result['probabilities'].get('TB', 0),
                'covid_prob': result['probabilities'].get('COVID', 0),
                'normal_prob': result['probabilities'].get('Normal', 0)
            })
            
            for class_name, prob in result['probabilities'].items():
                bar_length = int(prob * 15)
                bar = "█" * bar_length + "░" * (15 - bar_length)
 
  
            if result['predicted_class'] == 'TB' and result['confidence'] > 0.7:
                print(f"TB detected with high confidence!")
            elif result['predicted_class'] == 'TB':
                print(f" Possible TB - low confidence, consult doctor")
            else:
               return
                
        except Exception as e:
            return
    

    if results:
        print(f"\n SUMMARY RESULTS")
        print("=" * 60)
        
        df = pd.DataFrame(results)
        
        print(f"Total files tested: {len(results)}")
        
        pred_counts = df['prediction'].value_counts()
        print(f"\nPrediction distribution:")
        for pred, count in pred_counts.items():
            percentage = count / len(results) * 100
            print(f"  {pred}: {count} files ({percentage:.1f}%)")
        
        avg_confidence = df['confidence'].mean()
        print(f"\nAverage confidence: {avg_confidence:.1%}")
        
        high_conf = (df['confidence'] > 0.8).sum()
        print(f"High confidence (>80%): {high_conf}/{len(results)} predictions")
        
        df.to_csv('prediction_results.csv', index=False)
   
       
        print(f"\n DETAILED RESULTS TABLE:")
        print("-" * 80)
        print(f"{'File':<35} {'Prediction':<10} {'Confidence':<12} {'TB%':<8} {'COVID%':<10}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            filename = row['file'][:32] + "..." if len(row['file']) > 35 else row['file']
            print(f"{filename:<35} {row['prediction']:<10} {row['confidence']:<12.1%} "
                  f"{row['tb_prob']:<8.1%} {row['covid_prob']:<10.1%}")

def test_single_file():
    
    detector = TBCoughDetectorPrediction()
    if not detector.load_model('tb_cough_detector'):
        return
    filename = input("Enter audio filename: ").strip()
    
    if not os.path.exists(filename):
        print(f" File '{filename}' not found!")
        return
    
    try:
        result = detector.predict(filename)
        
        print(f"\n PREDICTION RESULT:")
        print(f"  File: {filename}")
        print(f"  Prediction: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        
        print(f"\n All Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.1%}")
        
        if result['predicted_class'] == 'TB':
            print(f"\n WARNING: TB detected! Consult a medical professional.")
        
    except Exception as e:
        return

def main():
    try:
        choice = input("\nEnter choice (1-3) or press Enter for option 3: ").strip()
        
        if choice == "1" or choice == "3" or choice == "":
            test_all_files()
        elif choice == "2":
            test_single_file()
        else:
            print("Invalid choice, running all files...")
            test_all_files()
            
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    main()