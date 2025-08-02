import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class TBCoughDetector:
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
            if not os.path.exists(audio_path):
                print(f"Warning: File {audio_path} not found!")
                return np.zeros(self.n_mels * 2 + self.n_mfcc * 3 + 12 + 12 + 9)
            
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            if len(y) == 0:
                print(f"Warning: Empty audio file {audio_path}")
                return np.zeros(self.n_mels * 2 + self.n_mfcc * 3 + 12 + 12 + 9)
            
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
    
    def prepare_dataset(self, data_df):
        features = []
        labels = []
        
        for idx, row in data_df.iterrows():
            print(f"Processing file {idx+1}/{len(data_df)}: {row['audio_path']}")
            
            feature_vector = self.extract_features(row['audio_path'])
            features.append(feature_vector)
            labels.append(row['label'])
        
        X = np.array(features)
        y = self.label_encoder.fit_transform(labels)
        X = self.scaler.fit_transform(X)
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique_labels, counts))
        print(f"Label distribution: {label_dist}")
        
        return X, y
    
    def build_model(self, input_dim, num_classes=3):
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_small_dataset(self, X, y, epochs=50):
        self.model = self.build_model(X.shape[1], len(np.unique(y)))
        early_stopping = callbacks.EarlyStopping(
            monitor='loss', patience=20, restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, min_lr=1e-7
        )
    
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        class_weights = {}
        for i, count in enumerate(counts):
            class_weights[unique[i]] = total / (len(unique) * count)
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=min(8, len(X)),  
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        n_samples = len(X)
        n_classes = len(np.unique(y))
        
        print(f"\n📊 Dataset Analysis:")
        print(f"Total samples: {n_samples}")
        print(f"Number of classes: {n_classes}")
        print(f"Features: {X.shape[1]}")
        

        unique, counts = np.unique(y, return_counts=True)
        min_class_size = np.min(counts)
        
        print(f"Samples per class: {dict(zip(unique, counts))}")
        print(f"Smallest class has {min_class_size} samples")
        
        if n_samples < 10 or min_class_size < 2:
            return self.train_small_dataset(X, y, epochs)
        
    
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, stratify=y, random_state=42
            )
        except ValueError as e:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
        
        
        self.model = self.build_model(X.shape[1], len(np.unique(y)))
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7
        )
        
    
        class_weights = {}
        unique, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        for i, count in enumerate(counts):
            class_weights[unique[i]] = total / (len(unique) * count)
    
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def evaluate_small_dataset(self, X, y):
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        
        if len(X) >= 3: 
            cv_scores = cross_val_score(rf, X, y, cv=min(3, len(X)//2))
        else:
            rf.fit(X, y)
            train_score = rf.score(X, y)
        if self.model is not None:
            y_pred_proba = self.model.predict(X)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            labels = self.label_encoder.classes_
            for i, (true_label, pred_label, confidence) in enumerate(zip(y, y_pred, np.max(y_pred_proba, axis=1))):
                true_name = labels[true_label]
                pred_name = labels[pred_label]
    
    def predict(self, audio_path):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        features = self.extract_features(audio_path)
        features = features.reshape(1, -1)
        features = self.scaler.transform(features)
        
        probabilities = self.model.predict(features)[0]
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
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save!")
        self.model.save(f"{filepath}_model.h5")

        import joblib
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        joblib.dump(self.label_encoder, f"{filepath}_encoder.pkl")
    
    def load_model(self, filepath):
        import joblib
        self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.label_encoder = joblib.load(f"{filepath}_encoder.pkl")
        

def main():
    detector = TBCoughDetector()
    data = {
        'audio_path': [
            'Whooping Cough in an Adult  NEJM 1.wav',
            'Whooping Cough in an Adult  NEJM 2.wav',
            'Whooping Cough in an Adult  NEJM 3.wav'
        ],
        'label': [
            'TB',
            'TB', 
            'COVID'
        ]
    }
    
   
    if len(data['audio_path']) != len(data['label']):
        return
    
    try:
        df = pd.DataFrame(data)
    except Exception as e:
        return
    try:
        X, y = detector.prepare_dataset(df)
    except Exception as e:
        return
    try:
        history = detector.train(X, y, epochs=30)
    except Exception as e:
        return

    detector.evaluate_small_dataset(X, y)
    try:
        detector.save_model('tb_cough_detector')
    except Exception as e:
        return
    

if __name__ == "__main__":
    main()