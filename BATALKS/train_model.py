import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    
    mfccs = np.mean(mfcc.T, axis=0)
    chroma = np.mean(chroma.T, axis=0)
    mel = np.mean(mel.T, axis=0)
    
    return np.hstack([mfccs, chroma, mel])


dataset_path = "path_to_audio_dataset"  
emotion_labels = {
    '01': 'neutral',
    '02': 'calm',                
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


features = []
labels = []

for file in os.listdir(dataset_path):
    if file.endswith(".wav"):
        file_path = os.path.join(dataset_path, file)
        emotion_code = file.split("-")[2]
        emotion = emotion_labels.get(emotion_code)
        
        try:
            data = extract_features(file_path)
            features.append(data)
            labels.append(emotion)
        except Exception as e:
            print("Error processing file:", file)


X = np.array(features)
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
