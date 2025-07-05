import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import tempfile
import joblib
import sounddevice as sd
import matplotlib.pyplot as plt


model = joblib.load("model.pkl")


def extract_features(file_path):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    result = np.array([])
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfccs, chroma, mel))
    return result.reshape(1, -1), X, sample_rate

def plot_waveform(audio_data, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio_data, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)


st.set_page_config(page_title="🎤 Voice Emotion Detection", layout="centered")
st.title("🎧 Emotion Detection from Voice")

option = st.radio("Choose input method:", ("Upload WAV File", "Record Live Audio"))

if option == "Upload WAV File":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        try:
            features, audio, sr = extract_features(temp_path)
            plot_waveform(audio, sr)

            st.audio(temp_path, format='audio/wav') 

            prediction = model.predict(features)
            result_text = f"Detected Emotion: {prediction[0].capitalize()}"
            st.success(f"🎯 {result_text}")

            st.download_button(
                label="📥 Download Result",
                data=result_text,
                file_name="emotion_result.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Error: {e}")

elif option == "Record Live Audio":
    duration = st.slider("Select recording duration (seconds):", 2, 10, 5)
    if st.button("🎙️ Start Recording"):
        try:
            sample_rate = 22050
            st.info("Recording... Speak now.")
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()
            st.success("Recording complete!")

            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                sf.write(tmp_wav.name, audio, sample_rate)
                features, audio_data, sr = extract_features(tmp_wav.name)

                plot_waveform(audio_data, sr)

                st.audio(tmp_wav.name, format='audio/wav')  

                prediction = model.predict(features)
                result_text = f"Detected Emotion: {prediction[0].capitalize()}"
                st.success(f"🎯 {result_text}")

               
                st.download_button(
                    label="📥 Download Result",
                    data=result_text,
                    file_name="emotion_result.txt",
                    mime="text/plain"
                )
        except Exception as e:
            st.error(f"Recording failed: {e}")
