import streamlit as st
import numpy as np  
import pandas as pd
from audiorecorder import audiorecorder

#Loading model

from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("./Trained_Models/Speech_Emotion_Recognition_Model.h5")



def show_predict_page():
    st.title("Emotion Analyser from Speech")

    st.write("""### We need to record audio to predict/analyse the emotion""")
    
    st.title(" ")

    
    st.title("Audio Recorder")
    audio = audiorecorder("Click to record", "Recording...")

    if len(audio) > 0:
        # To play audio in frontend:
        st.audio(audio)
    
        # To save audio to a file:
        wav_file = open("audio.mp3", "wb")
        wav_file.write(audio.tobytes())

    ok = st.button("Predict Emotion")
    if ok:
        demo_mfcc, demo_pitch, demo_mag, demo_chrom = get_audio_features(audio,sampling_rate)

        mfcc = pd.Series(demo_mfcc)
        pit = pd.Series(demo_pitch)
        mag = pd.Series(demo_mag)
        C = pd.Series(demo_chrom)
        demo_audio_features = pd.concat([mfcc,pit,mag,C],ignore_index=True)

        demo_audio_features= np.expand_dims(demo_audio_features, axis=0)
        demo_audio_features= np.expand_dims(demo_audio_features, axis=2)

        livepreds = loaded_model.predict(demo_audio_features, batch_size=32, verbose=1)
        index = livepreds.argmax(axis=1).item()
        emotions[index]
        st.subheader(f"The predicted emotion is ${emotions[index]}")