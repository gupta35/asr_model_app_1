import streamlit as st
import speech_recognition as sr
from pocketsphinx import pocketsphinx, Jsgf, FsgModel
import requests
import os


st.title("Speech to text recognition")

st.markdown("## Here we use pocketsphinx model for automatic speech recognition")

audio = st.file_uploader(label = "Upload your audio file here in .wav format", type=['wav'])


# audio_file = '/Users/kapilgupta/Downloads/audio/videoplayback.wav'
text_filename = "https://drive.google.com/drive/folders/1btjSoOWAiRMxgSkNfF8BhcibZsEUNnXb?usp=sharing"


language_model = 'https://drive.google.com/file/d/1kmtij_yY98aRMfkvkMLPn6DO0QVg0ic-/view?usp=sharing'
acoustic_model = 'https://drive.google.com/drive/folders/1eqpxJWolJf06uZcuKWe138UgvdeZVCWZ?usp=sharing'
pronunciation_dict = 'https://drive.google.com/file/d/1zdevgBOSZblV-7zMgCAKviSIoKZe0fgX/view?usp=sharing'

@st.cache
def model(audio, text_filename):
    framerate = 100
    config = pocketsphinx.Config()
    config.set_string('-hmm', acoustic_model)
    config.set_string('-lm', language_model)
    config.set_string('-dict', pronunciation_dict)
    decoder = pocketsphinx.Decoder(config)

    def recognize_sphinx(audio, show_all=True):
        decoder.start_utt()
        decoder.process_raw(audio.get_raw_data(), False, True)
        decoder.end_utt()
        hypothesis = decoder.hyp()
        return decoder, hypothesis.hypstr

    # Create a Recognizer instance
    r = sr.Recognizer()

    # Set the recognize_sphinx() function as the speech recognition method
    r.recognize_sphinx = recognize_sphinx

    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
        sample_rate = audio.sample_rate
        decoder, recognized_text = r.recognize_sphinx(audio, show_all=True)


    with open(text_filename, 'w') as text_file:
        for seg in decoder.seg():
            segment_info = (seg.word, seg.start_frame/sample_rate, seg.end_frame/sample_rate)
            text_file.write(str(segment_info) + "\n")

    return segment_info


if audio is not None:
    with st.spinner("code is at Working! "):
        segment_info = model(audio, text_filename)
        st.write(segment_info)
    st.balloons()
else:
    st.write("Upload an audio")

