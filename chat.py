import streamlit as st
import time
from PIL import Image
import requests

st.title("Shakespeare GPT")
image = Image.open('Shakesphere.jpeg')
st.image(image, caption='William Shakespeare',use_column_width=True)

prompt = st.text_input("Enter a prompt")

data = {'input': prompt}

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)


if st.button("Generate"):
    with st.spinner("Generating next 500 chars..."):
        model_output = requests.post("https://llm-app-pbvf6ehg2a-ue.a.run.app/predict", json= data)
        print(model_output)
        model_output = model_output.json()['prediction']

    st.write_stream(stream_data(model_output))
    st.button("Thumbs up 👍🏻")
    st.button("Thumbs down 👎🏻")