import streamlit as st

import pandas as pd

import streamlit as st
from PIL import Image
import numpy as np

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from yolov5_v28112023.classify.predict_laurieres import run
from embedding import embed_and_vectorize_pdf, communicate_with_manual

from chatbot_function import first_call, answer_query #speech

# Front End
st.set_page_config(page_title="Talking Toaster", layout="wide")

#st.write('this is version 2')

# Image at the top
custom_html = """
<div class="banner">
    <img src="https://img.freepik.com/premium-photo/3d-illustration-yellow-honeycomb-monochrome-honeycomb-honey_116124-2277.jpg?size=626&ext=jpg&ga=GA1.1.1803636316.1701302400&semt=ais">
</div>
<style>
    .banner {
        width: 120%;
        height: 200px;
        overflow: hidden;
    }
    .banner img {
        width: 100%;
        object-fit: cover;
    }
</style>
"""
# Display the custom HTML
#st.components.v1.html(custom_html)

# Displaying an image on the app

#website_image = Image.open('yolov5_v28112023/toaster_streamlit.jpg')
#st.image(website_image)

# Running our model

st.markdown("""# Welcome to Talking Toaster App 🍞🤖""")

st.markdown("""### Please take of picture of your domestic appliance ☕️""")

#img_file_buffer = st.camera_input("")
image_pred = None
question = None

if st.button("Open Camera") or 'main_button' in st.session_state:

    st.session_state['main_button']=True

    img_file_buffer = st.camera_input("")

    if img_file_buffer:
        img = Image.open(img_file_buffer)
        img.save("camera.jpg")
        image_pred = run(source="camera.jpg")
        st.markdown(f"Your photo is a {image_pred[0]}")

        if 'previous_prediction' in st.session_state and st.session_state['previous_prediction'] != image_pred[0]:
            del st.session_state['welcome_message']

        st.session_state['previous_prediction'] = image_pred[0]

    else:
        st.write(f"We were not able to upload your photo, please try again 🙌")
        if "vector_db" in st.session_state:
            del st.session_state["vector_db"]

    # Calling the PDF
    if image_pred and image_pred[0] in ['oven', 'refrigerator','toaster', 'projector', 'espresso machine']:
        object = image_pred[0]
        #st.session_state['welcome_message']="Test"
        #st.write(st.session_state['welcome_message'])

        # Implementing first ChatGPT 'Hello Message'
        if 'welcome_message' not in st.session_state:
            st.session_state['welcome_message'] = first_call(object)

        st.write(st.session_state['welcome_message'])

            #welcome_message = first_call(object)
            #st.write(welcome_message)


        if "vector_db" not in st.session_state:
            st.session_state["vector_db"] = embed_and_vectorize_pdf(object)

        vector_db = st.session_state["vector_db"]

        question = st.text_input(' ')

        # Calling ChatGPT only after object is recognized.
        if question:
            response = communicate_with_manual(vector_db, question)
            st.write(f"{response}")

            # Implemeting ChatGPT Query
            #st.write(answer_query(question, response, st.session_state['welcome_message']))

            # Implementing Audio

            #audio_file_path = "output.mp3"
            #speech(st.session_state['welcome_message']).stream_to_file(audio_file_path)

            # Play the audio file
            #st.audio(audio_file_path, format='audio/mp3')


    else:
        st.write(f"These object is not talking to you, please try with a toaster or alike")

# Writing the same with file upload

# Add space between buttons
st.markdown("<br>", unsafe_allow_html=True)

# Upload file button
if st.button("Upload File") or 'file_button' in st.session_state:

    st.session_state['file_button']=True

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        img.save("uploaded_image.jpg")
        image_pred = run(source="uploaded_image.jpg")
        st.markdown(f"Your photo is a {image_pred[0]}")


        if 'previous_prediction' in st.session_state and st.session_state['previous_prediction'] != image_pred[0]:
            del st.session_state['welcome_message']

        st.session_state['previous_prediction'] = image_pred[0]

    else:
        st.write(f"We were not able to upload your photo, please try again 🙌")
        if "vector_db" in st.session_state:
            del st.session_state["vector_db"]

    # Calling the PDF
    if image_pred and image_pred[0] in ['oven', 'refrigerator','toaster', 'projector', 'espresso machine']:
        object = image_pred[0]
        #st.session_state['welcome_message']="Test"
        #st.write(st.session_state['welcome_message'])

        # Implementing first ChatGPT 'Hello Message'
        if 'welcome_message' not in st.session_state:
            st.session_state['welcome_message'] = first_call(object)

        st.write(st.session_state['welcome_message'])

            #welcome_message = first_call(object)
            #st.write(welcome_message)


        if "vector_db" not in st.session_state:
            st.session_state["vector_db"] = embed_and_vectorize_pdf(object)

        vector_db = st.session_state["vector_db"]

        question = st.text_input(' ')

        # Calling ChatGPT only after object is recognized.
        if question:
            response = communicate_with_manual(vector_db, question)
            st.write(f"{response}")

            # Implemeting ChatGPT Query
            #st.write(answer_query(question, response, st.session_state['welcome_message']))

    else:
        st.write(f"These object is not talking to you, please try with a toaster or alike")
