import streamlit as st

import pandas as pd

import streamlit as st
from PIL import Image
import numpy as np

from yolov5_v28112023.classify.predict_laurieres import run
from embedding import embed_and_vectorize_pdf, communicate_with_manual

st.set_page_config(layout="wide")

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
st.components.v1.html(custom_html)

# Displaying an image on the app

#website_image = Image.open('yolov5_v28112023/toaster_streamlit.jpg')
#st.image(website_image)

# Running our model

st.markdown("""# Welcome to Talking Toaster App 🍞🤖""")

st.markdown("""### Please take of picture of your domestic appliance ☕️""")

img_file_buffer = st.camera_input("")

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    img.save("camera.jpg")

    image_pred = run(source="camera.jpg")

    st.markdown(f"Your photo is a {image_pred[0]}")
    #st.write(f"Your photo is {image_pred[0]} with a probability of {round(image_pred[1].item(),2)}")

else:
    st.write(f"We were not able to upload your photo, please try again 🙌")


# Calling the PDF

question = st.text_input('Please input your question')

vector_db = embed_and_vectorize_pdf(image_pred[0])

communicate_with_manual(vector_db, question)

# Calling ChatGPT
