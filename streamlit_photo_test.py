import streamlit as st

import pandas as pd

import streamlit as st
from PIL import Image
import numpy as np

from yolov5_v28112023.classify.predict_laurieres import run

# acceptable_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']

#st.set_page_config(layout="wide")

# Image at the top

custom_html = """
<div class="banner">
    <img src="https://t4.ftcdn.net/jpg/02/71/29/75/360_F_271297554_0DAlzyFb8jzYg0lfmUOzyhtMer0orz4h.jpg">
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
# st.components.v1.html(custom_html)

website_image = Image.open('yolov5_v28112023/toaster_streamlit.jpg')

st.image(website_image)

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


#st.sidebar.test_input('Name')
