import streamlit as st

import pandas as pd

import streamlit as st
from PIL import Image
import numpy as np

from yolov5_v28112023.classify.predict_laurieres import run

# acceptable_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']

st.markdown("""# Welcome to Toaster Website 🍞""")

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Check the type of img_array:
    # Should output: <class 'numpy.ndarray'>
    st.write(type(img_array))

    # Check the shape of img_array:
    # Should output shape: (height, width, channels)
    st.write(img_array.shape)

res = run(source='oven_test.png')

st.write(f"Your photo is {res[0]}")

#path_to_img_file_buffer = ''

# Calling the model and make a predictions

#image_pred = run(source='img_file_buffer')

#if image_pred is not None:
#    st.write(f"Your photo is {image_pred[0]} with a probability of {round(image_pred[1].item(),2)}")
#else:
#    st.write(f"We were not able to upload your photo, please try again 🙌")
