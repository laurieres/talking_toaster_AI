import streamlit as st
from yolov5_v28112023.classify.predict_laurieres import run
from PIL import Image

img_file_buffer = st.camera_input("")


def image_recognition(img):
    img = Image.open(img)
    img.save("camera.jpg")
    image_pred = run(source="camera.jpg")

    return image_pred

if img_file_buffer:
    result = image_recognition(img_file_buffer)
    print(result)
