import streamlit as st
from PIL import Image
from yolov5_v28112023.classify.predict_laurieres import run

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://t4.ftcdn.net/jpg/05/97/49/09/360_F_597490918_dugDbSuqx6YSRmCaYiZJ6pCr37cXK3Rv.jpg");
background-size: 500%;
background-position: center top;
background-repeat: no-repeat;
background-attachment: local;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

def display_faqs(object):
    st.sidebar.markdown(f"## FAQs for {object}")
    if object in faqs:
        for question in faqs[object]:
            st.sidebar.text(question)

        # Add space after FAQs
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
    else:
        st.sidebar.text("No FAQs available for this product.")

faqs = {
    'oven': ["How to clean the oven?", "What are the cooking modes?", "How to set the timer?"],
    'refrigerator': ["How to adjust the temperature?", "What is the defrosting process?", "How to organize shelves?"],
    'toaster': ["How to adjust browning settings?", "Can it toast frozen bread?", "Cleaning instructions"],
    'projector': ["How to connect to external devices?", "Adjusting projection settings", "Troubleshooting tips"]
}

st.markdown("""# Welcome to Talking Toaster App 🍞🤖""")
st.markdown("""### Choose how to provide the photo of your domestic appliance ☕️""")
image_pred = None
question = None

st.markdown("<br>", unsafe_allow_html=True)

# Button to open the camera
if st.button("Open Camera"):
    img_file_buffer = st.camera_input("")
    if img_file_buffer:
        img = Image.open(img_file_buffer)
        img.save("camera.jpg")
        image_pred = run(source="camera.jpg")
        st.markdown(f"Your photo is a {image_pred[0]}")

# Add space between buttons
st.markdown("<br>", unsafe_allow_html=True)

# Button to upload a file
if st.button("Upload File"):
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        img.save("uploaded_image.jpg")
        image_pred = run(source="uploaded_image.jpg")
        st.markdown(f"Your photo is a {image_pred[0]}")

st.markdown("<br>", unsafe_allow_html=True)

# Display the message only when a wrong appliance is detected
if image_pred and image_pred[0] not in ['oven', 'refrigerator', 'toaster', 'projector']:
    st.write("This object is not talking to you, please try with a toaster or alike")

if image_pred and image_pred[0] in ['oven', 'refrigerator', 'toaster', 'projector']:
    object = image_pred[0]
    display_faqs(object)

    # Add space before text input
    st.markdown("<br>", unsafe_allow_html=True)

    question = st.text_input('Please input your question')
    if question:
        response = "I don't know, fool"
        st.write(response)
