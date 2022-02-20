import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import joblib


rf_model = joblib.load("rf_mnist.pkl")
scaler = joblib.load("scaler_mnist.save")

st.title("MNIST GUI and ML model by M Zean for Practice")

canvas = st_canvas(
    fill_color="rgba(255, 165)",
    background_color="white",
    stroke_width=10,
    update_streamlit = True,
    height = 200,
    width = 200,
    drawing_mode="freedraw",
    key="canvas"
)
def scale_image(image):
    image = scaler.transform([image])
    return image

def predict(image):
    
    prediction = rf_model.predict(image)
    return prediction

if canvas.image_data is not None:
    # st.image(canvas.image_data)
    if 0 in canvas.image_data:
        image = np.array(tf.image.resize(canvas.image_data, [28,28])[:, :, 0])
        image = image.flatten()
        image = scale_image(image)
        pred = predict(image)

        # image = image[:, :, 0]
        # image = np.expand_dims(i, axis=0)
        # image = np.squeeze(image, axis=0)

        st.write(image.shape)
        st.header(f"The prediction is {pred[0]}")
# use the scaler from the file which we used to scale the data
# transform the data
# make predicting from the image
# show the result
# upload the project to github
# get heroku up and runnig