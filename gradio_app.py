from logging import warning
import gradio as gr
import joblib
import warnings

from matplotlib.pyplot import flag

warnings.filterwarnings("ignore")
# load the model
model = joblib.load("rf_mnist.pkl")
scaler = joblib.load("scaler_mnist.save")
def predict(image):
    image = image.flatten()
    # return image
    scaled_image = scaler.transform([image])
    prediction = model.predict(scaled_image)
    return prediction[0]

iface = gr.Interface(fn=predict, inputs="sketchpad",
 outputs="text", title="MNIST Using RF - M Zean",
 description="This is the ist app in the list of \
     beginner projects and the ones that will be \
         deployed to the server for the world.",
theme="grass",
article="This model was not trained on th augmented \
    data, so it does not predict well on the \
        scaled, tilted etc digits. For Better performance\
            write the digits straight in the middle\
                preferably. Thanks",
allow_flagging="never")

iface.launch()