import streamlit as st
import pandas as pd
import numpy as np
from streamlit_drawable_canvas import st_canvas

import torch
import torchvision.transforms as transforms

from modelNN import DigitClassifier
from modelCNN2 import SimpleCNN2

import cv2

st.set_page_config(layout="centered")

st.title("Quick, Draw!")
st.text("But it Only works with Digits - a simple MNIST classifier")

# options panel
with st.sidebar:
    st.title("Options!")

    model_selection = st.selectbox("Model", ["Simple CNN", "Simple NN"])

    # pen stroke width
    strk_width = st.slider(
        "Pen Stroke Width", min_value=3, max_value=15, value=10, step=1
    )

    # predictions can be affected by the normalization of the input,
    normalize = True
    normalize = st.checkbox("Custom Input Normalization", value=False)
    if normalize:
        normalization = st.slider(
            "Normalization?",
            min_value=1.0,
            max_value=255.0,
            value=127.0,
            step=1.0,
            format="%d",
        )


# select your model
@st.cache_resource
def load_model(model_selection):
    model = None
    if model_selection == "Simple NN":
        model = DigitClassifier(28 * 28 * 1, 10)
        try:
            model.load_state_dict(torch.load("model_nn_adam_b32.pth"))
            print("Model loaded! (NN)")
        except RuntimeError as e:
            st.text(f"An error occurred while loading the model: {e}")
    elif model_selection == "Simple CNN":
        model = SimpleCNN2(10)
        try:
            model.load_state_dict(torch.load("model_cnn2_adam_b32.pth"))
            print("Model loaded! (CNN)")
        except RuntimeError as e:
            st.text(f"An error occurred while loading the model: {e}")
    else:
        st.text("Model not found?")

    return model


md = load_model(model_selection)
if md:
    md.eval()
else:
    st.text("Model loading failed...")


# canvas and drawing display
canv, display = st.columns(2)
sub_disp1, sub_disp2 = st.columns(2)

img = None
canvas = None

with canv:
    st.subheader("Draw!")
    # the square canvas
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=strk_width,
        stroke_color="rgb(255, 255, 255)",
        background_color="#000",
        update_streamlit=True,
        height=140,
        width=140,
        drawing_mode="freedraw",
        display_toolbar=True,
        initial_drawing={},
        key="canvas",
    )

predicted_class = None
predictions = None

with display:
    st.subheader("Your Drawing!")
    if canvas and canvas.image_data is not None:
        # get rid of the alpha channel and resize to 28x28
        canvas.image_data = cv2.cvtColor(canvas.image_data, cv2.COLOR_RGBA2RGB)
        img = cv2.resize(canvas.image_data, (28, 28))

        # convert to tensor (acceptable by simple NN and CNN) and grayscale
        img_tensor = (torch.tensor(img).float() / (normalization if normalize else 127.0)).permute(2, 0, 1)  # [28, 28, 3] -> [3, 28, 28]
        transform = transforms.Grayscale(num_output_channels=1)  # [3, 28, 28] -> [1, 28, 28]
        img_tensor = transform(img_tensor)

        if model_selection == "Simple CNN":
            img_tensor = img_tensor.unsqueeze(0)  # [1, 28, 28] -> [1, 1, 28, 28]

        # display raw image and processed tensor
        st.image(img_tensor.view(28, 28).numpy(), clamp=True, width=140)
        st.text("⬆️ smol tensor (28 * 28) as input")

        # feed into model and get prediction
        with torch.no_grad():
            output = md(img_tensor)
            # predicted_class = torch.argmax(output, dim=1).item()
            raw_output_probabilities = torch.softmax(output, dim=1).squeeze().tolist()
            predictions = [
                {"class": i, "probability": prob}
                for i, prob in enumerate(raw_output_probabilities)
            ]
            predictions.sort(key=lambda x: x["probability"], reverse=True)

    else:
        pass


# Predictions
st.subheader("I'm seeing a...")
c1, c2 = st.columns(2)
if predictions is None:
    st.text("I can't see anything yet...")
else:
    with c1:
        st.markdown(
            f"<div style='display: flex; align-items: baseline; margin-top: 0;'><h1 style='font-size: 100px'>{predictions[0]['class']}</h1><p>({predictions[0]['probability'] * 100 :.3f}%)</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.text("I'm seeing other things too...")
        st.text(
            f"... {predictions[1]['class']} ({predictions[1]['probability'] * 100 :.3f}%)"
        )
        st.text(
            f"... {predictions[2]['class']} ({predictions[2]['probability'] * 100 :.3f}%)"
        )
        st.text(
            f"... {predictions[3]['class']} ({predictions[3]['probability'] * 100 :.3f}%)"
        )
