import streamlit as st
import pandas as pd
import numpy as np
from streamlit_drawable_canvas import st_canvas

import torch
from modelNN import DigitClassifier

# load model
@st.cache_resource
def load_model():
    model = DigitClassifier(28*28, 10)
    try:
        model.load_state_dict(torch.load('model_nn_0430.pth'))
        print('Model loaded!')
        return model
    except RuntimeError as e:
        st.text(f'An error occurred while loading the model: {e}')


md = load_model()
if md:
    st.text('Model loaded!')
    md.eval()
else:
    st.text('Model loading failed!')


st.title('Quick, Draw!')
st.text('But it Only works with Digits - a simple MNIST classifier')

canv, display = st.columns(2)
sub_disp1, sub_disp2 = st.columns(2)

img = None
canvas = None

with canv:
    st.subheader('Draw!')
    # the square canvas
    canvas = st_canvas(
        fill_color='rgba(0, 0, 0, 0)',
        stroke_width=15,
        stroke_color='rgb(255, 255, 255)',
        background_color='#000',
        update_streamlit=True,
        height=140,
        width=140,
        drawing_mode='freedraw',
        display_toolbar=True,
        initial_drawing={},
        key='canvas'
    )

with display:
    predicted_class = None
    st.subheader('Your Drawing!')
    if canvas and canvas.image_data is not None:
        # resize image_data to 28x28
        img = canvas.image_data.astype(np.uint8)
        img = img[::5, ::5]
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(canvas.image_data)
        with col2:
            st.image(img)
            st.text(f'{img.shape}')

        # try passing the image to the model and get some prediction as text
        img_tensor = torch.tensor(img).float()
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.mean(dim=3, keepdim=True)  # Convert image to grayscale
        st.text(f"tensor shape {img_tensor.shape}")

        with torch.no_grad():
            output = md(img_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            # print(predicted_class)
            raw_output_probabilities = torch.softmax(output, dim=1).squeeze().tolist()
            predictions = [{'class': i, 'probability': prob} for i, prob in enumerate(raw_output_probabilities)]
            predictions.sort(key=lambda x: x['probability'], reverse=True)
            print(predictions)
        
    else:
        pass

# Thought: pass the image (once img is updated) to the model and get the prediction
st.subheader('Prediction!')
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"<div style='display: flex; align-items: baseline; margin-top: 0;'><h1 style='font-size: 100px'>{predicted_class}</h1><p>({predictions[0]['probability'] * 100}%)</p></div>", unsafe_allow_html=True)
with c2:
    st.text('\n')
    st.text(f"... or a {predictions[1]['class']} ({predictions[1]['probability'] * 100}%)")
    st.text(f"... or a {predictions[2]['class']} ({predictions[2]['probability'] * 100}%)")
    st.text(f"... or a {predictions[3]['class']} ({predictions[3]['probability'] * 100}%)")