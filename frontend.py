import streamlit as st
import pandas as pd
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.title('Quick, Draw!')
st.text('But it Only works with Digits - a simple MNIST classifier')

canvas, display = st.columns(2)

with canvas:
    st.subheader('Draw!')
    # the square canvas
    canvas = st_canvas(
        fill_color='rgba(255, 165, 0, 0.3)',
        stroke_width=5,
        stroke_color='rgb(0, 0, 0)',
        background_color='#FFF',
        update_streamlit=True,
        height=150,
        width=150,
        drawing_mode='freedraw',
        display_toolbar=True,
        initial_drawing={},
        key='canvas'
    )
with display:
    st.subheader('Your Drawing!')
    if canvas and canvas.image_data is not None:
        st.image(canvas.image_data)
    else:
        pass

# Thought: pass the image (once available) to the model and get the prediction
st.subheader('Prediction!')
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"<div style='display: flex; align-items: baseline; margin-top: 0;'><h1 style='font-size: 100px'>{7}</h1><p>({90}%)</p></div>", unsafe_allow_html=True)
with c2:
    st.text('\n')
    st.text('...or maybe a 1 (49%)')
    st.text('...or a 2 (30%)')
    st.text('...or a 0 (10%)')