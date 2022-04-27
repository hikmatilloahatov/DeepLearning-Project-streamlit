import streamlit as st
from fastai.vision.all import *
import plotly.express as px

#title
st.title('DeepLearning Project')
st.subheader('Classification model for transports')

hd_st_style = """
        <style>
         #MainMenu {visibility: hidden;}
         footer {visibility: hidden;}
         header {visibility: hidden;}
        </style>
        """
#file_uploader
file = st.file_uploader('Upload image', type=['svg', 'png', 'jpg', 'jpeg'])

#display and predict it
if file and file.type != 'image/svg+xml':
    
    st.image(file)
    image = PILImage.create(file)
    model = load_learner('transport_model.pkl')
    
    pred, pred_id, probs = model.predict(image)

    st.success(f"This is a {pred}")
    st.info(f"Probability: {probs[pred_id]*100:.1f}%")
    #plotly chart
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
else:
    st.info("Please, don't upload image of svg format!")

