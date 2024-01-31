import streamlit as st
import os
from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing  import image
from tensorflow.keras.layers import GlobalMaxPooling2D,MaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import time

def save_uploaded_file(uploded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploded_file.getbuffer())
        return 1
    except:
        return 0

model= ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable =False

model= tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

feature_lis= np.array(pickle.load(open('features.pkl','rb')))
filenames= pickle.load(open('path.pkl','rb'))

def  features(path , model):
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    predicted = model.predict(preprocessed_img).flatten()
    features_extracted= predicted/norm(predicted)
    return(features_extracted)


st.title('Fashion Recommender System')
st.divider()

with st.sidebar:
    st.header('CHOOSE')
    st.divider()
    user_input= st.radio('',('File Upload','Camera Upload'))

if user_input=='Camera Upload':
    st.subheader('PLEASE CLICK THE PICTURE.')
    picture = st.camera_input('click')
    if picture is not None:
        img = Image.open(picture)
        st.image(img,width=350,caption='clicked picture')

        img = image.load_img(picture, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        predicted = model.predict(preprocessed_img).flatten()
        features_extracted = predicted / norm(predicted)
        neighbours = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
        neighbours.fit(feature_lis)
        distaces, indices = neighbours.kneighbors([features_extracted])
        st.divider()

        with st.spinner('Please hold on...'):
            time.sleep(5)
        st.success('Done!')
        st.subheader('Just Designed for you.')
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            image = Image.open(filenames[indices[0][0]])
            st.image(image, width=150)

        with col2:
            image = Image.open(filenames[indices[0][1]])
            st.image(image, width=150)

        with col3:
            image = Image.open(filenames[indices[0][2]])
            st.image(image, width=150)

        with col4:
            image = Image.open(filenames[indices[0][3]])
            st.image(image, width=150)

        with col5:
            image = Image.open(filenames[indices[0][4]])
            st.image(image, width=150)
        st.toast('Hurrah We found  specially for you.', icon='😍')
        st.toast('Hurrah We found specially for you.', icon='😍')
        st.toast('Hurrah We found specially for you.', icon='😍')
        st.balloons()

else:
    st.subheader('BROWSE YOUR IMAGE TO UPLOAD.')
    uploaded_file = st.file_uploader(type=['jpg','jpeg','png'],label='')  #upload a file.
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            uploaded_image= Image.open(uploaded_file)
            st.image(uploaded_image, width= 350,caption='Uploaded File')

            #feature extract
            features_extracted= features((os.path.join('uploads',uploaded_file.name)),model)
            neighbours = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
            neighbours.fit(feature_lis)
            distaces, indices = neighbours.kneighbors([features_extracted])
            st.divider()

            with st.spinner('Please hold on...'):
                time.sleep(5)
            st.success('Done!')
            st.subheader('Just Designed For You.')
            col1, col2, col3,col4,col5 = st.columns(5)

            with col1:
                    image = Image.open(filenames[indices[0][0]])
                    st.image(image, width=150)

            with col2:
                    image = Image.open(filenames[indices[0][1]])
                    st.image(image, width=150)

            with col3:
                    image = Image.open(filenames[indices[0][2]])
                    st.image(image, width=150)

            with col4:
                    image = Image.open(filenames[indices[0][3]])
                    st.image(image, width=150)

            with col5:
                    image = Image.open(filenames[indices[0][4]])
                    st.image(image, width=150)

            st.toast('Hurrah We found  specially for you.', icon='😍')
            st.toast('Hurrah We found specially for you.', icon='😍')
            st.toast('Hurrah We found specially for you.', icon='😍')
            st.snow()













