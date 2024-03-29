import pickle
import numpy as np
import cv2
import tensorflow
import  streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D,MaxPooling2D
from tensorflow.keras.applications.resnet50  import ResNet50, preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

feature_lis= np.array(pickle.load(open('features.pkl','rb')))
filenames= pickle.load(open('path.pkl','rb'))

model= ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable =False

model= tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
img = image.load_img('sample/1163.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
predicted = model.predict(preprocessed_img).flatten()
print(preprocessed_img.shape)
features_extracted= predicted/norm(predicted)

neighbours= NearestNeighbors(n_neighbors=5,algorithm='brute', metric='euclidean')
neighbours.fit(feature_lis)
distaces, indices = neighbours.kneighbors([features_extracted])

#for i in indices[0]:
    #print(filenames[i])


for i in indices[0]:
    temp_var = cv2.imread(filenames[i])
    cv2.imshow('output', cv2.resize(temp_var, (512, 512)))
    cv2.waitKey(0)


