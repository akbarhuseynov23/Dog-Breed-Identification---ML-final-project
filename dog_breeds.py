import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
import os
import random

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

#=================================== Title ===============================
st.title("""
Dog Breed Recognition
	""")

#================================= Title Image ===========================
st.text("""""")
img_path_list = ["static/happy_dog1.jpg",
				"static/happy_dog2.jpg"]
index = random.choice([0,1])
top_image = Image.open(img_path_list[index])
st.image(
	        top_image,
	        use_column_width=True,
	    )

#================================= About =================================
st.write("""
## About
	""")

st.write("""
In case you want to get a dog and you already have one special breed in your mind, but don't know the name of that breed
	""")
st.write("""
You can use this app to recognize the breeds.
	""")

#============================ How To Use It ===============================
st.write("""
## Please upload the dog image.
	""")

#========================== File Uploader ===================================
def predict_class(image):
    classify_model = tf.keras.models.load_model('inceptionV3.h5')
    test_image = image.resize((224,224))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255.
    prediction = classify_model.predict(test_image)
    dog_breed = ['affenpinscher','afghan_hound','african_hunting_dog','airedale','american_staffordshire_terrier','appenzeller','australian_terrier','basenji','basset','beagle','bedlington_terrier','bernese_mountain_dog','black-and-tan_coonhound','blenheim_spaniel','bloodhound','bluetick','border_collie','border_terrier','borzoi','boston_bull','bouvier_des_flandres','boxer','brabancon_griffon','briard','brittany_spaniel','bull_mastiff','cairn','cardigan','chesapeake_bay_retriever','chihuahua','chow','clumber','cocker_spaniel','collie','curly-coated_retriever','dandie_dinmont','dhole','dingo','doberman','english_foxhound','english_setter','english_springer','entlebucher','eskimo_dog','flat-coated_retriever','french_bulldog','german_shepherd','german_short-haired_pointer','giant_schnauzer','golden_retriever','gordon_setter','great_dane','great_pyrenees','greater_swiss_mountain_dog','groenendael','ibizan_hound','irish_setter','irish_terrier','irish_water_spaniel','irish_wolfhound','italian_greyhound','japanese_spaniel','keeshond','kelpie','kerry_blue_terrier','komondor','kuvasz','labrador_retriever','lakeland_terrier','leonberg','lhasa','malamute','malinois','maltese_dog','mexican_hairless','miniature_pinscher','miniature_poodle','miniature_schnauzer','newfoundland','norfolk_terrier','norwegian_elkhound','norwich_terrier','old_english_sheepdog','otterhound','papillon','pekinese','pembroke','pomeranian','pug','redbone','rhodesian_ridgeback','rottweiler','saint_bernard','saluki','samoyed','schipperke','scotch_terrier','scottish_deerhound','sealyham_terrier','shetland_sheepdog','shih-tzu','siberian_husky','silky_terrier','soft-coated_wheaten_terrier','staffordshire_bullterrier','standard_poodle','standard_schnauzer','sussex_spaniel','tibetan_mastiff','tibetan_terrier','toy_poodle','toy_terrier','vizsla','walker_hound','weimaraner','welsh_springer_spaniel','west_highland_white_terrier','whippet','wire-haired_fox_terrier','yorkshire_terrier']
    winner = dog_breed[np.argmax(prediction)]
    all_preds = prediction[0]
    predicted_class = f"The dog breed is: {winner}"
    return predicted_class, all_preds

file_uploaded = st.file_uploader("Please upload the image here", type=['png','jpeg','jpg'])
if file_uploaded is not None:
    image = Image.open(file_uploaded)
    figure = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    result, all_preds = predict_class(image)
    st.write(result)
    st.pyplot(figure) 

    

st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("If you consider to get a dog, better adopt one from your local shelter.")
