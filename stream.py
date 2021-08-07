# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 04:28:27 2021

@author: ANSH SHANKER
"""

from pickle import load
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import streamlit as st
import numpy as np
from PIL import Image

def adjust_and_pass(input_image,model):
    
    # load and prepare the photograph
    photo = extract_features(input_image)
    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)
    description=description.split(' ', 1)[1]
    description=description.replace('startseq','').replace('endseq','')    
    return description

# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# load the photo
	# convert the image pixels to a numpy array
	image = img_to_array(filename)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = np.argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
@st.cache(allow_output_mutation=True)
def load_models(img):
    model=tensorflow.keras.models.load_model('model_4.h5')
    return adjust_and_pass(img,model)
@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 
st.set_option('deprecation.showfileUploaderEncoding',False)
st.title("Image Captioning")
html_temp="""
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">ImCapto</h2>
</div>
"""
st.subheader("Image")
show_file=st.empty()
img=st.file_uploader("Upload an image",type=["png","jpg"])
if img is None:
    show_file.info("Please upload an image.")
else:
    #img=load_image(img)
    show_file.image(img)
    img=Image.open(img)
    img=img.resize((224,224))    
    st.subheader(load_models(img))
        
    
    
    