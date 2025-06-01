
# -*- coding: utf-8 -*-
"""
Victor Manuel Gonzalez Aguayo
Machine Learning ago-dic 2023
Catedratico: Luis Carlos Padierna

Práctica Autónoma 5: Comparador de Clasificadores Feature Maps
"""

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from numpy import expand_dims

# load the VGG16 model
model = VGG16()

# Define models for each block of convolutions
block1_model = Model(inputs=model.input, outputs=model.layers[1].output)
block2_model = Model(inputs=model.input, outputs=model.layers[5].output)
block3_model = Model(inputs=model.input, outputs=model.layers[9].output)
block4_model = Model(inputs=model.input, outputs=model.layers[13].output)
block5_model = Model(inputs=model.input, outputs=model.layers[17].output)

# List of block models
block_models = [block1_model, block2_model, block3_model, block4_model, block5_model]

# Load and preprocess the image
path = 'biznaga.jpg'
img = load_img(path, target_size=(224, 224))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

# Generate feature maps for each block
for i, block_model in enumerate(block_models):
    feature_maps = block_model.predict(img)
    # Plot all feature maps in an 8x8 grid
    square = 8
    ix = 1
    for _ in range(square):
        for _ in range(square):
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    pyplot.suptitle(f'Block {i+1}')
    pyplot.show()
