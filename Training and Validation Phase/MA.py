import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import time
import copy


import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Model, layers
from tensorflow.keras.models import load_model, model_from_json


import json
import os
import io

WORKING_DIRECTORY = '/'

os.chdir(WORKING_DIRECTORY)
labels = ["M8","M12","M15","M17","M19","M28","M48","M51","M53","M59","M102","M104","M141","M162","M173","M174","M176","M204","M205",
          "M211","M216","M245","M249","M267","M287","M294","M296","M302","M307","M326","M327","M328","M330","M336","M342","M387","M389","M391","M403","Other"]
print(len(labels))
train_datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.2,1.0],
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    # train_data_dir,
    '#change to data path',
    batch_size=32, #batch size of 
    class_mode='categorical', ##more than 2 classification
    target_size=(224,224)) ##resnet target size


validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
    # val_data_dir,
    '#change to data path',
    shuffle=False,
    batch_size = 32,
    class_mode='categorical',
    target_size=(224,224))


# Load ResNet50 with pretrained weights
conv_base = ResNet50(include_top=False, weights='imagenet')

# Freeze all layers in the ResNet50 base
for layer in conv_base.layers:
    layer.trainable = False

# Build the top layers for your model
x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)  # Add dropout for regularization
x = layers.Dense(256, activation='relu')(x)
predictions = layers.Dense(40, activation='softmax')(x)  

# Create the final model
model = Model(conv_base.input, predictions)

# Compile the model
optimizer = tf.keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

#get summary of the model
print(model.summary())
#train the model 
history_data = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator)
#Save the model
model.save('#change to model path/Mamodel.h5')
# Plotting accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Mahadevan Model Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Motif Model Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



