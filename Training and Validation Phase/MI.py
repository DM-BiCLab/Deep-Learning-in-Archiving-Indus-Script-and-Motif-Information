from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import tensorflow 
from tensorflow.keras.models import load_model, model_from_json


import json
import os
import io

labels = ["buffalo","bull","elephant","horned ram","man holding tigers","pashupati","sharp horn and long trunk",
         "short horned bull with head lowered towards a trough","swastik","tiger looking man on tree","unicorn"]

# Set up data generators
train_datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.2, 1.0],
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',)

train_generator = train_datagen.flow_from_directory(
    '#Change to data path',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=labels)

validation_datagen = ImageDataGenerator()

validation_generator = validation_datagen.flow_from_directory(
    '#Change to data path'',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=labels,
    shuffle=False)
 
# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(labels), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
print(model.summary())
# Train the model
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator)
#Save the model
model.save('#Change to model path/motif_cnn_model.h5')

# Plotting accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Motif Model Training and Validation Accuracy')
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
