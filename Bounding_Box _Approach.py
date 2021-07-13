# Assuming the ResNet50 model
# Libraries
import tensorflow as tf
from tensorflow import keras
import sklearn
import matplotlib.pyplot as plt
import seaborn
import numpy as np
%matplotlib inline

# Model 
# Define 2 Output layers: 1 for Bounding Boxes predictions and other for Class predictions
base_model = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet")
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
# Outputs (Number of Classes will change)
class_output = keras.layers.Dense(5, activation="softmax", name = "Class_Prediction")(avg)
bounding_box_prediction = keras.layers.Dense(4, name="Bounding_Box_Prediction")(avg)
model = keras.Model(inputs = [base_model.input], outputs = [class_output, bounding_box_prediction])

# Plot model using Keras
keras.utils.plot_model(model , show_layer_names=True, show_shapes=True, to_file="Bounding_Box_Prediction.png")

model.compile(optimizer='adam',
              loss=['sparse_categorical_crossentropy', "mse"],
              loss_weights=[0.8, 0.2],
              metrics=['accuracy'])
