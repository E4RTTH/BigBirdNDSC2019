import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib.image as img
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# def build_resnet50( image_shape=(224, 224, 3), 
#                     n_classes=26, 
#                     load_pretrained=False,
#                     freeze_layers_from='base_model'):
#     # Decide if load pretrained weights from imagenet
#     if load_pretrained:
#         weights = 'imagenet'
#     else:
#         weights = None

#     # Get base model
#     base_model = ResNet50(include_top=False, weights=weights, input_tensor=None, input_shape=image_shape)
    
#     # Add final layers
#     x = base_model.output
#     x = Flatten()(x)
#     prediction = Dense(n_classes, activation='softmax', name='fc1000')(x)

#     # This is the model we will train
#     model = Model(input=base_model.input, output=prediction)

#     # Freeze some layers
#     if freeze_layers_from == 'base_model':
#         for layer in base_model.layers:
#             print ('Freezing base model layers')
#             layer.trainable = False
#         else:
#             for i, layer in enumerate(model.layers):
#                 print(i, layer.name)
#             for layer in model.layers[:freeze_layers_from]:
#                layer.trainable = False
#             for layer in model.layers[freeze_layers_from:]:
#                layer.trainable = True

#     return model

NUM_CLASSES = 26

model = Sequential()

model.add(ResNet50(include_top = False, pooling='avg', weights='imagenet'))

# 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

image_size = 224
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100
train_generator = data_generator.flow_from_directory(
        '/mnt/wangpangpang/mobile_train',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_TRAINING,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '/mnt/wangpangpang/mobile_validate',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_VALIDATION,
        class_mode='categorical') 

EARLY_STOP_PATIENCE = 3
cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = 'working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

NUM_EPOCHS = 100
STEPS_PER_EPOCH_TRAINING = 10 
STEPS_PER_EPOCH_VALIDATION = 10
fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
        epochs = NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=STEPS_PER_EPOCH_VALIDATION,
        callbacks=[cb_checkpointer, cb_early_stopper]
)
model.load_weights("../working/best.hdf5")


