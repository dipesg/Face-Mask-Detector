import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logger
from imutils import paths
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from preprocessing import Preprocess

class Train:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../logs/train.logs", 'a+')
    def model_building(self):
        # Preprocess data
        self.log_writer.log(self.file_object,"Inside training preprocess data begins...")
        data, labels = Preprocess().preprocess("../dataset", ["with_mask", "without_mask"])

        # Doing one hot encoding
        self.log_writer.log(self.file_object,"Inside training one hot encoding begins...")
        label = Preprocess().one_hot(labels)

        # Changing to numpy array
        self.log_writer.log(self.file_object,"Inside training changing to numpy array...")
        data, labels = Preprocess().change_to_numpy(data, label)


        # Now train_test_split
        self.log_writer.log(self.file_object,"Performing train_test_split...")
        x_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.20,
                                                    stratify=labels,random_state=42)

        # Constructing image data generator
        self.log_writer.log(self.file_object,"Constructing image data generator...")
        augment = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")


        # Loading MobileNetV2 network, Here we put our fully connected layer off here.
        self.log_writer.log(self.file_object,"Loading MobileNetV2 network...")
        basemodel = MobileNetV2(weights="imagenet",include_top=False,
                                input_tensor=Input(shape=(224,224,3)))

        # Now declare the head of the model that stays on the top of the basemodel.
        self.log_writer.log(self.file_object,"Declaring model...")
        headmodel=basemodel.output
        headmodel = AveragePooling2D(pool_size=(7,7))(headmodel)
        headmodel = Flatten(name="flatten")(headmodel)
        headmodel = Dense(128,activation='relu')(headmodel)
        headmodel = Dropout(0.5)(headmodel)
        headmodel = Dense(2,activation='softmax')(headmodel)

        # Place fullyconnected model on top of the base model
        self.log_writer.log(self.file_object,"Place fullyconnected model on top of the base model...")
        model =  Model(inputs=basemodel.input,outputs=headmodel)

        # Loop over all layers in the base model and freeze them so they will not be updated during the first training process
        for layer in basemodel.layers:
            layer.trainable=False
        
        # Setting up hyperparameters.
        lr=1e-4
        epochs = 20
        BS = 32
        self.log_writer.log(self.file_object,"Compiling our model...")
        optimizer=Adam(learning_rate=lr,decay=lr/epochs)
        model.compile(loss="binary_crossentropy",optimizer=optimizer,
                    metrics=["accuracy"])

        # Fitting our model
        self.log_writer.log(self.file_object,"Fitting our model...")
        best = model.fit(
            augment.flow(x_train, y_train, batch_size=BS),
            steps_per_epoch=len(x_train) // BS,
            validation_data=(X_test, y_test),
            validation_steps=len(X_test) // BS,
            epochs=epochs)
        
        return model, best