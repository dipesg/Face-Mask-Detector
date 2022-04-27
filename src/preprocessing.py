import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import logger


class Preprocess:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../logs/preprocessing.log", 'a+')
        def preprocess(self,content, categories):
            try:
                self.log_writer.log(self.file_object,"Starting data preprocessing...")
                data=[]
                labels=[]
                for category in categories:
                    path = os.path.join(content,category)
                    for img in os.listdir(path):
                        img_path=os.path.join(path,img)
                        image=load_img(img_path,target_size=(224,224))
                        image=img_to_array(image)
                        image=preprocess_input(image)

                        data.append(image)
                        labels.append(category)
                
                return data, labels
            except Exception as ex:
                self.log_writer.log(self.file_object, 'Error occured while running the preprocess function!! Error:: %s' % ex)
                raise ex

        def one_hot(self, labels):
            try:
                self.log_writer.log(self.file_object,"Performing one hot encoding...")
                lb = LabelBinarizer()
                label = lb.fit_transform(labels)
                label = to_categorical(label)
                
                return label
            except Exception as ex:
                self.log_writer.log(self.file_object, 'Error occured while running the one_hot function!! Error:: %s' % ex)
                raise ex

        # Changing to numpy array
        def change_to_numpy(self, data, label):
            try:
                self.log_writer.log(self.file_object,"Converting to numpy array...")
                data = np.array(data,dtype="float32")
                labels = np.array(label)
                data.shape
                
                return data, labels
            except Exception as ex:
                self.log_writer.log(self.file_object, 'Error occured while running the change_to_numpy function!! Error:: %s' % ex)
                raise ex
    
    