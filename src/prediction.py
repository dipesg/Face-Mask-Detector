#importing custom module
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import logger
from preprocessing import Preprocess
from train_mask import model_building

class Prediction:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../logs/prediction.log", 'a+')
        
    def pred(self):
        try:
            # Preprocess data
            self.log_writer.log(self.file_object,"Inside prediction preprocess data begins...")
            data, labels = Preprocess().preprocess("../dataset", ["with_mask", "without_mask"])

            # Doing one hot encoding
            self.log_writer.log(self.file_object,"Inside prediction one hot encoding begins...")
            label = Preprocess().one_hot(labels)

            # Changing to numpy array
            self.log_writer.log(self.file_object,"Inside prediction changing to numpy array...")
            data, labels = Preprocess().change_to_numpy(data, label)


            # Now train_test_split
            self.log_writer.log(self.file_object,"Performing train_test_split inside prediction...")
            x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.20,
                                                            stratify=labels,random_state=42)

            # Importing model
            models, best = model_building()

            # Predicting
            self.log_writer.log(self.file_object,"Now predicting...")
            predict = models.predict(x_test, batch_size=32)
            predict = np.argmax(predict, axis=1)
            print(predict)
            
            # Saving our model
            self.log_writer.log(self.file_object,"Saving our model...")
            models.save("mask_detector")
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the pred function!! Error:: %s' % ex)
            raise ex
        
    def visualize(self):
        try:
            # Visualizing the training loss and accuracy of our model.
            self.log_writer.log(self.file_object,"Visualizing the training loss and accuracy of our model...")
            # Importing model
            #logger.info("Calling model from train_mask.py...")
            models, best = model_building()
            N = 20
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(np.arange(0, N), best.history["loss"], label="train_loss")
            plt.plot(np.arange(0, N), best.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, N), best.history["accuracy"], label="train_acc")
            plt.plot(np.arange(0, N), best.history["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("#Epoch")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            plt.savefig("plot.png")
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the visualize function!! Error:: %s' % ex)
            raise ex
        
if __name__ == "__main__":
    Prediction().pred()
    
