from tensorflow.keras.models import load_model
from detect_and_predict_mask import Mask
import imutils
import numpy as np
import time
import cv2
import logger

class Videopred:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../logs/Detect_from_video.log", 'a+')
    def detect_video(self):
        self.log_writer.log(self.file_object,"Loading required files for execution....")
        prototxtpath = "../face_detector/deploy.prototxt"
        weightspath = "../face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtpath, weightspath)

        self.log_writer.log(self.file_object,"Loading mask_detector.model...")
        maskNet = load_model("../mask_detector/mask_detector.model")

        self.log_writer.log(self.file_object,"Starting video stream...")
        vs = cv2.VideoCapture(0)
        time.sleep(2.0)

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            _, frame = vs.read()
            frame = imutils.resize(frame, width=400)

            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = Mask().detect_and_predict_mask(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        vs.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    Videopred().detect_video()