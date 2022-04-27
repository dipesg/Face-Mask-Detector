from tensorflow.keras.models import load_model
from src.detect_and_predict_mask import Mask
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import streamlit as st

try:
    prototxtpath = "./face_detector/deploy.prototxt"
    weightspath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtpath, weightspath)
    maskNet = load_model("./mask_detector/mask_detector.model")

except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        print('getting frame')
        img = frame.to_ndarray(format="bgr24")

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = Mask().detect_and_predict_mask(img, faceNet, maskNet)

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
            cv2.putText(img, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            print('finishing process')

        # show the output frame
        #cv2.imshow("Frame", frame)
        return img

st.header("Webcam Live Feed")
st.write("Click on start to use webcam and detect your face emotion")
html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Mask detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
st.markdown(html_temp_home1, unsafe_allow_html=True)
st.write("""
            Real time face mask detection using web cam feed.
            """)
st.sidebar.markdown(
        """ 
        Developed by Dipesh Silwal
         
        You can follow or connect with me on Github and LinkedIN.
        
        Github: https://github.com/dipesg   
        Email : dipeshsilwal31@gmail.com 
        LinkedIN:https://www.linkedin.com/in/dipesh-silwal
            
        """)
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)