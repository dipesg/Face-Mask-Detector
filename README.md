# Face-Mask Detection :mask:

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/dipesh-silwal/)

<p align="middle">
    <img src="./images/1-with-mask.png" height=300 width=450>
    <img src="./images/2-with-mask.png" height=300 width=450>
    
### :woman_technologist: Introduction

In the COVID-19 crisis wearing masks is absolutely necessary for public health and in terms of controlling the spread of the pandemic. 
This project's aim is to develop a system that could detect masked and unmasked faces in images and real-time video. This can, for example, be used to alert people that do not wear a mask when using the public transport, airports or in a theatre.


### :raising_hand: Project Workflow 

Our pipeline consists of three steps:
  1. An AI model which detect all human faces in an image.
  2. An AI model which predict if that face wears mask/no_mask.
  3. The output is an annotated image with the prediction.
  
  
### 🚀 Model's performance

The face-mask model is trained with 900 images but in order to increase their volume it was used data augmentation and the weights of the MobileNetV2 model. More about this architecture can be found [here](https://arxiv.org/pdf/1801.04381.pdf). 

The facemask model has an accuracy of 99%.

![plot](https://user-images.githubusercontent.com/75604769/165533384-89e12b01-0be3-4c57-8cca-821f57c15cc5.png)

## :star: Streamlit app

Face Mask Detector with video using Tensorflow & Streamlit Webrtc

command
First clone the repository.
```
git clone https://github.com/dipesg/Face-Mask-Detector.git
```
Create new conda environment.
```
conda create -n venv python=3.8 -y
```
Install Necessary Requirements for the project.
```
pip install -r requirements.txt
```
Run following command to run the app.
```
streamlit run main.py 
```

**IMAGES**

## :warning: Technology Stack

- OpenCV
- Caffe-based face detector
- Keras
- TensorFlow
- MobileNetV2
- Streamlit & Streamlit Webrtc


## :open_file_folder: Folder Structure

``` 

|   LICENSE
|   main.py
|   Procfile
|   README.md
|   requirements.txt
|   Resultant.txt
|   runtime.txt
|   setup.sh
|   
+---face_detector
|       deploy.prototxt
|       haarcascade_frontalface_default.xml
|       res10_300x300_ssd_iter_140000.caffemodel
|       
+---logs
|       Detect_and_predict_mask.log
|       Detect_and_predict_mask.logs
|       Detect_from_video.log
|       Detect_from_video.logs
|       info.logs
|       prediction.logs
|       preprocessing.logs
|       train.logs
|       
+---mask_detector
|   |   keras_metadata.pb
|   |   mask_detector.model
|   |   saved_model.pb
|   |   
|   \---variables
|           variables.data-00000-of-00001
|           variables.index
|           
+---plot
|       plot.png
|       
\---src
        detect_and_predict_mask.py
        detect_from_video.py
        logger.py
        prediction.py
        preprocessing.py
        train_mask.py
        __init__.py

```


## :eyes: Next Steps

- Upload the streamlit ad-hoc app to Amazon Web Services. 
- Keep improving the performance of face and face-mask AI model. 
- Keep improving the detection of faces with low light and low quality pictures/webcams.


## :mailbox: Contact info

For questions, suggestions and other inquiries... ping me [here](dipeshsilwal31@gmail.com).
