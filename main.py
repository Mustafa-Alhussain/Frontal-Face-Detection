#Emotion & Age & Gender detection using facial images

import re
from io import BytesIO
import ktrain
import os
from config.definitions import ROOT_DIR
import pandas as pd
import cv2
import sqlite3
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from ktrain import load_predictor
from tensorflow import keras
from streamlit_option_menu import option_menu
import tensorflow as tf
from keras.models import model_from_json, load_model
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import h5py
import av
import time
from datetime import datetime
path1 = os.getcwd()
path = os.path.join(path1 ,"models")
#ROOT_DIR
#Setup Models
st.cache(allow_output_mutation=True)

#load Age model

gender_json_file = open(os.path.join(path ,"model_gen.json"),'r')
loaded_gender_model_json = gender_json_file.read()
gender_json_file.close()
gender_model_weights = os.path.join(path ,"model_gen.h5")
gender_loaded_model = model_from_json(loaded_gender_model_json)
#load weights into gender model
gender_loaded_model.load_weights("model_gen.h5")
gender_loaded_model.compile(
    optimizer = 'adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],)

#load Age model
age_predictor = load_predictor(os.path.join(path))

#Load Emotion model
emotion_json_file = open(os.path.join(path ,"emotion_model.json"),'r')
loaded_emotion_model_json = emotion_json_file.read()
emotion_json_file.close()
emotion_loaded_model = model_from_json(loaded_emotion_model_json)
emotion_loaded_model.load_weights(os.path.join(path ,"emotion_weights.h5"))
emotion_loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
emotion_ranges = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise']
# Importing the Haar Cascades classifier XML file.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Defining a function to shrink the detected face region by a scale for better prediction in the model.
def shrink_face_roi(x, y, w, h, scale=0.9):
    wh_multiplier = (1-scale)/2
    x_new = int(x + (w * wh_multiplier))
    y_new = int(y + (h * wh_multiplier))
    w_new = int(w * scale)
    h_new = int(h * scale)
    return (x_new, y_new, w_new, h_new)

# Defining a function to create the predicted age overlay on the image by centering the text.
def create_age_text(img, text, pct_text, emotion_text ,  x, y, w, h):

    # Defining font, scales and thickness.
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.9
    yrsold_scale = 0.9
    pct_text_scale = 0.9

    # Getting width, height and baseline of age text and "years old".
    (text_width, text_height), text_bsln = cv2.getTextSize(text, fontFace=fontFace, fontScale=text_scale, thickness=2)
    (yrsold_width, yrsold_height), yrsold_bsln = cv2.getTextSize(emotion_text, fontFace=fontFace, fontScale=yrsold_scale, thickness=2)
    (pct_text_width, pct_text_height), pct_text_bsln = cv2.getTextSize(pct_text, fontFace=fontFace, fontScale=pct_text_scale, thickness=2)

    # Calculating center point coordinates of text background rectangle.
    x_center = x + (w/2)
    y_pct_text_center = y + h + 20
    y_text_center = y + h + 48
    y_yrsold_center = y + h + 75

    # Calculating bottom left corner coordinates of text based on text size and center point of background rectangle calculated above.
    x_text_org = int(round(x_center - (text_width / 2)))
    y_text_org = int(round(y_text_center + (text_height / 2)))
    x_yrsold_org = int(round(x_center - (yrsold_width / 2)))
    y_yrsold_org = int(round(y_yrsold_center + (yrsold_height / 2)))
    x_pct_text_org = int(round(x_center - (pct_text_width / 2)))
    y_pct_text_org = int(round(y_pct_text_center + (pct_text_height / 2)))

    face_age_background = cv2.rectangle(img, (x-1, y+h), (x+w+1, y+h+94), (0, 100, 0), cv2.FILLED)
    face_age_text = cv2.putText(img, text, org=(x_text_org, y_text_org), fontFace=fontFace, fontScale=text_scale, thickness=2, color=(255, 255, 255), lineType=cv2.LINE_AA)
    yrsold_text = cv2.putText(img, emotion_text, org=(x_yrsold_org, y_yrsold_org), fontFace=fontFace, fontScale=yrsold_scale, thickness=2, color=(255, 255, 255), lineType=cv2.LINE_AA)
    pct_age_text = cv2.putText(img, pct_text, org=(x_pct_text_org, y_pct_text_org), fontFace=fontFace, fontScale=pct_text_scale, thickness=2, color=(255, 255, 255), lineType=cv2.LINE_AA)

    return (face_age_background, face_age_text, yrsold_text)

# Defining a function to find faces in an image and then classify each found face into three models ranges defined above.
def classify_age(img):
    # Making a copy of the image for overlay of ages and making a grayscale copy for passing to the loaded model for age classification.
    img_copy = np.copy(img)
    img_copy2 = np.copy(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecting faces in the image using the face_cascade loaded above and storing their coordinates into a list.
    faces = face_cascade.detectMultiScale(img_copy,
                                          scaleFactor=1.3,
                                          minNeighbors=3,
                                          minSize=(30, 30))

    num_faces = len(faces)
    #print(f"{len(faces)} faces found.")

    # Looping through each face found in the image.
    for i, (x, y, w, h) in enumerate(faces):
        count = 0

        # Drawing a rectangle around the found face.
        face_rect = cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 100, 0), thickness=2)

        # Predicting the age of the found face using the model loaded above.
        x2, y2, w2, h2 = shrink_face_roi(x, y, w, h)
        face_roi = img_gray[y2:y2 + h2, x2:x2 + w2]
        face_roi = cv2.resize(face_roi, (256, 256))
        age1 = 'age' + str(i) + '.jpg'
        cv2.imwrite(age1, face_roi)
        # face_roi = face_roi.reshape(-1, 256, 256, 1)
        face_age = str(round(age_predictor.predict_filename(age1)[0]))

        # Gender prediction
        face_roi2 = img_copy2[y2:y2 + h2, x2:x2 + w2]
        face_roi2 = cv2.resize(face_roi2, (256, 256))
        # face_roi2 = face_roi2.reshape(-1, 256, 256, 3)
        gender1 = 'gender' + str(i) + '.jpg'
        cv2.imwrite(gender1, face_roi2)
        im = Image.open(gender1)
        ar = np.asarray(im)
        ar = ar.astype('float32')
        ar /= 255.0
        ar = ar.reshape(-1, 256, 256, 3)
        gender = np.round(gender_loaded_model.predict(ar))
        if gender == 0:
            gender = 'Male'
        else:
            gender = 'Female'
        face_age_pct = gender

        # Emotion Prediction
        face_roi3 = img_gray[y2:y2 + h2, x2:x2 + w2]
        face_roi3 = cv2.resize(face_roi3, (48, 48))
        emotion1 = 'emotion' + str(i) + '.jpg'
        cv2.imwrite(emotion1, face_roi3)
        im = Image.open(emotion1)
        ar = np.asarray(im)
        ar = ar.astype('float32')
        ar /= 255.0
        ar = ar.reshape(-1, 48, 48, 1)
        face_emotion_pct = emotion_ranges[np.argmax(emotion_loaded_model.predict(ar))]
        # Calling the above defined function to create the predicted age overlay on the image.
        face_age_background, face_age_text, yrsold_text = create_age_text(img_copy, face_age, face_age_pct,
                                                                          face_emotion_pct, x, y, w, h)
        try:
            os.remove(age1)
            os.remove(gender1)
            os.remove(emotion1)
        except:
            pass

    return img_copy , num_faces

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# -------------Edit------------------------------------------------
frames_per_seconds = 10
my_res = '720p'
# -------------Edit------------------------------------------------

STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),}

def get_dims(cap, res= my_res):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'MP4V'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'MP4V'),}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

conn = sqlite3.connect('feedback.db')
c = conn.cursor()
    
def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS feedback(date_submitted DATE, Q1 TEXT, Q2 INTEGER, Q3 INTEGER, Q4 TEXT, Q5 TEXT)')

def add_feedback(date_submitted, Q1, Q2, Q3, Q4, Q5):
    c.execute('INSERT INTO feedback (date_submitted,Q1, Q2, Q3, Q4, Q5) VALUES (?,?,?,?,?,?)',(date_submitted,Q1, Q2, Q3, Q4, Q5))
    conn.commit()

def main():
    # Set page configs. Get emoji names from WebFx
    st.set_page_config(page_title="Real-time Face Detection", page_icon="./assets/faceman_cropped.png", layout="centered")

# -------------Header Section------------------------------------------------

    title = '<p style="text-align: center;font-size: 40px;font-weight: 550; "> Realtime Face Detection</p>'
    st.markdown(title, unsafe_allow_html=True)

# -------------Sidebar Section------------------------------------------------

    with st.sidebar:
        st.image(os.path.join(path1, "side_image.jpeg"))
        selected = option_menu(None, ["Home", "Share your Feedback"] , icons=['house', 'search'], menu_icon="cast", default_index=1)
        #choice = st.sidebar.selectbox(label = " ",options = activities)
        choice = selected

# -------------Home Section------------------------------------------------

    if choice == "Home":
        st.markdown(
        "This website is designed to predict Age, Gender and Emotion using Realtime Face Detection"
        " Detection. Please share your experience with us in **Share your Feedback section** for improvement.")

        supported_modes = "<html> " \
                      "<body><div> <b>Supported Face Detection Modes (Change modes from sidebar menu)</b>" \
                      "<ul><li>Image Upload</li><li>Webcam Image Capture</li><li>Webcam Video Realtime</li></ul>" \
                      "</div></body></html>"
        st.markdown(supported_modes, unsafe_allow_html=True)
        st.warning("NOTE : Click the arrow icon at Top-Left to open Sidebar menu. ")
        with st.sidebar:
            page = st.radio("Choose Face Detection Mode", ('Upload Image',  'Webcam Image Capture', 'Webcam Realtime'), index=0)
            st.info("NOTE: quality of detection will depend on lights, Alignment & Distance to Camera.")

    # About the programmer
            st.markdown("## Made by **Mustafa Al Hussain**")
            ("## Email: **Mustafa.alhussain97@gmail.com**")
            ("[*Linkedin Page*](https://www.linkedin.com/in/mustafa-al-hussain-16026019a)")
            # line break

            st.markdown(" ")
            st.markdown(" ")
            st.markdown(" ")

# -------------Upload Image Section------------------------------------------------

        if page == "Upload Image":
        
            # You can specify more file types below if you want
            image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
            if image_file is not None:
                # Reading the image from filepath provided above and passing it through the age clasification method defined above.
                image = Image.open(image_file)
        
                if st.button("Process"):
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    age_img , num_faces = classify_age(img)
                    imageRGB = cv2.cvtColor(age_img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(imageRGB)
                    st.image(img, use_column_width=True)
                    if num_faces == 0:
                        st.warning("No Face Detected in Image. Make sure your face is visible in the camera with proper lighting.")
                    elif num_faces == 1:
                        st.success(
                            "Only 1 face detected inside the image. Try adjusting minimum object size if we missed anything.")
                    else:
                        st.success("Total of " + str(num_faces) + " faces detected inside the image. Try adjusting your position for better detection if we missed anything.")
                    img = np.array(img)
                    img = Image.fromarray(img)
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    # Creating columns to center button
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pass
                    with col3:
                        pass
                    with col2:
                        st.download_button(
                            label="Download image",
                            data=buffered.getvalue(),
                            file_name="output.png",
                            mime="image/png")

# -------------Webcam Image Capture Section------------------------------------------------

        if page == "Webcam Image Capture":
            st.info("NOTE : In order to use this mode, you need to give webcam access.")
            img_file_buffer = st.camera_input("Capture an Image from Webcam", disabled=False, key=1,help="Make sure you have given webcam permission to the site")

            if img_file_buffer is not None:
                image = Image.open(img_file_buffer)
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                age_img , num_faces = classify_age(img)
                imageRGB = cv2.cvtColor(age_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(imageRGB)
                st.image(img, use_column_width=True)
                if num_faces == 0:
                    st.warning("No Face Detected in Image. Make sure your face is visible in the camera with proper lighting.")
                elif num_faces == 1:
                    st.success(
                        "Only 1 face detected inside the image. Try adjusting minimum object size if we missed anything.")
                else:
                    st.success("Total of " + str(num_faces) + " faces detected inside the image. Try adjusting your position for better detection if we missed anything.")
                img = np.array(img)
                img = Image.fromarray(img)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                # Creating columns to center button
                col1, col2, col3 = st.columns(3)
                with col1:
                    pass
                with col3:
                    pass
                with col2:
                    st.download_button(
                        label="Download image",
                        data=buffered.getvalue(),
                        file_name="output.png",
                        mime="image/png")

# -------------Webcam Realtime Section------------------------------------------------


        if page == "Webcam Realtime":
            st.warning("NOTE : In order to use this mode, you need to give webcam access. "
               "After clicking 'Start' , it takes about 10-20 seconds to ready the webcam.")

            spinner_message = "Wait a sec, getting some things done..."

            with st.spinner(spinner_message):

                class VideoProcessor:

                    def recv(self, frame):
                        # convert to numpy array
                        frame = frame.to_ndarray(format="bgr24")
                        age_frame , num_faces = classify_age(frame)
                        frame = av.VideoFrame.from_ndarray(age_frame, format="bgr24")
                        if num_faces == 0:
                            st.warning("No Face Detected in Image. Make sure your face is visible in the camera with proper lighting.")
                        elif num_faces == 1:
                            st.success(
                                "Only 1 face detected inside the image. Try adjusting minimum object size if we missed anything.")
                        else:
                            st.success("Total of " + str(num_faces) + " faces detected inside the image. Try adjusting your position for better detection if we missed anything.")

                        return frame

            webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                            rtc_configuration=RTCConfiguration(
                                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))

# -------------Share your Feedback Section------------------------------------------------

    elif choice == "Share your Feedback":
        st.title("Feedback")

        d = st.date_input("Today's date",None, None, None, None)
        
        question_1 = st.selectbox('Which mode have you tried?',('' ,'Upload Image', 'Webcam Image Capture', 'Webcam Realtime' ,'All'))
        st.write('You selected:', question_1)
        
        question_2 = st.slider('How was the overall experience? (10 being very good and 1 being very dissapointed) ', 1, 1,10)
        st.write('You selected:', question_2) 

        question_3 = st.selectbox('Was the website fun and interactive?',('','Yes', 'No'))
        st.write('You selected:', question_3)

        question_4 = st.selectbox('Do you have a similar experience of what you tried?',('','Yes', 'No'))
        st.write('You selected:', question_4)

        question_5 = st.text_input('What could have been better?', max_chars=200)

        if st.button("Submit feedback"):
            create_table()
            add_feedback(d, question_1, question_2, question_3, question_4, question_5)
            st.success("Feedback submitted")
        # lines I added to display your table
        

        #query = pd.read_sql_query('''
        #select * from feedback''', conn)
        #data = pd.DataFrame(query)
        #st.write(data)


    # About the programmer
        with st.sidebar:
            st.markdown("## Made by **Mustafa Al Hussain**")
            ("## Email: **Mustafa.alhussain97@gmail.com**")
            ("[**Linkedin Page**](https://www.linkedin.com/in/mustafa-al-hussain-16026019a)")

# -------------About - Contact us Section------------------------------------------------

    #else:
    #    pass

# -------------Hide Streamlit Watermark------------------------------------------------

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()