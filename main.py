from io import BytesIO
import ktrain
import os
import pandas as pd
import cv2
import s3fs
import sqlite3
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from ktrain import load_predictor
from streamlit_option_menu import option_menu
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
import av
import pickle
import time
from datetime import datetime

# -------------General Setup------------------------------------------------

path1 = os.getcwd()

# Set page configs. Get emoji names from WebFx
st.set_page_config(page_title="Real-time Face Detection", page_icon="./assets/faceman_cropped.png", layout="centered")


fs = s3fs.S3FileSystem(anon=False)
#Define Directory for models
filename_gender = "frontal-face-detection/model_gen.sav"
filename_emotion = "frontal-face-detection/emotion_model.sav"
filename_age = "frontal-face-detection/age_model.sav"

@st.cache(allow_output_mutation=True)
def load_model(model_name):
    with fs.open(model_name , "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model

# -------------models loading------------------------------------------------

gender_loaded_model = load_model(filename_gender)
gender_loaded_model.compile(
    optimizer = 'adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],)
print("Loaded Gender Model from disk")
emotion_loaded_model = load_model(filename_emotion)
emotion_loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
print("Loaded Emotion Model from disk")
age_predictor = load_model(filename_age)
print("Loaded Age Model from disk")

emotion_ranges = ['Angry', 'Disgust', 'Fear', 'Happy', 'Normal', 'Sad', 'Suprise']
gender_ranges = ['Male' , 'Female']

# Importing the Face Detection classifier.
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)
# -------------Functions to Detect Faces & create texts------------------------------------------------



# Defining a function to create the predicted age overlay on the image by centering the text.

#@st.cache(ttl=600)
def create_age_text(img, age_text, gender_text, emotion_text ,  x, y, w, h):

    # Defining font, scales and thickness.
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    age_scale = 0.8
    emotion_scale = 0.8
    gender_text_scale = 0.8

    # Getting width, height and baseline of age text and emotion text and gender text
    (age_width, age_height), age_bsln = cv2.getTextSize(age_text, fontFace=fontFace, fontScale=age_scale, thickness=1)
    (emotion_width, emotion_height), emotion_bsln = cv2.getTextSize(emotion_text, fontFace=fontFace, fontScale=emotion_scale, thickness=1)
    (gender_text_width, gender_text_height), gender_text_bsln = cv2.getTextSize(gender_text, fontFace=fontFace, fontScale=gender_text_scale, thickness=1)

    # Calculating center point coordinates of text background rectangle.
    x_center = x + (w/2)
    y_gender_text_center = y + h + 20
    y_age_center = y + h + 48
    y_emotion_center = y + h + 75

    # Calculating bottom left corner coordinates of text based on text size and center point of background rectangle calculated above.
    x_age_org = int(round(x_center - (age_width / 2)))
    y_age_org = int(round(y_age_center + (age_height / 2)))
    x_emotion_org = int(round(x_center - (emotion_width / 2)))
    y_emotion_org = int(round(y_emotion_center + (emotion_height / 2)))
    x_gender_text_org = int(round(x_center - (gender_text_width / 2)))
    y_gender_text_org = int(round(y_gender_text_center + (gender_text_height / 2)))

    face_age_background = cv2.rectangle(img, (x-1, y+h), (x+w+1, y+h+94), (0, 150, 0), cv2.FILLED)
    face_age_text = cv2.putText(img, age_text, org=(x_age_org, y_age_org), fontFace=fontFace, fontScale=age_scale, thickness=2, color=(255, 255, 255), lineType=cv2.LINE_AA)
    emotion_text = cv2.putText(img, emotion_text, org=(x_emotion_org, y_emotion_org), fontFace=fontFace, fontScale=emotion_scale, thickness=2, color=(255, 255, 255), lineType=cv2.LINE_AA)
    pct_age_text = cv2.putText(img, gender_text, org=(x_gender_text_org, y_gender_text_org), fontFace=fontFace, fontScale=gender_text_scale, thickness=2, color=(255, 255, 255), lineType=cv2.LINE_AA)

    return (face_age_background, face_age_text, emotion_text)

# Defining a function to find faces in an image and then classify each found face into three models ranges defined above.
@st.cache(ttl=600)
def model_prediction(img , x , y , w , h):
    # Making a copy of the image for overlay of ages and making a grayscale copy for passing to the loaded model for age classification.
    img_copy = np.copy(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_roi = img_gray[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (256, 256))
    age1 = 'age' + str(id) +'.jpg'
    cv2.imwrite(age1, face_roi)
    face_age = str(round(age_predictor.predict_filename(age1)[0]))
    
    #Gender prediction
    face_roi2 = img_copy[y:y+h, x:x+w]
    face_roi2 = cv2.resize(face_roi2, (256, 256))
    gender1 = 'gender' + str(id) +'.jpg'
    cv2.imwrite(gender1, face_roi2)
    im = Image.open(gender1)
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 256, 256, 3)
    face_gender = gender_ranges[np.argmax(gender_loaded_model.predict(ar))]
    
    #Emotion Prediction
    face_roi3 = img_gray[y:y+h, x:x+w]
    face_roi3 = cv2.resize(face_roi3, (48, 48))
    emotion1 = 'emotion' + str(id) +'.jpg'
    cv2.imwrite(emotion1, face_roi3)
    im = Image.open(emotion1)
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 48, 48, 1)
    face_emotion_pct = emotion_ranges[np.argmax(emotion_loaded_model.predict(ar))]
    try:
        os.remove(age1)
        os.remove(gender1)
        os.remove(emotion1)
    except:
        pass

    return face_age, face_gender, face_emotion_pct

conn = sqlite3.connect('feedback.db')
c = conn.cursor()


def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS feedback(date_submitted DATE, Q1 TEXT, Q2 INTEGER, Q3 INTEGER, Q4 TEXT, Q5 TEXT)')


def add_feedback(date_submitted, Q1, Q2, Q3, Q4, Q5):
    c.execute('INSERT INTO feedback (date_submitted,Q1, Q2, Q3, Q4, Q5) VALUES (?,?,?,?,?,?)',(date_submitted,Q1, Q2, Q3, Q4, Q5))
    conn.commit()

# -------------Header Section------------------------------------------------
title = '<p style="text-align: center;font-size: 40px;font-weight: 550; "> Realtime Face Detection</p>'
st.markdown(title, unsafe_allow_html=True)

# -------------Sidebar Section------------------------------------------------

with st.sidebar:
    st.image(os.path.join(path1, "side_image.jpeg"))
    selected = option_menu(None, ["Home", "Share your Feedback"] , icons=['house', 'search'], menu_icon="cast")
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
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = faceDetection.process(imgRGB)
                if results.detections:
                    num_faces = len(results.detections)
                    for id, detection in enumerate(results.detections):
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, ic = img.shape
                        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                               int(bboxC.width * iw), int(bboxC.height * ih)
                        x = int(bboxC.xmin * iw)
                        y = int(bboxC.ymin * ih)
                        w = int(bboxC.width * iw)
                        h = int(bboxC.height * ih)
                        cv2.rectangle(img, bbox, (0, 150, 0), 2)
                        try:
                            face_age, face_gender, face_emotion_pct = model_prediction(img , x ,y ,w, h)
                        except:
                            pass
                        face_age_background, face_age_text, emotion_text = create_age_text(img, face_age, face_gender, face_emotion_pct, x, y, w, h)
                imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = faceDetection.process(imgRGB)
            if results.detections:
                try:
                    num_faces = len(results.detections)
                except:
                    num_faces = 0
                    pass
                for id, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = img.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                           int(bboxC.width * iw), int(bboxC.height * ih)
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    cv2.rectangle(img, bbox, (0, 150, 0), 2)
                    try:
                        face_age, face_gender, face_emotion_pct = model_prediction(img , x ,y ,w, h)
                    except:
                        pass
                    face_age_background, face_age_text, emotion_text = create_age_text(img, face_age, face_gender, face_emotion_pct, x, y, w, h)
            imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
            RTC_CONFIGURATION = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            pTime = 0
            process_time = 0
            x_lst = []
            y_lst = []
            h_lst = []
            w_lst = []
            age_lst = []
            gender_lst = []
            emotion_lst = []
            class VideoProcessor:
                def recv(self, frame):
                    global pTime, process_time, x_lst,y_lst, h_lst , w_lst,age_lst,gender_lst,emotion_lst
                    # convert to numpy array
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.flip(img, 1)
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = faceDetection.process(imgRGB)
                    if results.detections:
                        num_faces = len(results.detections)
                        for id, detection in enumerate(results.detections):
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, ic = img.shape
                            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                   int(bboxC.width * iw), int(bboxC.height * ih)
                            x_lst.append(int(bboxC.xmin * iw))
                            y_lst.append(int(bboxC.ymin * ih))
                            w_lst.append(int(bboxC.width * iw))
                            h_lst.append(int(bboxC.height * ih))
                            cv2.rectangle(img, bbox, (0, 150, 0), 2)
                            if process_time % 30 == 0 and process_time != 0:
                                try:
                                    face_age, face_gender, face_emotion_pct = model_prediction(img , x_lst[id], y_lst[id], w_lst[id], h_lst[id])
                                    age_lst.append(face_age)
                                    gender_lst.append(face_gender)
                                    emotion_lst.append(face_emotion_pct)
                                except:
                                    pass
                        for i in range(len(x_lst)):
                            try:
                                face_age_background, face_age_text, emotion_text = create_age_text(img, age_lst[i], gender_lst[i], emotion_lst[i], x_lst[i], y_lst[i], w_lst[i], h_lst[i])
                            except:
                                pass
                    x_lst.clear()
                    y_lst.clear()
                    w_lst.clear()
                    h_lst.clear()
                    process_time += 1
                    if process_time % 30 == 0:
                        age_lst.clear()
                        gender_lst.clear()
                        emotion_lst.clear()
                    cTime = time.time()
                    time_difference = cTime - pTime
                    fps = 1 / time_difference
                    pTime = cTime
                    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                                3, (0, 150, 0), 3)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_ctx = webrtc_streamer(
            key="WYH",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=VideoProcessor,
            async_processing=True,
            )
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

    # About the programmer
    with st.sidebar:
        st.markdown("## Made by **Mustafa Al Hussain**")
        ("## Email: **Mustafa.alhussain97@gmail.com**")
        ("[**Linkedin Page**](https://www.linkedin.com/in/mustafa-al-hussain-16026019a)")

# -------------Hide Streamlit Watermark------------------------------------------------

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)