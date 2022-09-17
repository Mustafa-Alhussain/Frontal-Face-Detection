from io import BytesIO
import ktrain
import os
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import gcsfs
from tensorflow.keras.models import load_model
from ktrain import load_predictor
from streamlit_option_menu import option_menu
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
from bokeh.models.widgets import Div
import av
import pickle
import time
from datetime import datetime

st.set_page_config(
    page_title = 'Realtime Face Detection',
    layout = 'wide', #centered
    initial_sidebar_state = 'auto', #collapsed, expanded
    menu_items={
        'Get Help': 'https://streamlit.io',
        'Report a bug': 'https://github.com',
        'About':'About your application: **Realtime Face Detection**'
    }
    )

# -------------General Setup------------------------------------------------

path1 = os.getcwd()

#Define Google Cloud Directory for models
filename_emotion = "gs://streamlit-face-detection/emotion_model.sav"
filename_gender = "gs://streamlit-face-detection/model_gen.sav"
filename_age = "gs://streamlit-face-detection/age_model.sav"

FS = gcsfs.GCSFileSystem(
    project= st.secrets["gcp_service_account"],
    token= st.secrets["gcp_service_account"])

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    with FS.open(model_path , "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model

# -------------Models loading------------------------------------------------

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


# -------------Header Section------------------------------------------------

st.image(os.path.join(path1, "HEADER_2.jpg"))
selected = option_menu(
    menu_title = None,
    options = ["Home", "Feedback" , "About" , "Contact Us"],
    icons=['house', 'chat-dots' , 'info-circle' ,'envelope'], menu_icon="cast",
           default_index = 0 , orientation = "horizontal")
choice = selected

# -------------Home Section------------------------------------------------

if choice == "Home":
    title = '<p style="text-align: center;font-size: 40px;font-weight: 550; "> Realtime Face Detection</p>'
    st.markdown(title, unsafe_allow_html=True)
    page = st.radio(
        label = "Choose Face Detection Mode",
        options=('Upload Image',  'Webcam Image Capture', 'Webcam Realtime'),
        index=0,
        horizontal = False)
    st.info("NOTE: quality of detection will depend on lights, Alignment & Distance to Camera.")
    st.warning("For your privacy, All Images processed online & not shared with us.")

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
                try:
                    num_faces = len(results.detections)
                except:
                    num_faces = 0
                if results.detections:
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
                        face_age, face_gender, face_emotion_pct = model_prediction(img , x ,y ,w, h)
                        face_age_background, face_age_text, emotion_text = create_age_text(img, face_age, face_gender, face_emotion_pct, x, y, w, h)
                imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(imageRGB)
                st.image(img, use_column_width=True)
                if num_faces > 1:
                    st.success("Total of " + str(num_faces) + " faces detected inside the image. Try adjusting your position for better detection if we missed anything.")
                elif num_faces == 1:
                    st.success(
                            "Only 1 face detected inside the image. Try adjusting minimum object size if we missed anything.")
                else:
                    st.warning("No Face Detected in Image. Make sure your face is visible in the camera with proper lighting.")
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
            try:
                num_faces = len(results.detections)
            except:
                num_faces = 0
            if results.detections:
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
                    face_age, face_gender, face_emotion_pct = model_prediction(img , x ,y ,w, h)
                    face_age_background, face_age_text, emotion_text = create_age_text(img, face_age, face_gender, face_emotion_pct, x, y, w, h)
            imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(imageRGB)
            st.image(img, use_column_width=True)
            if num_faces >= 1:
                st.success("Total of " + str(num_faces) + " faces detected inside the image. Try adjusting your position for better detection if we missed anything.")
            elif num_faces == 1:
                st.success(
                        "Only 1 face detected inside the image. Try adjusting minimum object size if we missed anything.")
            else:
                st.warning("No Face Detected in Image. Make sure your face is visible in the camera with proper lighting.")
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
# -------------Feedback Section------------------------------------------------

elif choice == "Feedback":

    st.components.v1.iframe(src = "https://docs.google.com/forms/d/e/1FAIpQLSclCA2nfKxPdfC2NXot2tcQnHpXsNNawu5lIfxxXoXeCn_tag/viewform?embedded=true", height = 1660, scrolling=False)

# -------------About Section------------------------------------------------

elif choice == "About":
    Project = "<html> " \
                      "<body><div> <b>About the Project:</b>" \
                      "</div></body></html>"
    Works = "<html> " \
                  "<body><div> <b>How it Works:</b>" \
                  "</div></body></html>"
    st.markdown(Project, unsafe_allow_html = True)
    st.markdown("This website to present you with our artificial intelligence and machine learning models that have capabilities of computer vision for image and video processing to get (faces detection, gender detection, age estimation and facial emotions detection)")
    st.markdown(Works, unsafe_allow_html = True)
    st.markdown(
        "A Convolutional Neural Network (CNN) is our main algorithm in this project. It mainly takes in an \
        input image, assign importance (learnable weights and biases) to various objects in the image.\
        The pre-processing part is less required in CNN compared to other classification algorithms.\
        While in primitive methods filters are hand-engineered, with enough training, CNN have the \
        ability to learn these filters/characteristics. In this project, there will be multiple models \
        used for prediction.")
    flow = st.image(os.path.join(path1, "Flow1.png"))
    Objectives = "<html> " \
                      "<body><div><b>Objectives:</b>" \
                      "<ul><li>Detect Faces</li><li>Estimate ages</li><li>Classify gender</li><li>Predict emotion state<li>Display the results</li></li></ul>" \
                      "</div></body></html>"
    st.markdown(Objectives, unsafe_allow_html=True)
    supported_modes = "<html> " \
                      "<body><div><b>Supported Face Detection Modes</b>" \
                      "<ul><li><b>Image Upload:</b> upload any image from your phone our computer to process the image.</li> \
                      <li><b>Webcam Image Capture:</b> use your camera to capture a live image to get processed through the models.</li> \
                      <li><b>Webcam Video Realtime:</b> by using your phone or computer camera to capture live video and process it in real time to see the actual results in the video</li></ul>" \
                      "</div></body></html>"
    st.markdown(supported_modes, unsafe_allow_html=True)

# -------------About Section------------------------------------------------

elif choice == "Contact Us":
    js = "window.open('https://linktr.ee/zeronex?utm_source=linktree_profile_share&ltsid=b4104f6e-d7b6-4602-8dc6-e0f416c54976')"  # New tab or window
    #js = "window.location.href = 'https://linktr.ee/zeronex?utm_source=linktree_profile_share&ltsid=b4104f6e-d7b6-4602-8dc6-e0f416c54976'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

# -------------Hide Streamlit Watermark------------------------------------------------

hide_footer_style = '''
<style>
.reportview-container .main footer {visibility: hidden;}
'''
st.markdown(hide_footer_style, unsafe_allow_html=True)


hide_menu_style = '''
<style>
#MainMenu {visibility: hidden;}
'''
st.markdown(hide_menu_style, unsafe_allow_html=True)