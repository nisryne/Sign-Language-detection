import streamlit as st
import cv2
import numpy as np
import pickle
import mediapipe as mp
import speech_recognition as sr
import os
from PIL import Image
import time

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        background-color: LightCyan;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        background-color: LightCyan;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<h1 style="color: teal;">Sign Language Detection</h1>', unsafe_allow_html=True)

@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:

        r = height / float(h)
        dim = (int(w * r), height)


    else:

        r = width / float(w)
        dim = (width, int(h * r))


    resized = cv2.resize(image, dim, interpolation=inter)


    return resized


app_mode = st.sidebar.selectbox('Menu',
['About App','Sign Language to Text','Speech to sign Language']
)
if app_mode =='About App':
    
    st.markdown('<h1 style="color: teal;">Our App Sign Language Detection</h1>', unsafe_allow_html=True)

    st.image("https://img.freepik.com/free-vector/sign-language-alphabet-hand-drawn-style_23-2147872270.jpg?w=360", caption="Image Caption",use_column_width=True)
    
    st.markdown('<span style="font-size: 24px; color: teal; font-weight: bold;">Welcome to our application NOC</span>', unsafe_allow_html=True)

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    
    st.markdown(''' Our Sign Language Detection Web Application is an innovative tool designed to bridge the communication gap between individuals who use sign language and those who may not understand it. The application utilizes advanced machine learning techniques to recognize and interpret sign language gestures in real-time, providing an interactive and inclusive experience.''')
elif app_mode == 'Sign Language to Text':
    st.title('Sign Language to Text')    
 # Chargement du modèle
    model_dict = pickle.load(open('model.p', 'rb'))
    model = model_dict['model']

    # Chargement des étiquettes
    labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: '1', 27: '2', 28: '3', 29: '4',
    30: '5', 31: '6', 32: '7', 33: '8', 34: '9'
}


    # Fonction d'inférence
    def predict_character(frame):
        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        return frame

    # Configuration Mediapipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    # Début de l'application Streamlit
    st.title("Application de classification de caractères")

    # Configuration de la caméra
    cap = cv2.VideoCapture(0)

    # Configuration du lecteur vidéo
    video_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        processed_frame = predict_character(frame)

        # Affichage du cadre vidéo
        video_placeholder.image(processed_frame, channels="BGR")

        # Arrêter la boucle si l'utilisateur appuie sur la touche "Esc"
        if cv2.waitKey(1) == 27:
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
else:
    st.title('Speech to Sign Language')
    # initialize the speech recognition engine
    # initialize the speech recognition engine
    r = sr.Recognizer()


    # define function to display sign language images
    def display_images(text):
        # get the file path of the data directory
        data_dir = "data/"

        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display sign language images
        for char in text:
            if char.isalpha():
                # display sign language image for the alphabet
                img_path = os.path.join(data_dir, char, "0.jpg")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(2)

                # remove the image
                image_pos.empty()
            elif char == ' ':
                # display space image for space character
                img_path = os.path.join(data_dir, "space", "1.jpg")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(2)

                # remove the image
                image_pos.empty()

        # wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()


    # add start button to start recording audio
    if st.button("Start Talking"):
        # record audio for 5 seconds
        with sr.Microphone() as source:
            st.write("Say something!")

            audio = r.listen(source, phrase_time_limit=5)
            text = ""
            try:
                text = r.recognize_google(audio)
            except sr.UnknownValueError:
                st.write("Sorry, I did not understand what you said.")
            except sr.RequestError as e:
                st.write(f"Could not request results from Google Speech Recognition service; {e}")

        # convert text to lowercase
        text = text.lower()
        # display the final result
        st.write(f"You said: {text}", font_size=41)

        # display sign language images
        display_images(text)

