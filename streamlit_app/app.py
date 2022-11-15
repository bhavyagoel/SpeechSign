import queue
from typing import List, NamedTuple

import av
import cv2
import firebase_admin
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import whisper
from audiorecorder import audiorecorder
from firebase_admin import credentials, firestore
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
import queue
import os 
import glob
import requests
import json
import time
# np.set_printoptions(suppress=True)


def firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(
            'static/speechsign-23477-8f5b84f0980a.json')
        app = firebase_admin.initialize_app(cred)
    else:
        app = firebase_admin.get_app()
    db = firestore.client()
    return app, db


def cache_query_param():
    try:
        query_param = st.experimental_get_query_params()
        user_id = query_param['user'][0]
        st.session_state['key'] = user_id
    except Exception as e:
        st.error("Please enter the user id, or try logging in from the home page")
        user_id = st.text_input("Enter your user id", key="user_id")
        st.session_state['key'] = user_id
        if user_id:
            st.experimental_set_query_params(user=user_id)
            st.experimental_rerun()


def extract_feature(image):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(static_image_mode=False, model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while True:
            results = hands.process(
                cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
            image_height, image_width, _ = image.shape
            # Print handedness (left v.s. right hand).
            # Caution : Uncomment these print command will resulting long log of mediapipe log
            #print(f'Handedness of {input_image}:')
            # print(results.multi_handedness)

            # Draw hand landmarks of each hand.
            # Caution : Uncomment these print command will resulting long log of mediapipe log
            #print(f'Hand landmarks of {input_image}:')
            if not results.multi_hand_landmarks:
                # Here we will set whole landmarks into zero as no handpose detected
                # in a picture wanted to extract.

                # Wrist Hand
                wristX = 0
                wristY = 0
                wristZ = 0

                # Thumb Finger
                thumb_CmcX = 0
                thumb_CmcY = 0
                thumb_CmcZ = 0

                thumb_McpX = 0
                thumb_McpY = 0
                thumb_McpZ = 0

                thumb_IpX = 0
                thumb_IpY = 0
                thumb_IpZ = 0

                thumb_TipX = 0
                thumb_TipY = 0
                thumb_TipZ = 0

                # Index Finger
                index_McpX = 0
                index_McpY = 0
                index_McpZ = 0

                index_PipX = 0
                index_PipY = 0
                index_PipZ = 0

                index_DipX = 0
                index_DipY = 0
                index_DipZ = 0

                index_TipX = 0
                index_TipY = 0
                index_TipZ = 0

                # Middle Finger
                middle_McpX = 0
                middle_McpY = 0
                middle_McpZ = 0

                middle_PipX = 0
                middle_PipY = 0
                middle_PipZ = 0

                middle_DipX = 0
                middle_DipY = 0
                middle_DipZ = 0

                middle_TipX = 0
                middle_TipY = 0
                middle_TipZ = 0

                # Ring Finger
                ring_McpX = 0
                ring_McpY = 0
                ring_McpZ = 0

                ring_PipX = 0
                ring_PipY = 0
                ring_PipZ = 0

                ring_DipX = 0
                ring_DipY = 0
                ring_DipZ = 0

                ring_TipX = 0
                ring_TipY = 0
                ring_TipZ = 0

                # Pinky Finger
                pinky_McpX = 0
                pinky_McpY = 0
                pinky_McpZ = 0

                pinky_PipX = 0
                pinky_PipY = 0
                pinky_PipZ = 0

                pinky_DipX = 0
                pinky_DipY = 0
                pinky_DipZ = 0

                pinky_TipX = 0
                pinky_TipY = 0
                pinky_TipZ = 0

                # Return Whole Landmark and Image
                return (wristX, wristY, wristZ,
                        thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                        thumb_McpX, thumb_McpY, thumb_McpZ,
                        thumb_IpX, thumb_IpY, thumb_IpZ,
                        thumb_TipX, thumb_TipY, thumb_TipZ,
                        index_McpX, index_McpY, index_McpZ,
                        index_PipX, index_PipY, index_PipZ,
                        index_DipX, index_DipY, index_DipZ,
                        index_TipX, index_TipY, index_TipZ,
                        middle_McpX, middle_McpY, middle_McpZ,
                        middle_PipX, middle_PipY, middle_PipZ,
                        middle_DipX, middle_DipY, middle_DipZ,
                        middle_TipX, middle_TipY, middle_TipZ,
                        ring_McpX, ring_McpY, ring_McpZ,
                        ring_PipX, ring_PipY, ring_PipZ,
                        ring_DipX, ring_DipY, ring_DipZ,
                        ring_TipX, ring_TipY, ring_TipZ,
                        pinky_McpX, pinky_McpY, pinky_McpZ,
                        pinky_PipX, pinky_PipY, pinky_PipZ,
                        pinky_DipX, pinky_DipY, pinky_DipZ,
                        pinky_TipX, pinky_TipY, pinky_TipZ,
                        cv2.flip(image, 1))

            annotated_image = cv2.flip(image.copy(), 1)
            for hand_landmarks in results.multi_hand_landmarks:
                # Wrist Hand /  Pergelangan Tangan
                wristX = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
                wristY = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height
                wristZ = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

                # Thumb Finger / Ibu Jari
                thumb_CmcX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width
                thumb_CmcY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height
                thumb_CmcZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z

                thumb_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width
                thumb_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height
                thumb_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z

                thumb_IpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width
                thumb_IpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height
                thumb_IpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z

                thumb_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
                thumb_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
                thumb_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z

                # Index Finger / Jari Telunjuk
                index_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width
                index_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height
                index_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z

                index_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width
                index_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height
                index_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z

                index_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width
                index_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height
                index_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z

                index_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                index_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
                index_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

                # Middle Finger / Jari Tengah
                middle_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width
                middle_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height
                middle_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z

                middle_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width
                middle_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height
                middle_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z

                middle_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width
                middle_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height
                middle_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z

                middle_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
                middle_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height
                middle_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z

                # Ring Finger / Jari Cincin
                ring_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width
                ring_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height
                ring_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z

                ring_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width
                ring_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height
                ring_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z

                ring_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width
                ring_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height
                ring_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z

                ring_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width
                ring_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height
                ring_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z

                # Pinky Finger / Jari Kelingking
                pinky_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width
                pinky_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height
                pinky_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z

                pinky_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width
                pinky_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height
                pinky_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z

                pinky_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width
                pinky_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height
                pinky_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z

                pinky_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width
                pinky_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height
                pinky_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z

                # Draw the Skeleton
                mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # draw bounding box
                bounding_box = {}
                bounding_box['x_min'] = int(min(wristX, thumb_CmcX, thumb_IpX, thumb_TipX, index_McpX, index_PipX, index_DipX, index_TipX, middle_McpX,
                                            middle_PipX, middle_DipX, middle_TipX, ring_McpX, ring_PipX, ring_DipX, ring_TipX, pinky_McpX, pinky_PipX, pinky_DipX, pinky_TipX))
                bounding_box['y_min'] = int(min(wristY, thumb_CmcY, thumb_IpY, thumb_TipY, index_McpY, index_PipY, index_DipY, index_TipY, middle_McpY,
                                            middle_PipY, middle_DipY, middle_TipY, ring_McpY, ring_PipY, ring_DipY, ring_TipY, pinky_McpY, pinky_PipY, pinky_DipY, pinky_TipY))
                bounding_box['x_max'] = int(max(wristX, thumb_CmcX, thumb_IpX, thumb_TipX, index_McpX, index_PipX, index_DipX, index_TipX, middle_McpX,
                                            middle_PipX, middle_DipX, middle_TipX, ring_McpX, ring_PipX, ring_DipX, ring_TipX, pinky_McpX, pinky_PipX, pinky_DipX, pinky_TipX))
                bounding_box['y_max'] = int(max(wristY, thumb_CmcY, thumb_IpY, thumb_TipY, index_McpY, index_PipY, index_DipY, index_TipY, middle_McpY,
                                            middle_PipY, middle_DipY, middle_TipY, ring_McpY, ring_PipY, ring_DipY, ring_TipY, pinky_McpY, pinky_PipY, pinky_DipY, pinky_TipY))

                cv2.rectangle(annotated_image, (bounding_box['x_min'], bounding_box['y_min']), (
                    bounding_box['x_max'], bounding_box['y_max']), (0, 255, 0), 2)

            return (wristX, wristY, wristZ,
                    thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                    thumb_McpX, thumb_McpY, thumb_McpZ,
                    thumb_IpX, thumb_IpY, thumb_IpZ,
                    thumb_TipX, thumb_TipY, thumb_TipZ,
                    index_McpX, index_McpY, index_McpZ,
                    index_PipX, index_PipY, index_PipZ,
                    index_DipX, index_DipY, index_DipZ,
                    index_TipX, index_TipY, index_TipZ,
                    middle_McpX, middle_McpY, middle_McpZ,
                    middle_PipX, middle_PipY, middle_PipZ,
                    middle_DipX, middle_DipY, middle_DipZ,
                    middle_TipX, middle_TipY, middle_TipZ,
                    ring_McpX, ring_McpY, ring_McpZ,
                    ring_PipX, ring_PipY, ring_PipZ,
                    ring_DipX, ring_DipY, ring_DipZ,
                    ring_TipX, ring_TipY, ring_TipZ,
                    pinky_McpX, pinky_McpY, pinky_McpZ,
                    pinky_PipX, pinky_PipY, pinky_PipZ,
                    pinky_DipX, pinky_DipY, pinky_DipZ,
                    pinky_TipX, pinky_TipY, pinky_TipZ,
                    annotated_image)


@st.cache(ttl=24*60*60, allow_output_mutation=True)
def load_model():
    num_classes = 26
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1,
                               padding="causal", activation="relu", input_shape=(63, 1)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=256, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(filters=256, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(rate=0.2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.load_weights('model/model_SIBI.h5')
    return model


def predict(frame, model):
    (wristX, wristY, wristZ,
     thumb_CmcX, thumb_CmcY, thumb_CmcZ,
     thumb_McpX, thumb_McpY, thumb_McpZ,
     thumb_IpX, thumb_IpY, thumb_IpZ,
     thumb_TipX, thumb_TipY, thumb_TipZ,
     index_McpX, index_McpY, index_McpZ,
     index_PipX, index_PipY, index_PipZ,
     index_DipX, index_DipY, index_DipZ,
     index_TipX, index_TipY, index_TipZ,
     middle_McpX, middle_McpY, middle_McpZ,
     middle_PipX, middle_PipY, middle_PipZ,
     middle_DipX, middle_DipY, middle_DipZ,
     middle_TipX, middle_TipY, middle_TipZ,
     ring_McpX, ring_McpY, ring_McpZ,
     ring_PipX, ring_PipY, ring_PipZ,
     ring_DipX, ring_DipY, ring_DipZ,
     ring_TipX, ring_TipY, ring_TipZ,
     pinky_McpX, pinky_McpY, pinky_McpZ,
     pinky_PipX, pinky_PipY, pinky_PipZ,
     pinky_DipX, pinky_DipY, pinky_DipZ,
     pinky_TipX, pinky_TipY, pinky_TipZ,
     output_IMG) = extract_feature(frame)

    input_IMG = np.array([[[wristX], [wristY], [wristZ],
                           [thumb_CmcX], [thumb_CmcY], [thumb_CmcZ],
                           [thumb_McpX], [thumb_McpY], [thumb_McpZ],
                           [thumb_IpX], [thumb_IpY], [thumb_IpZ],
                           [thumb_TipX], [thumb_TipY], [thumb_TipZ],
                           [index_McpX], [index_McpY], [index_McpZ],
                           [index_PipX], [index_PipY], [index_PipZ],
                           [index_DipX], [index_DipY], [index_DipZ],
                           [index_TipX], [index_TipY], [index_TipZ],
                           [middle_McpX], [middle_McpY], [middle_McpZ],
                           [middle_PipX], [middle_PipY], [middle_PipZ],
                           [middle_DipX], [middle_DipY], [middle_DipZ],
                           [middle_TipX], [middle_TipY], [middle_TipZ],
                           [ring_McpX], [ring_McpY], [ring_McpZ],
                           [ring_PipX], [ring_PipY], [ring_PipZ],
                           [ring_DipX], [ring_DipY], [ring_DipZ],
                           [ring_TipX], [ring_TipY], [ring_TipZ],
                           [pinky_McpX], [pinky_McpY], [pinky_McpZ],
                           [pinky_PipX], [pinky_PipY], [pinky_PipZ],
                           [pinky_DipX], [pinky_DipY], [pinky_DipZ],
                           [pinky_TipX], [pinky_TipY], [pinky_TipZ]]])

    predictions = model.predict(input_IMG)
    char = chr(np.argmax(predictions)+65)
    confidence = np.max(predictions)/np.sum(predictions)

    if confidence > 0.4:
        cv2.putText(output_IMG, char, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(output_IMG, str(confidence), (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return char, confidence, output_IMG

    return None, None, output_IMG


class Detection(NamedTuple):
    name: str
    prob: float


class VideoTransformer(VideoProcessorBase):

    result_queue: "queue.Queue[List[Detection]]"

    def __init__(self) -> None:
        self.threshold1 = 224
        self.result_queue = queue.Queue()
        self.data = np.ndarray(shape=(1, 240, 240, 3), dtype=np.float32)

    def _predict_image(self, image):
        result: List[Detection] = []
        model = load_model()
        label, confidence, output_img = predict(image, model)
        if label is not None:
            result.append(Detection(label, float(confidence)))
        return result, output_img

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame = frame
        result, output_img = self._predict_image(
            frame.to_ndarray(format="bgr24"))
        self.result_queue.put(result)

        return av.VideoFrame.from_ndarray(output_img, format="bgr24")


def sign_detection(db, user_id):
    st.image("static/sign.jpg")
    ctx = webrtc_streamer(
        key="SpeechSign",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [
            {
                "urls": "stun:openrelay.metered.ca:80",
            },
            {
                "urls": "turn:openrelay.metered.ca:80",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": "turn:openrelay.metered.ca:443",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": "turn:openrelay.metered.ca:443?transport=tcp",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoTransformer,
        async_processing=True,)

    if st.checkbox("Show the detected labels", value=True):
        if ctx.state.playing:
            labels_placeholder = st.empty()
            while True:
                if ctx.video_processor:
                    try:
                        result = ctx.video_processor.result_queue.get(
                            timeout=2
                        )
                        doc_ref = db.collection(u'users').document(user_id).collection(
                            'sign-detected')
                        # check if result is empty
                        if result:
                            doc_ref.add({result[0].name: result[0].prob})
                        # doc_ref.add({result[0].name: result[0].prob})
                    except queue.Empty:
                        result = None

                    labels_placeholder.table(result)
                else:
                    break


@st.cache(ttl=24*60*60, allow_output_mutation=True)
def audio_model():
    model = whisper.load_model("tiny.en")
    return model


def speech_detection():
    # app_sst()
    audio = audiorecorder("Click to record", "Recording...")

    if len(audio) > 0:
        st.audio(audio)

        # model = audio_model()

        # if not audio.mp3 exists then create it
        wav_file = open("audio_proc/audio.mp3", "wb")
        wav_file.write(audio.tobytes())

        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small.en"
        headers = {"Authorization": "Bearer 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"}

        def query(filename):
            with open(filename, "rb") as f:
                data = f.read()
            response = requests.request("POST", API_URL, headers=headers, data=data)
            return json.loads(response.content.decode("utf-8"))

        result = query("audio_proc/audio.mp3")
        placeholder = st.empty()

        while 'error' in result:
            placeholder.error("Model is still loading, please wait...")
            time.sleep(30)
            result = query("audio_proc/audio.mp3")
        
        placeholder.success("Model loaded successfully")
        text = result["text"]
        # result = model.transcribe("audio_proc/audio.mp3")
        # text = result["text"]

        
        if text != '':
            original_text = text
            text = text.upper()
            text = text.replace(' ', '')

            video_name = 'video_proc/{}.webm'.format(text)
            print(text[0])
            frame = cv2.imread("static/sign_alpha/"+text[0]+".jpg")
            height, width, layers = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'VP90')
            video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

            for i in text:
                if not i.isalpha():
                    continue
                video.write(cv2.imread("static/sign_alpha/"+i+".jpg"))

            cv2.destroyAllWindows()
            video.release()
            st.header(original_text)
            st.video("video_proc/{}.webm".format(text))
            
    files = glob.glob('video_proc/*')
    for f in files:
        os.remove(f)

    files = glob.glob('audio_proc/*')
    for f in files:
        os.remove(f)

    return 0


def show_database(db, user_id):
    doc_ref = db.collection(u'users').document(user_id)
    doc = doc_ref.get()
    user_det = doc.to_dict()
    st.header("User Details")
    user_df = pd.DataFrame({"Type": ['Name', 'DOB', 'Email'], "Value": [
                           user_det['name'], user_det['dob'], user_det['email']]})
    st.dataframe(user_df)

    st.header('Sign Detected')
    sign_df = pd.DataFrame(columns=['Alphabet', 'Confidence'])
    sign_ref = doc_ref.collection(u'sign-detected').stream()
    for sign in sign_ref:
        res = sign.to_dict()
        for alphabet, prob in res.items():
            # use pandas.concat
            sign_df = pd.concat([sign_df, pd.DataFrame(
                {"Alphabet": [alphabet], "Confidence": [prob]})], ignore_index=True)

    st.dataframe(sign_df)


def main():
    image = Image.open("static/vid_call.jpg")
    logo = Image.open("static/logo.png")
    st.set_page_config(page_title="SpeechSign", page_icon=logo)

    st.image(image)
    st.title("@SpeechSign")
    cache_query_param()

    if st.session_state.key:
        user_id = st.session_state.key

        app, db = firebase()
        st.sidebar.title("Select the process to your convinience")
        st.sidebar.markdown("Select the conversion method accordingly:")
        algo = st.sidebar.selectbox(
            "Select the Operation", options=["Sign-to-Speech", "Speech-to-Sign", "Access Database", "Sign Recog Model Architecture"]
        )

        if algo == "Sign-to-Speech":
            sign_detection(db, user_id=user_id)
        elif algo == "Speech-to-Sign":
            speech_detection()
        elif algo == "Access Database":
            show_database(db, user_id=user_id)
        elif algo == "Sign Recog Model Architecture":
            st.title("Sign Recog Model Architecture")
            st.image("static/arch.png")


if __name__ == "__main__":
    main()
