import streamlit as st
import queue
from streamlit.elements import form
import av
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pyrebase
from streamlit_webrtc import (
    webrtc_streamer, 
    VideoProcessorBase,
    WebRtcMode,
)
from typing import List, NamedTuple
from PIL import Image, ImageOps
import cv2
import tensorflow.keras
import numpy as np
np.set_printoptions(suppress=True)


def firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate('static/speechsign-23477-8f5b84f0980a.json')
        app = firebase_admin.initialize_app(cred)
    app = firebase_admin.get_app()
    db = firestore.client()
    return app, db

@st.cache
def cache_query_param():
    query_param = st.experimental_get_query_params()
    return query_param

def read_user_info(db, query_param):    
    doc_ref = db.collection(u'users').document(query_param['user'][0])
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()

def gen_labels():
        labels = {}
        with open("model/labels.txt", "r") as label:
            text = label.read()
            lines = text.split("\n")
            for line in lines[0:-1]:
                    hold = line.split(" ", 1)
                    labels[hold[0]] = hold[1]
        return labels

class Detection(NamedTuple):
        name: str
        prob: float

class VideoTransformer(VideoProcessorBase):

    result_queue: "queue.Queue[List[Detection]]"

    def __init__(self) -> None:
        self.threshold1 = 224
        self.result_queue = queue.Queue()

    def _predict_image(self, image, model):
        result: List[Detection] = []
        labels = gen_labels()
        prediction = model.predict(image)
        confidence = max(prediction[0])
        st.write(confidence)
        idx = np.where(prediction[0] == confidence)
        alpha = labels.get(str(idx[0][0]))
        result.append(Detection(name=alpha, prob=float(confidence)))
        return result

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        frame = cv2.resize(img, (224,224))
        frame = Image.fromarray(frame)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        image = ImageOps.fit(frame, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        model = tensorflow.keras.models.load_model("model/keras_model.h5")
        
        result = self._predict_image(data, model)
        
        self.result_queue.put(result)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def sign_detection():
    
    ctx = webrtc_streamer(
        key="SpeechSign",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoTransformer, 
        async_processing=True,)
    
    if st.checkbox("Show the detected labels", value=True):
        if ctx.state.playing:
            labels_placeholder = st.empty()
            while True:
                if ctx.video_processor:
                    try:
                        result = ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break

def main():
    image = Image.open("static/vid_call.jpg")
    logo = Image.open("static/logo.png")
    st.set_page_config(page_title="SpeechSign", page_icon=logo)
    st.image(image)
    st.title("@SpeechSign")
    app, db = firebase()
    query_param = cache_query_param()
    user_det = read_user_info(db, query_param)
    sign_detection()

if __name__ == "__main__":
    main()