import re
import streamlit as st
import queue
import av
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from streamlit_webrtc import (
    webrtc_streamer, 
    VideoProcessorBase,
    WebRtcMode,
    ClientSettings
)
from typing import List, NamedTuple
from PIL import Image, ImageOps
import cv2
import tensorflow.keras
import numpy as np
import pandas as pd
from bokeh.models.widgets import Button, widget
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

np.set_printoptions(suppress=True)
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)

global query_param 

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
    print(query_param)
    user_id = query_param['user'][0]
    return user_id, query_param

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
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
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
        self.frame = frame
        img = frame.to_ndarray(format="bgr24")
        frm = cv2.resize(img, (224,224))
        frm = Image.fromarray(frm)
        size = (224, 224)
        image = ImageOps.fit(frm, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        self.data[0] = normalized_image_array
        model = tensorflow.keras.models.load_model("model/keras_model.h5", compile=False)
        result = self._predict_image(self.data, model)
        self.result_queue.put(result)

        return 0

def sign_detection(db, user_id):
    
    ctx = webrtc_streamer(
        key="SpeechSign",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
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
                        doc_ref = db.collection(u'users').document(user_id).collection('sign-detected').add({result[0].name:result[0].prob})
                    except queue.Empty:
                        result = None
                    
                    labels_placeholder.table(result)
                else:
                    break

def speech_detection():
    st.header("Press the following button to speak.")
    st.write("As soon as you press the button, microphone of your device gets activated and your audio is converted to Sign Language.")
    stt_button = Button(label="Speak", width=100)

    stt_button.js_on_event("button_click", CustomJS(code="""
        var recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
    
        recognition.onresult = function (e) {
            var value = "";
            for (var i = e.resultIndex; i < e.results.length; ++i) {
                if (e.results[i].isFinal) {
                    value += e.results[i][0].transcript;
                }
            }
            if ( value != "") {
                document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
            }
        }
        recognition.start();
        """))

    result = streamlit_bokeh_events(
        stt_button,
        events="GET_TEXT",
        key="listen",
        refresh_on_update=False,
        override_height=75,
        debounce_time=0)

    if result:
        if "GET_TEXT" in result:
            text = result.get("GET_TEXT")
            text = text.upper()
            text = text.replace(' ','')
            mean_width = 0
            mean_height = 0
            num_of_images = len(text)
            for i in text:
                im = Image.open("static/sign_alpha/"+i+".jpg")
                width, height = im.size
                mean_width += width
                mean_height += height

            mean_width = int(mean_width/ num_of_images)
            mean_height = int(mean_height/ num_of_images)
            images = []
            for i in text:
                im = Image.open("static/sign_alpha/"+i+".jpg")
                width, height = im.size

                imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)
                imResize.save("video_proc/"+i+".jpeg",'JPEG', quality=95)
            
            video_name = 'video_proc/{}.webm'.format(text)
            frame = cv2.imread("video_proc/"+text[0]+".jpeg")
            height, width, layers = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'VP90')
            video = cv2.VideoWriter(video_name,fourcc, 1, (width, height)) 

            for i in text:
                video.write(cv2.imread("video_proc/"+i+".jpeg"))

            cv2.destroyAllWindows()
            video.release()
            st.header(result.get("GET_TEXT").title())
            st.video("video_proc/{}.webm".format(text))
    return 0

def show_database(db, user_id):
    doc_ref = db.collection(u'users').document(user_id)
    doc = doc_ref.get()
    user_det = doc.to_dict()
    st.header("User Details")
    user_df = pd.DataFrame({"Type":['Name','DOB','Email'], "Value":[user_det['name'],user_det['dob'],user_det['email']]})
    st.dataframe(user_df)

    st.header('Sign Detected')
    sign_df = pd.DataFrame(columns=['Alphabet','Confidence'])
    sign_ref = doc_ref.collection(u'sign-detected').stream()
    for sign in sign_ref:
        res = sign.to_dict()
        for alphabet,prob in res.items():
            sign_df = sign_df.append({'Alphabet':alphabet,'Confidence':prob}, ignore_index=True)

    st.dataframe(sign_df)


def main():
    image = Image.open("static/vid_call.jpg")
    logo = Image.open("static/logo.png")
    st.set_page_config(page_title="SpeechSign", page_icon=logo)
    
    st.image(image)
    st.title("@SpeechSign")
    user_id, query_param = cache_query_param()

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