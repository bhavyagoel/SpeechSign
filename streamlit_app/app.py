import streamlit as st
import asyncio
import queue
import threading
from pathlib import Path
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pyrebase
from streamlit_webrtc import webrtc_streamer
import json
from PIL import Image


def firebase():
    if not firebase_admin._apps:
            cred = credentials.Certificate('../speechsign-23477-8f5b84f0980a.json')
            app = firebase_admin.initialize_app(cred)
    app = firebase_admin.get_app()
    db = firestore.client()
    return app, db

@st.cache
def cache_query_param():
    query_param = st.experimental_get_query_params()
    return query_param

def read_user_info(db):    
    doc_ref = db.collection(u'users').document(query_param['user'][0])
    doc = doc_ref.get()
    if doc.exists:
        user_det = doc.to_dict()
        
        st.write(doc.to_dict())
        
webrtc_streamer(key="example")

def main():
    image = Image.open("vid_call.png")
    logo = Image.open("logo.png")
    st.set_page_config(page_title="ManPages", page_icon=logo)
    st.image(image)
    st.title("@SpeechSign")
    app,db = firebase()
    query_param = cache_query_param()

    
