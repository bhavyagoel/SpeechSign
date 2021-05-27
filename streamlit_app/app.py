import streamlit as st
import asyncio
import queue
import threading
from pathlib import Path
# import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


cred = credentials.Certificate('speechsign-23477-8f5b84f0980a.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

a = st.experimental_get_query_params()
st.write(a)


