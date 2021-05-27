from flask import Flask, render_template, flash, redirect, url_for, session, request
import pyrebase
from functools import wraps
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import urllib
import webbrowser
from itsdangerous import URLSafeTimedSerializer

config = {
    "apiKey": "AIzaSyADzfEN_Rc2zLB66Uf0fAGXjxV0QZKMAOo",
    "authDomain": "speechsign-23477.firebaseapp.com",
    "databaseURL": "https://speechsign-23477-default-rtdb.firebaseio.com",
    "projectId": "speechsign-23477",
    "storageBucket": "speechsign-23477.appspot.com",
    "messagingSenderId": "532919592392",
    "appId": "1:532919592392:web:7a2b387e025eb54eb34cfd",
    "measurementId": "G-PVZRH63J9K"
}




app = Flask(__name__)
app.secret_key = 'secret123'
s = URLSafeTimedSerializer('secret123')

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

cred = credentials.Certificate('speechsign-23477-8f5b84f0980a.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login', methods=['POST','GET'])
def login():

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if email != '' and password != '':
            try:
                user = auth.sign_in_with_email_and_password(email, password)            
                user_token = user['idToken']
                user_info = auth.get_account_info(user_token)
                user_id = user_info['users'][0]['localId']
                if user_info['users'][0]['emailVerified']:
                    session['logged_in'] = True
                    # webbrowser.open_new_tab('https://share.streamlit.io/bhavyagoel/speechsign-model/main/app.py/?user='+user_id)
                    return redirect('https://share.streamlit.io/bhavyagoel/speechsign-model/main/app.py/?user='+user_id)
                else:
                    unsuccessful = "Please verify your Email ID."
                    auth.send_email_verification(user_token)
                    print("Email Not Verified")
                    return render_template("login.html", umessage=unsuccessful)
            except:
                unsuccessful = "Please check your credentials."
                print("Unsuccessful login")
                return render_template("login.html", umessage=unsuccessful)
        else:
            unsuccessful = "Please fill the credentials properly."
            print("Empty Credentials")
            return render_template("login.html", umessage=unsuccessful)
    return render_template("login.html")


@app.route('/register', methods=['POST','GET'])
def register():

    if request.method == 'POST':
        password = request.form['password']
        email = request.form['email']
        name = request.form['name']
        dob = request.form['dob']
        if password!='' and email !='' and name !='' and dob !='':
            user = auth.create_user_with_email_and_password(email,password)
            user_token = user['idToken']
            auth.send_email_verification(user_token)
            user_info = auth.get_account_info(user_token)
            user_id = user_info['users'][0]['localId']

            doc_ref = db.collection(u'users').document(user_id)
            doc_ref.set({
                u'name':name,
                u'dob':dob,
                u'email':email
            })
    return render_template("login.html")


if __name__ == '__main__':
    app.secret_key = 'secret123'
    app.run(debug=True, threaded=True)