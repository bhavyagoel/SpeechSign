from flask import Flask, render_template, flash, redirect, url_for, session, request
import pyrebase
from functools import wraps


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


firebase = pyrebase.initialize_app(config)

app = Flask(__name__)
auth = firebase.auth()


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login', methods=['POST','GET'])
def login():

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            auth.sign_in_with_email_and_password(email, password)
            # user_id = auth.get_account_info(user['idToken'])
            
            session["logged_in"] = True

            return render_template("login.html")
        
        except:
            unsuccessful = " Please check your credentials."
            print("Unsuccessful login")
            return render_template("login.html", umessage=unsuccessful)
        
    return render_template("login.html")


@app.route('/register', methods=['POST','GET'])
def register():

    if request.method == 'POST':
        password = request.form['password']
        email = request.form['email']
        auth.create_user_with_email_and_password(email,password)

    return render_template("login.html")



# to prevent using of app without login
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorised, Please Login', 'danger')
            return redirect(url_for('login'))

    return wrap


