import os
from flask import Flask, flash, request, redirect, url_for, session,render_template,jsonify,Response
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging
from keras.models import load_model
from Text.Text import Text
from Audio.Audio import Diariazation
from Video.test import Video_Analysis
import hashlib
logging.basicConfig(level=logging.INFO)
import json
from flask_sqlalchemy import SQLAlchemy
import validators
logger = logging.getLogger('HELLO WORLD')
directory = os.getcwd()



'''
UPLOAD_FOLDER = './Video'
model=load_model('./Video/saved_model.h5')
'''

ALLOWED_TEXT_EXTENSIONS = ['txt', 'pdf', 'docx','odt']
ALLOWED_VIDEO_EXTENSIONS=['mp4','mov','wmv','avi','mkv','gif']
ALLOWED_AUDIO_EXTENSIONS=['wav']
app = Flask(__name__)
import os.path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "database.db")
#with sqlite3.connect(db_path) as db:
app.config[ 'SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app. config[ 'SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)




db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'Users'
    id = db.Column(db.Integer, primary_key=True) 
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(80), nullable=False)
    password = db.Column(db.String(80), nullable=False)

    def json(self):
        if(self):
            return {'id': self.id, 'username': self.username,
                    'email': self.email, 'password': self.password}
        return None

    def add_user(_username, _email, _password):

        new_user = User(username=_username, email= _email, password=_password)
        db.session.add(new_user) 
        db.session.commit()  

    def get_all_users():
        return [User.json(user) for user in User.query.all()]


db.create_all()


#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/',methods=['GET'])
def check():
    # text_class=Text()
    # a,b,c=list(text_class.topic_modelling(file_path='C:\\Users\\vishu\\OneDrive\\Desktop\\Vishal\\final year project\\Text-analysis~\\Final_project_Server-main\\xyz.pdf'))
    return "Listening on 5000"
    #print(summary)
    #a=[str(i) for i in a]
    #a=" ".join(a)
    #return a


@app.route('/login', methods=['POST'])
def login():
    request_data = request.form
    email=request_data["email"]
    password = request_data["password"]
    print(email,password)
    password = hashlib.sha256(password.encode()).hexdigest()

    user = User.json(User.query.filter_by(email=email).first())
    response={'loggedin':'false','msg':"Invalid Username or Password"}
    if user and password==user["password"]:
        response['loggedin']='true'
        response['username']=user['username']
        response['msg']="Login Successfully"        
        return response
    else:
        return response


@app.route('/register', methods=['POST'])
def add_user():
    
    request_data = request.form
    username = request_data["username"]
    email = request_data["email"]
    password = request_data["password"]
    print(username,email,password)
    password = hashlib.sha256(password.encode()).hexdigest()
    User.add_user(username, email, password)
    response = Response("User Registered Successfully", 201, mimetype='application/json')
    return response


@app.route('/<filename>',methods=['GET'])
@cross_origin()
def lol(filename):
    print(filename)
    return render_template(str(filename))




@app.route('/uploadText', methods=['POST','GET'])
@cross_origin()
def fileUpload():
    UPLOAD_FOLDER="./Text"
    target=os.path.join(UPLOAD_FOLDER,'TextD')
    if not os.path.isdir(target):
        os.mkdir(target)
    logger.info("welcome to upload`")
    file_path = None
    url=None
    text=None
    scanned=request.form["scanned"]
    if(scanned=='1'):
        scanned=True
    else:
        scanned=False
    if(request.files):
        file = request.files['file']
        filename = secure_filename(file.filename)
        txt_ext=filename.split('.')[-1]
        if scanned and txt_ext!='pdf':
            return {'msg':'Only pdf scanned Files are supported'}
        if txt_ext not in ALLOWED_TEXT_EXTENSIONS:
            s=",".join(ALLOWED_TEXT_EXTENSIONS)
            return {'msg':'only ' +s +' files are Supported'} 
        
        print("format ?>>>>>>>>>>>>>",filename.split('.')[-1])
        destination="/".join([target, filename])
        print(destination,filename)
        file.save(destination)
        file_path=destination
    elif(request.form["url"]):
        url=request.form["url"]
        if not validators.url(url):
            return {'msg':'Invalid URL chcek URL again'}
        print("lol",url)
    else:
        text=request.form["text"]
    
   
    
    text_class=Text(scan=scanned)
    text_class.file_path_ext(file_path, url,text)
    summary=text_class.extractive_summary()
    if summary==None:
        return {'msg':"At least 2 paragraphs are required in Text input"}
    a,b,c,d=list(text_class.topic_modelling())
    
    readability=text_class.readability_analysis()
    
    response={'summary':summary,'readability':readability,'topic_modelling':[a,b,c,d]}

    # session['uploadFilePath']=destination
    # response={"test":"heyy","response":res,"reconstruction":reconstruction_array}
    
    #response={'summary':'summary','readability':'readability','topic_modelling':[['a','b','c'],['a','b','c'],['a','b','c']]}
    print(response)
    return response

@app.route('/uploadVideo', methods=['POST','GET'])
@cross_origin()
def fileVideoUpload():

    UPLOAD_FOLDER = directory+"\\Video"
    target=os.path.join(UPLOAD_FOLDER,'videosD')
    demo= os.getcwd()
    print(demo)
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    logger.info("welcome to upload`")
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    video_ext=filename.split('.')[-1]
    print('>>>>>>>>>>>',filename.split('.')[-1])
    if video_ext not in ALLOWED_VIDEO_EXTENSIONS:
        s=",".join(ALLOWED_VIDEO_EXTENSIONS)
        return {'msg':"Only "+s+ ' are Supported','flag':1}
    destination="/".join([target, filename])
    print(destination,filename)
    file.save(destination)
    
    model=load_model(directory+"\\Video\\saved_model.h5")
    res,reconstruction_array=Video_Analysis(filename,model,destination)


    print("response",res)
    print("on server")
    session['uploadFilePath']=destination
    response={"test":"heyy","response":res,"reconstruction":reconstruction_array}
    
    
    return response



@app.route('/uploadAudio', methods=['POST','GET'])
@cross_origin()
def AudioFileUpload():
    UPLOAD_FOLDER="./Audio"
    target=os.path.join(UPLOAD_FOLDER,'AudioD')
    if not os.path.isdir(target):
        os.mkdir(target)
    logger.info("welcome to upload`")
    file_path = None
    url=None
    text=None
    if(request.files):
        file = request.files['file']
        filename = secure_filename(file.filename)
        if filename.split('.')[-1] not in ALLOWED_AUDIO_EXTENSIONS:
            s=",".join(ALLOWED_AUDIO_EXTENSIONS)
            return {'msg':"Only "+s+" are allowed",'flag':1}
        destination="/".join([target, filename])
        print(destination,filename)
        file.save(destination)
        file_path=destination
        ans=Diariazation(file_path,filename)
        return ans
    else :
        return {"msg":"No file uploaded"}



if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(debug=True,use_reloader=False,host='0.0.0.0')

# CORS(app, expose_headers='Authorization')
