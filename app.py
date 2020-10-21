from flask import Flask, request, redirect, send_file, g, make_response
from flask_restful import Api
from flask_socketio import SocketIO
from flask_cors import CORS
from api import UploadImage, Classify
from precess import registrate, detect
import cv2
import os
import io

app = Flask(__name__)

api = Api(app)
socketio = SocketIO(app)
CORS(app)

app.config['SECRET_KEY'] = 'secret!'
# app.config['UPLOAD_DIR'] = './uploads'
print(os.environ.get('UPLOAD_DIR'))


api.add_resource(UploadImage, '/api/upload/<string:fname>')
api.add_resource(Classify, '/api/classify')


@app.route('/reg')
def reg():
    imflawpath = os.path.join(os.environ.get('UPLOAD_DIR'), 'flaw.jpg')
    imrefpath = os.path.join(os.environ.get('UPLOAD_DIR'), 'base.jpg')
    registrate(imflawpath, imrefpath)
    
    return send_file(os.path.join(os.environ.get('UPLOAD_DIR'), 'reg.jpg'), mimetype='image/jpg')
    
    # imgReg = registrate(imflawpath, imrefpath)
    # _, buffer = cv2.imencode('.jpg', imgReg)
    # response = make_response(buffer.tobytes())
    # response.headers['Content-Type'] = 'image/jpg'
    # return response

@app.route('/det')
def det():
    imregpath = os.path.join(os.environ.get('UPLOAD_DIR'), 'reg.jpg')
    imrefpath = os.path.join(os.environ.get('UPLOAD_DIR'), 'base.jpg')
    detect(imregpath, imrefpath)

    return send_file(os.path.join(os.environ.get('UPLOAD_DIR'), 'det.jpg'), mimetype='image/jpg')


if __name__ == '__main__':
    socketio.run(app)