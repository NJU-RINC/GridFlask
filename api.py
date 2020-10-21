from werkzeug import datastructures
from flask_restful import Resource, reqparse
import os
import cv2
from mobilenet import classify

class UploadImage(Resource):
    def post(self, fname):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=datastructures.FileStorage, location='files')
        args = parse.parse_args()
        img_file = args['file']
        img_file.save(os.path.join(os.environ.get('UPLOAD_DIR'), fname))

        return {'Code': 'OK'}

class Classify(Resource):
    def get(self):
        imCrop = cv2.imread(os.path.join(os.environ.get('UPLOAD_DIR'), 'crop.jpg'))
        category = classify(imCrop)

        return {'Class': category}

