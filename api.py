from flask import Flask
from flask_restplus import Api, Resource, fields, abort, reqparse
from flask_cors import CORS
from werkzeug.utils import secure_filename

from load_model import load_model
from test_model import test_model
from sources import utils

import logging
import os
import werkzeug
import json
import sys

app = Flask(__name__)
CORS(app)
api = Api(app)

file_upload = api.parser()
file_upload = reqparse.RequestParser()
file_upload.add_argument('img',type=werkzeug.datastructures.FileStorage,location='files',required=True,help='Image File')  

config = utils.load_json_file("config.json")

@api.route('/test')
class Test(Resource):
    @api.doc(responses={ 201: 'ok', 500: 'internal server error' })  
    @api.expect(file_upload)
    def post(self):
        # Read config file
        model_name = config["model_name"]
        img_width = config["img_width"]
        img_height = config["img_height"]
        classes = config["classes"]
        
        args = file_upload.parse_args()
        
        args['img'].save(os.path.join('test_images/',secure_filename(args['img'].filename)))
        
        # Load trained model
        model_path = 'models/{}'.format(model_name)
        model = load_model(model_path)

        # Attributes
        img_path = os.path.join('test_images/',secure_filename(args['img'].filename))

        predicted_classes = test_model(img_path,classes,model,img_width,img_height)
        
        return {'result': predicted_classes}


if __name__ == '__main__':
    
    app.run(host="127.0.0.1",port="5000",debug=True)
