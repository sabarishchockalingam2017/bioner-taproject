import logging

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
from helpers import text2features
from nltk.tokenize import word_tokenize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('text')

# load fitted model
model = pickle.load(open('./models/bestbioner.model', 'rb'))

class BioNER(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['text']
        tokd = word_tokenize(user_query)
        label = model.predict([text2features(user_query)])
        taggedout = [tokd[l]+' - ' + label[0][l] for l in range(len(label[0])) if label[0][l] in ['B', 'I']]
        # create JSON object
        output = {'Gene/Protein': taggedout}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(BioNER, '/')

if __name__ == '__main__':
    app.run(debug=True)


