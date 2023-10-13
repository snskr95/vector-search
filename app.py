from flask import Flask, jsonify, request
from vectorSearch import *
from triples_data import *

# creating a Flask app
app = Flask(__name__)


# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/', methods=['GET', 'POST'])
def home():
    if (request.method == 'GET'):
        query = request.args.get('query')
        return get_most_relevant_doc(query)


@app.route('/generate', methods=['GET'])
def generate_data():
    generate_text_embeddings(triples)
    return "done"


@app.route('/all-data', methods=['GET'])
def all_data():
    return text_embedding_list


@app.route('/get-data', methods=['GET'])
def get_data():
    call_api()
    return "ok"


# driver function
if __name__ == '__main__':
    app.run(debug=False)
