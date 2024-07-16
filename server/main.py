import logging

from flask import Flask, request
from pkg_resources import require

from embeddings import generate_embeddings

app = Flask(__name__)


@app.post('/embed')
def embeddings():
    text = request.get_json().get('inputs')
    out = generate_embeddings(text).tolist()
    logging.info(out)
    return out


app.run(host='0.0.0.0', port=7000, debug=True)
