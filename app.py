from model import Model
from flask import Flask

app = Flask(__name__)

model = Model()
model.load_weights()


@app.route("/")
def index():
    return model.predictSingleMatch('France', 'Spain')
