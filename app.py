from model import Model
from flask import Flask, request, jsonify

app = Flask(__name__)

model = Model()
model.load_weights()


@app.route("/")
def index():
    return '<h1>Welcome to Euro-Vision Flask server!</h1>'


@app.route("/predict_match", methods=['GET'])
def predict_match():
    try:
        team1 = request.args.get("team1")
        team2 = request.args.get("team2")
        return model.predictSingleMatch(team1, team2)
    except Exception as e:
        return "Error has occured"
