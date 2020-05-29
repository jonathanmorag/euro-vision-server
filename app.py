from model import Model
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = Model()
model.load_weights()


@app.route("/")
def index():
    return '<h1>Welcome to Euro-Vision Flask server!</h1>'


# Send a GET request such as: localhost:5000/predict_match?team1=France&team2=Germany
@app.route("/predict_match", methods=['GET'])
def predict_match():
    try:
        team1 = request.args.get("team1")
        team2 = request.args.get("team2")
        return jsonify(winner=model.predictSingleMatch(team1, team2))
    except Exception as e:
        return "Error has occured"
