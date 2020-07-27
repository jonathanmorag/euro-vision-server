from model import Model
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = Model()
model.load_weights()
group_stage_winners = model.predictGroupMatches()

# Using a dictionary in order to reduce computations
dict_stages = {}


@app.route("/")
def index():
    return '<h1>Welcome to Euro-Vision Flask server!</h1>'


# Send a GET request such as: localhost:5000/predict_match?team1=France&team2=Germany
@app.route("/predict_match", methods=['GET'])
def predict_match():
    try:
        team1 = request.args.get("team1")
        team2 = request.args.get("team2")
        predicted = model.predictSingleMatch(team1, team2)
        return jsonify(winner=predicted[0], home_rate=predicted[1], draw_rate=predicted[2], away_rate=predicted[3])
    except Exception as e:
        return "Error has occured"


# Send a GET request such as: localhost:5000/predict_stage?team=France
@app.route("/predict_stage", methods=['GET'])
def predict_stage():
    try:
        team = request.args.get("team")
        if(team not in dict_stages):
            model.predictStage(group_stage_winners, team, 0)
            dict_stages[team] = model.stage
        return jsonify(stage=dict_stages[team])
    except Exception as e:
        return "Error has occured"


# Send a GET request such as: localhost:5000/predict_winner
@app.route("/predict_winner", methods=['GET'])
def predict_winner():
    try:
        model.predictEuroWinner(group_stage_winners)
        return jsonify(eurowinner=model.euro_winner)
    except Exception as e:
        return "Error has occured"
