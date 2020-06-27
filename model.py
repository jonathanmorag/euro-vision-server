import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

world_cup = pd.read_csv('db/Euro 2020 + Overall.csv')
results = pd.read_csv('db/results.csv')
ranking = pd.read_csv('db/fifa_rankings.csv')

euro_teams = ['Austria', 'Belgium', 'Croatia', 'Czech Republic'
              'Denmark', 'England', 'Finland', 'France', 'Georgia',
              'Germany', 'Iceland', 'Italy', 'Netherlands', 'Norway', 'Poland',
              'Portugal', 'Russia', 'Slovakia', 'Spain', 'Sweden', 'Switzerland', 'Turkey',
              'Ukraine', 'Wales']


class Model:

    def __init__(self):
        self.init = True
        self.fixtures = pd.read_csv('db/fixtures.csv')
        winner = []
        for i in range(len(results['home_team'])):
            if results['home_score'][i] > results['away_score'][i]:
                winner.append(results['home_team'][i])
            elif results['home_score'][i] < results['away_score'][i]:
                winner.append(results['away_team'][i])
            else:
                winner.append('Draw')
        results['winning_team'] = winner

        # adding goal difference column
        results['goal_difference'] = np.absolute(
            results['home_score'] - results['away_score'])

        df_teams_home = results[results['home_team'].isin(euro_teams)]
        df_teams_away = results[results['away_team'].isin(euro_teams)]
        df_teams = pd.concat((df_teams_home, df_teams_away))
        df_teams.drop_duplicates()

        year = []
        for row in df_teams['date']:
            year.append(int(row[:4]))
        df_teams['match_year'] = year
        df_teams_1930 = df_teams[df_teams.match_year >= 1930]

        df_teams_1930 = df_teams.drop(['date', 'home_score', 'away_score', 'tournament',
                                       'city', 'country', 'goal_difference', 'match_year'], axis=1)

        df_teams_1930 = df_teams_1930.reset_index(drop=True)
        df_teams_1930.loc[df_teams_1930.winning_team ==
                          df_teams_1930.home_team, 'winning_team'] = 2
        df_teams_1930.loc[df_teams_1930.winning_team ==
                          'Draw', 'winning_team'] = 1
        df_teams_1930.loc[df_teams_1930.winning_team ==
                          df_teams_1930.away_team, 'winning_team'] = 0

        self.final = pd.get_dummies(df_teams_1930, prefix=[
            'home_team', 'away_team'], columns=['home_team', 'away_team'])

        # Separate X and y sets
        X = self.final.drop(['winning_team'], axis=1)
        y = self.final["winning_team"]
        y = y.astype('int')

        # Separate train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42)

        self.pred_set = []

        # Create new columns with ranking position of each team
        self.fixtures.insert(1, 'first_position', self.fixtures['Home Team'].map(
            ranking.set_index('Team')['Position']))
        self.fixtures.insert(2, 'second_position', self.fixtures['Away Team'].map(
            ranking.set_index('Team')['Position']))

        # We only need the group stage games, so we have to slice the dataset
        self.fixtures = self.fixtures.iloc[:36, :]

        for index, row in self.fixtures.iterrows():
            if row['first_position'] < row['second_position']:
                self.pred_set.append(
                    {'home_team': row['Home Team'], 'away_team': row['Away Team'], 'winning_team': None})
            else:
                self.pred_set.append(
                    {'home_team': row['Away Team'], 'away_team': row['Home Team'], 'winning_team': None})

        self.pred_set = pd.DataFrame(self.pred_set)
        self.backup_pred_set = self.pred_set

        self.pred_set = pd.get_dummies(self.pred_set, prefix=['home_team', 'away_team'], columns=[
            'home_team', 'away_team'])

        # Add missing columns compared to the model's training dataset
        missing_cols = set(self.final.columns) - set(self.pred_set.columns)
        for c in missing_cols:
            self.pred_set[c] = 0
        self.pred_set = self.pred_set[self.final.columns]

        # Remove winning team column
        self.pred_set = self.pred_set.drop(['winning_team'], axis=1)

    def save_weights(self):
        self.logreg = LogisticRegression()

        self.logreg.fit(self.X_train, self.y_train)

        pickle.dump(self.logreg, open('trained_model.sav', 'wb'))

    def load_weights(self):
        self.logreg = pickle.load(open('trained_model.sav', 'rb'))

    def predictGroupMatches(self):
        standings = {}
        predictions = self.logreg.predict(self.pred_set)
        for i in range(self.fixtures.shape[0]):
            # print(self.backup_pred_set.iloc[i, 1] +
            #      " and " + self.backup_pred_set.iloc[i, 0])
            if predictions[i] == 0:
                team = self.backup_pred_set.iloc[i, 1]
                # print("Winner: " + team)
                if(team in standings):
                    standings[team] += 3
                else:
                    standings[team] = 3
            elif predictions[i] == 1:
                team1 = self.backup_pred_set.iloc[i, 0]
                team2 = self.backup_pred_set.iloc[i, 1]
                # print("Draw")
                if(team1 in standings):
                    standings[team1] += 1
                else:
                    standings[team1] = 1
                if(team2 in standings):
                    standings[team2] += 1
                else:
                    standings[team2] = 1
            elif predictions[i] == 2:
                team = self.backup_pred_set.iloc[i, 0]
                # print("Winner: " + self.backup_pred_set.iloc[i, 0])
                if(team in standings):
                    standings[team] += 3
                else:
                    standings[team] = 3

            # print('Probability of ' + self.backup_pred_set.iloc[i, 1] + ' winning: ', '%.3f' % (
            #     self.logreg.predict_proba(self.pred_set)[i][0]))
            # print('Probability of Draw: ', '%.3f' %
            #       (self.logreg.predict_proba(self.pred_set)[i][1]))
            # print('Probability of ' + self.backup_pred_set.iloc[i, 0] + ' winning: ', '%.3f' % (
            #     self.logreg.predict_proba(self.pred_set)[i][2]))
            # print("")
        standings = {k: v for k, v in sorted(
            standings.items(), key=lambda value: value[1], reverse=True)}
        group = []
        tuples = []
        [group.append(team) for team in standings]
        group = group[:8]

        i = 0
        while(i < len(group) - 1):
            tuples.append((group[i], group[i+1]))
            i += 2

        return tuples

    def clean_and_predict(self, matches):

        # Initialization of auxiliary list for data cleaning
        positions = []

        # Loop to retrieve each team's position according to FIFA ranking
        for match in matches:
            positions.append(
                ranking.loc[ranking['Team'] == match[0], 'Position'].iloc[0])
            positions.append(
                ranking.loc[ranking['Team'] == match[1], 'Position'].iloc[0])

        # Creating the DataFrame for prediction
        pred_set = []

        # Initializing iterators for while loop
        i = 0
        j = 0

        # 'i' will be the iterator for the 'positions' list, and 'j' for the list of matches (list of tuples)
        while i < len(positions):
            dict1 = {}

            # If position of first team is better, he will be the 'home' team, and vice-versa
            if positions[i] < positions[i + 1]:
                dict1.update(
                    {'home_team': matches[j][0], 'away_team': matches[j][1]})
            else:
                dict1.update(
                    {'home_team': matches[j][1], 'away_team': matches[j][0]})

            # Append updated dictionary to the list, that will later be converted into a DataFrame
            pred_set.append(dict1)
            i += 2
            j += 1

        # Convert list into DataFrame
        pred_set = pd.DataFrame(pred_set)
        backup_pred_set = pred_set

        # Get dummy variables and drop winning_team column
        pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=[
                                  'home_team', 'away_team'])

        # Add missing columns compared to the model's training dataset
        missing_cols2 = set(self.final.columns) - set(pred_set.columns)
        for c in missing_cols2:
            pred_set[c] = 0
        pred_set = pred_set[self.final.columns]

        # Remove winning team column
        pred_set = pred_set.drop(['winning_team'], axis=1)

        # Predict!
        winners = []
        predictions = self.logreg.predict(pred_set)
        for i in range(len(pred_set)):
            # print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
            if predictions[i] == 0:
                winners.append(backup_pred_set.iloc[i, 1])
                # print("Winner: " + backup_pred_set.iloc[i, 1])
            elif predictions[i] == 1:
                # print("Draw")
                if(self.logreg.predict_proba(pred_set)[i][0] > self.logreg.predict_proba(pred_set)[i][2]):
                    winners.append(backup_pred_set.iloc[i, 1])
                else:
                    winners.append(backup_pred_set.iloc[i, 0])
            elif predictions[i] == 2:
                # print("Winner: " + backup_pred_set.iloc[i, 0])
                winners.append(backup_pred_set.iloc[i, 0])
            # print('Probability of ' + backup_pred_set.iloc[i, 1] + ' winning: ', '%.3f' % (
            #     self.logreg.predict_proba(pred_set)[i][0]))
            # print('Probability of Draw: ', '%.3f' %
            #       (self.logreg.predict_proba(pred_set)[i][1]))
            # print('Probability of ' + backup_pred_set.iloc[i, 0] + ' winning: ', '%.3f' % (
            #     self.logreg.predict_proba(pred_set)[i][2]))
            # print("")

        return winners

    def predictEuroWinner(self, group):
        if len(group) == 1:
            euro_winner = self.clean_and_predict(group)
            self.euro_winner = euro_winner[0].upper()
        else:
            winners = self.clean_and_predict(group)
            random.shuffle(winners)
            next_group = []
            i = 0
            while i < (len(winners) - 1):
                next_group.append((winners[i], winners[i+1]))
                i += 2
            self.predictEuroWinner(next_group)

    def predictSingleMatch(self, team1, team2):
        tup = (team1, team2)
        return self.clean_and_predict([tup])[0]

    def predictStage(self, group, team, index):
        if len(group) == 1:
            euro_winner = self.clean_and_predict(group)
            if(euro_winner[0] == team):
                self.stage = 'Euro Winner!'
                return

        exists = False
        for t in group:
            t1, t2 = t
            if(t1 == team or t2 == team):
                exists = True
        if not exists:
            if index == 0:
                self.stage = 'Group Stage'
            if index == 1:
                self.stage = 'Quarter Finals'
            if index == 2:
                self.stage = 'Semi Finals'
            if index == 3:
                self.stage = 'Final'
            if index == 4:
                self.stage = 'Euro Winner!'

        if exists:
            winners = self.clean_and_predict(group)
            random.shuffle(winners)
            next_group = []
            i = 0
            while i < (len(winners) - 1):
                next_group.append((winners[i], winners[i+1]))
                i += 2
            self.predictStage(next_group, team, index+1)


########################################################################

# m = Model()
# m.load_weights()
# group_stage_winners = m.predictGroupMatches()
# m.predictStage(group_stage_winners, 'Netherlands', 0)
# print(m.stage)

########################################################################

# winner = m.predictSingleMatch('England', 'Germany')
# print(winner)

########################################################################

# m.predictEuroWinner(group_stage_winners)
# print(m.euro_winner)