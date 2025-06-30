import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("clean_data/English_Football_2018-2023_With_Form.csv")
df['Date'] = pd.to_datetime(df['Date'])

le_team = LabelEncoder()
le_team.fit(pd.concat([df['Home'], df['Away']]))
df['HomeTeam_enc'] = le_team.transform(df['Home'])
df['AwayTeam_enc'] = le_team.transform(df['Away'])

# division gap & weighted form features
df['DivisionGap'] = df['AwayDivision'] - df['HomeDivision']
df['AbsoluteDivisionGap'] = df['DivisionGap'].abs()
df['HomeFormWeighted'] = df['HomeLast5_Wins'] / (df['AbsoluteDivisionGap'] + 1)
df['AwayFormWeighted'] = df['AwayLast5_Wins'] * (df['AbsoluteDivisionGap'] + 1)

# filter data
train_df = df[
    (~((df['Type'] == 'League') & (df['Season'] == 2023) & (df['HomeDivision'] == 1) & (df['AwayDivision'] == 1))) &
    (df['Type'] == 'League')
]
test_df = df[(df['Type'] == 'League') & (df['Season'] == 2023) & (df['HomeDivision'] == 1) & (df['AwayDivision'] == 1)]

features = [
    'HomeTeam_enc', 'AwayTeam_enc',
    'HomeDivision', 'AwayDivision',
    'NeutralVenue', 'DivisionGap', 'AbsoluteDivisionGap',
    'HomeLast5_Wins', 'AwayLast5_Wins',
    'HomeFormWeighted', 'AwayFormWeighted'
]

X_train = train_df[features]
y_train = train_df['Winner']
X_test = test_df[features]
y_test = test_df['Winner']

model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
accuracyFormatted=accuracy*100

if __name__ == '__main__':
    print(f"Logistic Regression Accuracy on Premier League 2023 test set: {accuracy * 100:.2f} %\n")

    classes = [0, 1, 2]
    target_names = ['Away Win', 'Draw', 'Home Win']

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=classes))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=classes, target_names=target_names, zero_division=0))

    # Display results
    results_df = test_df[['Date', 'Home', 'Away', 'Winner']].copy()
    results_df['Predicted'] = y_pred
    results_df['Correct'] = results_df['Winner'] == results_df['Predicted']

    outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    results_df['Actual Outcome'] = results_df['Winner'].map(outcome_map)
    results_df['Predicted Outcome'] = results_df['Predicted'].map(outcome_map)

    print("\nSample Predictions:")
    print(results_df[['Date', 'Home', 'Away', 'Actual Outcome', 'Predicted Outcome', 'Correct']].head(20))
