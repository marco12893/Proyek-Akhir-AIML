import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("data/English Football 2018-2023 XGBoost.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Encode teams
le_team = LabelEncoder()
le_team.fit(pd.concat([df['Home'], df['Away']]))
df['HomeTeam_enc'] = le_team.transform(df['Home'])
df['AwayTeam_enc'] = le_team.transform(df['Away'])

# Filter training and testing data
# Train on all data except Premier League 2023
train_df = df[~((df['Type'] == 'League') & (df['Season'] == 2023) & (df['HomeDivision'] == 1) & (df['AwayDivision'] == 1))]
# Test on Premier League 2023
test_df = df[(df['Type'] == 'League') & (df['Season'] == 2023) & (df['HomeDivision'] == 1) & (df['AwayDivision'] == 1)]

# Features and labels
features = ['HomeTeam_enc', 'AwayTeam_enc', 'HomeDivision', 'AwayDivision', 'NeutralVenue']

X_train = train_df[features]
y_train = train_df['Winner']

X_test = test_df[features]
y_test = test_df['Winner']

# Train XGBoost
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

# Predict classes
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy on Premier League 2023 test set: {accuracy * 100:.2f} %\n")

# Get all possible classes
classes = [0, 1, 2]
target_names = ['Home Win', 'Draw', 'Away Win']

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=classes))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=classes, target_names=target_names, zero_division=0))

# Create a results DataFrame with actual match information
results_df = test_df[['Date', 'Home', 'Away', 'Winner']].copy()
results_df['Predicted'] = y_pred
results_df['Correct'] = results_df['Winner'] == results_df['Predicted']

# Map numeric outcomes to text for better readability
outcome_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
results_df['Actual Outcome'] = results_df['Winner'].map(outcome_map)
results_df['Predicted Outcome'] = results_df['Predicted'].map(outcome_map)

print("\nSample Predictions:")
print(results_df[['Date', 'Home', 'Away', 'Actual Outcome', 'Predicted Outcome', 'Correct']].head(20))