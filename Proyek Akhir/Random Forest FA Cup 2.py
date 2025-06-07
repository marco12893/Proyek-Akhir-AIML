import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("clean_data/English_Football_2018-2023_With_Form.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Encode teams
le_team = LabelEncoder()
le_team.fit(pd.concat([df['Home'], df['Away']]))
df['HomeTeam_enc'] = le_team.transform(df['Home'])
df['AwayTeam_enc'] = le_team.transform(df['Away'])

# Add DivisionGap
df['DivisionGap'] = df['AwayDivision'] - df['HomeDivision']

# Filter training and test data
train_df = df[~((df['Type'] == 'FA Cup') & (df['Season'] == 2023))]
test_df = df[(df['Type'] == 'FA Cup') & (df['Season'] == 2023) & (df['Winner'] != 1)]

# Features and labels
features = ['HomeTeam_enc', 'AwayTeam_enc', 'DivisionGap', 'NeutralVenue']
X_train = train_df[features]
y_train = train_df['Winner']
X_test = test_df[features]
y_test = test_df['Winner']

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict probabilities
probs = model.predict_proba(X_test)

# Convert predicted class: replace '1' (Draw) with the more confident class (0 or 2)
y_pred_raw = np.argmax(probs, axis=1)
y_pred = []
for i, pred in enumerate(y_pred_raw):
    if pred == 1:
        y_pred.append(0 if probs[i][0] > probs[i][2] else 2)
    else:
        y_pred.append(pred)
y_pred = np.array(y_pred)

# Map outcomes
outcome_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}

# Results DataFrame
results_df = test_df[['Date', 'Home', 'Away', 'Winner']].copy()
results_df['Predicted'] = y_pred
results_df['Correct'] = results_df['Winner'] == results_df['Predicted']
results_df['Actual Outcome'] = results_df['Winner'].map(outcome_map)
results_df['Predicted Outcome'] = results_df['Predicted'].map(outcome_map)
results_df['Home Win Prob'] = (probs[:, 0] * 100).round(1)
results_df['Draw Prob'] = (probs[:, 1] * 100).round(1)
results_df['Away Win Prob'] = (probs[:, 2] * 100).round(1)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy on FA Cup test set: {accuracy * 100:.2f}%\n")

unique_classes = np.unique(y_test)
print(f"Classes present in test data: {unique_classes}")

target_names = ['Home Win', 'Away Win']

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=unique_classes))

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                           labels=unique_classes,
                           target_names=target_names,
                           zero_division=0))

# Feature importance plot
importances = model.feature_importances_
plt.figure(figsize=(8, 4))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
