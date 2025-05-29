import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# Load Premier League training data
pl_df = pd.read_csv("data/Premier League 2018-2023 XGBoost.csv")
pl_df['Date'] = pd.to_datetime(pl_df['Date'])

# Load FA Cup test data
fa_df = pd.read_csv("data/FA Cup 2023-2024 Test.csv")
fa_df['Date'] = pd.to_datetime(fa_df['Date'])

# Encode teams using the same encoder
le_team = LabelEncoder()
all_teams = pd.concat([pl_df[['Home']], pl_df[['Away']], fa_df[['Home']], fa_df[['Away']]])
le_team.fit(pd.concat([all_teams['Home'], all_teams['Away']]))

pl_df['HomeTeam_enc'] = le_team.transform(pl_df['Home'])
pl_df['AwayTeam_enc'] = le_team.transform(pl_df['Away'])
fa_df['HomeTeam_enc'] = le_team.transform(fa_df['Home'])
fa_df['AwayTeam_enc'] = le_team.transform(fa_df['Away'])

# Features and labels
X_train = pl_df[['HomeTeam_enc', 'AwayTeam_enc']]
y_train = pl_df['Winner']  # Already remapped to 0/1/2

X_test = fa_df[['HomeTeam_enc', 'AwayTeam_enc']]
y_test = fa_df['Winner']   # Also needs to be 0/1/2

# Train XGBoost
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

# Predict on FA Cup data
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy on FA Cup: {accuracy * 100:.2f} %\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
