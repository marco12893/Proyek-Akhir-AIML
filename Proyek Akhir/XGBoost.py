import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/Premier League 2018-2023 XGBoost.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Encode categorical team names
le_team = LabelEncoder()
df['HomeTeam_enc'] = le_team.fit_transform(df['Home'])
df['AwayTeam_enc'] = le_team.transform(df['Away'])

# Features and label
X = df[['HomeTeam_enc', 'AwayTeam_enc', 'xG', 'xG.1']]
y = df['Winner']  # should be -1 (away win), 0 (draw), 1 (home win)

# Train-test split
train_df = df[df['Date'].dt.year < 2023]
test_df = df[df['Date'].dt.year == 2023]

X_train = train_df[['HomeTeam_enc', 'AwayTeam_enc', 'xG', 'xG.1']]
y_train = train_df['Winner']
X_test = test_df[['HomeTeam_enc', 'AwayTeam_enc', 'xG', 'xG.1']]
y_test = test_df['Winner']

# Train XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy * 100:.2f} %\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
