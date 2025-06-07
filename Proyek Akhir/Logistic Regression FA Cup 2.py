import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv("clean_data/English_Football_2018-2023_With_Form.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Drop draws for binary classification: 0 = home loss, 2 = home win
df = df[df['Winner'] != 1].copy()

# Encode teams
le_team = LabelEncoder()
le_team.fit(pd.concat([df['Home'], df['Away']]))
df['HomeTeam_enc'] = le_team.transform(df['Home'])
df['AwayTeam_enc'] = le_team.transform(df['Away'])

# Add DivisionGap
df['DivisionGap'] = df['AwayDivision'] - df['HomeDivision']

# Train-test split: FA Cup 2023 as test set
train_df = df[~((df['Type'] == 'FA Cup') & (df['Season'] == 2023))]
test_df = df[(df['Type'] == 'FA Cup') & (df['Season'] == 2023)]

# Define features
features = ['HomeTeam_enc', 'AwayTeam_enc', 'DivisionGap', 'NeutralVenue']
X_train = train_df[features]
y_train = train_df['Winner']
X_test = test_df[features]
y_test = test_df['Winner']

# Train logistic regression model
model = LogisticRegression(
    class_weight='balanced',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluation
print("\nLogistic Regression (w/ DivisionGap & NeutralVenue)")
print("="*55)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 2]))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=[0, 2],
    target_names=['Home Loss (0)', 'Home Win (2)']
))

# Results dataframe
results_df = test_df[['Date', 'Home', 'Away', 'Winner']].copy()
results_df['Predicted'] = y_pred
results_df['Home Loss Prob'] = (y_proba[:, 0] * 100).round(1)
results_df['Home Win Prob'] = (y_proba[:, 1] * 100).round(1)
results_df['Correct'] = results_df['Winner'] == results_df['Predicted']

print("\nSample Predictions:")
print(results_df.head(10))

# Model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_[0]:.4f}")
for feature_name, coef in zip(features, model.coef_[0]):
    print(f"{feature_name}: {coef:.4f}")
