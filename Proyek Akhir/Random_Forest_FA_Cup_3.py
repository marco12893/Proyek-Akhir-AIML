import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("clean_data/English_Football_2018-2023_With_Form.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Encode teams
le_team = LabelEncoder()
le_team.fit(pd.concat([df['Home'], df['Away']]))
df['HomeTeam_enc'] = le_team.transform(df['Home'])
df['AwayTeam_enc'] = le_team.transform(df['Away'])

# Enhanced feature engineering
def create_advanced_features(df):
    df = df.copy()
    df['DivisionGap'] = df['AwayDivision'] - df['HomeDivision']
    df['HigherDivisionTeam'] = np.where(df['DivisionGap'] < 0, 'Home', 'Away')
    df['AbsoluteDivisionGap'] = abs(df['DivisionGap'])
    df['HomeFormWeighted'] = df['HomeLast5_Wins'] / (df['AbsoluteDivisionGap'] + 1)
    df['AwayFormWeighted'] = df['AwayLast5_Wins'] * (df['AbsoluteDivisionGap'] + 1)
    df['IsCup'] = (df['Type'] != 'Premier League').astype(int)
    return df

df = create_advanced_features(df)

# Filter training and testing data
train_df = df[~((df['Type'] == 'FA Cup') & (df['Season'] == 2023))]
test_df = df[(df['Type'] == 'FA Cup') & (df['Season'] == 2023) & (df['Winner'] != 1)]

# Feature selection
base_features = [
    'HomeTeam_enc', 'AwayTeam_enc',
    'HomeDivision', 'AwayDivision',
    'NeutralVenue', 'DivisionGap',
    'AbsoluteDivisionGap'
]

form_features = [
    'HomeLast5_Wins', 'AwayLast5_Wins',
    'HomeFormWeighted', 'AwayFormWeighted'
]

features = base_features + form_features

# Training and testing sets
X_train = train_df[features]
y_train = train_df['Winner']
X_test = test_df[features]
y_test = test_df['Winner']

# Random Forest model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Prediction with division-aware post-processing
def predict_with_division_rules(model, X, division_gap_threshold=3):
    probs = model.predict_proba(X)
    y_pred_raw = np.argmax(probs, axis=1)

    y_pred = []
    division_gaps = X['AbsoluteDivisionGap'].values

    for i, (pred, gap) in enumerate(zip(y_pred_raw, division_gaps)):
        if gap >= division_gap_threshold:
            if probs[i][0] > 0.7:
                y_pred.append(0)
            elif probs[i][2] > 0.7:
                y_pred.append(2)
            else:
                y_pred.append(0 if X.iloc[i]['DivisionGap'] < 0 else 2)
        else:
            y_pred.append(0 if pred == 1 and probs[i][0] > probs[i][2] else (2 if pred == 1 else pred))

    return np.array(y_pred), probs

y_pred, probs = predict_with_division_rules(model, X_test)

# Evaluation
results_df = test_df[['Date', 'Home', 'Away', 'Winner', 'DivisionGap', 'AbsoluteDivisionGap']].copy()
results_df['Predicted'] = y_pred
results_df['Correct'] = results_df['Winner'] == results_df['Predicted']
results_df['Actual Outcome'] = results_df['Winner'].map({0: 'Away Win', 2: 'Home Win'})
results_df['Predicted Outcome'] = results_df['Predicted'].map({0: 'Away Win', 2: 'Home Win'})

# Add probabilities
results_df[['Away Win Prob', 'Draw Prob', 'Home Win Prob']] = (probs * 100).round(1)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracyFormatted = accuracy*100

if __name__ =='__main__':
    print(f"\nRandom Forest Accuracy on FA Cup test set: {accuracy * 100:.2f}%\n")

    # Classification report
    print("Overall Classification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 2],
        target_names=['Away Win', 'Home Win'],
        zero_division=0
    ))

    # Accuracy by division gap
    print("\nPerformance by Division Gap:")
    for gap_range in [(0, 1), (2, 3), (4, 6)]:
        mask = results_df['AbsoluteDivisionGap'].between(*gap_range)
        if mask.any():
            gap_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"Division gap {gap_range[0]}-{gap_range[1]}: {gap_acc:.2%} accuracy ({mask.sum()} matches)")

    # Confusion matrix
    unique_classes = np.unique(y_test)
    print(f"\nClasses present in test data: {unique_classes}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=unique_classes))

    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = [features[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(len(features)), importances[indices], align='center')
    plt.xticks(range(len(features)), feature_names, rotation=45)
    plt.tight_layout()
    plt.show()
