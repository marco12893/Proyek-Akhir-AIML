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

# Enhanced feature engineering
df['DivisionGap'] = df['AwayDivision'] - df['HomeDivision']
df['AbsoluteDivisionGap'] = abs(df['DivisionGap'])
df['HomeFormWeighted'] = df['HomeLast5_Wins'] / (df['AbsoluteDivisionGap'] + 1)
df['AwayFormWeighted'] = df['AwayLast5_Wins'] * (df['AbsoluteDivisionGap'] + 1)

# Train-test split: FA Cup 2023 as test set
train_df = df[~((df['Type'] == 'FA Cup') & (df['Season'] == 2023))]
test_df = df[(df['Type'] == 'FA Cup') & (df['Season'] == 2023)]

# Feature sets
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

# Prepare data
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


def calculate_confidence(y_pred, y_proba, class_labels):
    """
    Calculate confidence for each prediction.
    Args:
        y_pred (numpy.ndarray): Predicted classes.
        y_proba (numpy.ndarray): Probabilities for each class.
        class_labels (list): List of class labels corresponding to columns in y_proba.

    Returns:
        numpy.ndarray: Confidence values corresponding to predictions.
    """
    # Create a mapping of class labels to column indices
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}

    # Extract the probability corresponding to each predicted class
    return np.array([y_proba[i, label_to_index[pred]] for i, pred in enumerate(y_pred)])

def predict_match_logistic(match_date_str, home_team, away_team, model=model, le=le_team, df_all=df):
    import warnings
    warnings.filterwarnings("ignore")

    match_date = pd.to_datetime(match_date_str)
    df = df_all.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    teams_set = set(df['Home']).union(set(df['Away']))
    if home_team not in teams_set or away_team not in teams_set:
        print("⚠️ One or both teams not found in dataset.")
        return

    def get_division(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)]
        if matches.empty:
            print(f"⚠️ No division data for {team} before {match_date_str}")
            return np.nan
        latest = matches.sort_values(by='Date', ascending=False).iloc[0]
        return latest['HomeDivision'] if latest['Home'] == team else latest['AwayDivision']

    def get_form(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)].sort_values(by='Date', ascending=False).head(5)
        wins = 0
        for _, row in matches.iterrows():
            if row['Winner'] == 2 and row['Home'] == team:
                wins += 1
            elif row['Winner'] == 0 and row['Away'] == team:
                wins += 1
        return wins

    home_div = get_division(home_team)
    away_div = get_division(away_team)
    home_form = get_form(home_team)
    away_form = get_form(away_team)

    try:
        row = pd.DataFrame([{
            'HomeTeam_enc': le.transform([home_team])[0],
            'AwayTeam_enc': le.transform([away_team])[0],
            'HomeDivision': home_div,
            'AwayDivision': away_div,
            'NeutralVenue': 1,
            'DivisionGap': away_div - home_div,
            'AbsoluteDivisionGap': abs(away_div - home_div),
            'HomeLast5_Wins': home_form,
            'AwayLast5_Wins': away_form,
            'HomeFormWeighted': home_form / (abs(away_div - home_div) + 1),
            'AwayFormWeighted': away_form * (abs(away_div - home_div) + 1)
        }])
    except:
        print("⚠️ Error encoding teams — likely unseen label.")
        return

    # Predict
    probs = model.predict_proba(row)
    prediction = model.predict(row)[0]
    confidence = np.max(probs) * 100

    outcome_map = {0: 'Away Win', 2: 'Home Win'}
    home_win_prob = probs[0][model.classes_ == 2][0] * 100
    away_win_prob = probs[0][model.classes_ == 0][0] * 100

    print(f"\n📅 {match_date_str}: {home_team} vs {away_team}")
    print(f"🏆 Predicted: {outcome_map[prediction]} ({confidence:.2f}% confidence)")
    print(f"📈 Home form: {home_form}/5 | 📉 Away form: {away_form}/5")
    print(f"🔢 Divisions: {home_team} (D{home_div}) vs {away_team} (D{away_div})")
    print(f"📊 Probabilities → Home Win: {home_win_prob:.1f}%, Away Win: {away_win_prob:.1f}%")

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Get confidence values
class_labels = model.classes_  # Retrieve class labels from the model
confidence = calculate_confidence(y_pred, y_proba, class_labels)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
accuracyFormatted = accuracy * 100

# Results dataframe
results_df = test_df[['Date', 'Home', 'Away', 'Winner']].copy()
results_df['Predicted'] = y_pred
results_df['Confidence'] = (confidence * 100).round(1)
results_df['Home Loss Prob'] = (y_proba[:, 0] * 100).round(1)
results_df['Home Win Prob'] = (y_proba[:, 1] * 100).round(1)
results_df['Correct'] = results_df['Winner'] == results_df['Predicted']

confidence_df=results_df[['Date', 'Home', 'Away', 'Winner', 'Predicted', 'Confidence', 'Home Loss Prob','Home Win Prob','Correct']].head()

# Generate classification report as a dictionary
report = classification_report(y_test, y_pred, labels=[0, 2], target_names=['Away Win', 'Home Win'], zero_division=0, output_dict=True)

# Extract metrics for each class
away_win_metrics = report['Away Win']  # Dict for 'Away Win'
home_win_metrics = report['Home Win']  # Dict for 'Home Win'

# Example: precision, recall, and F1-score for 'Away Win'
away_precision = away_win_metrics['precision']
away_recall = away_win_metrics['recall']
away_f1 = away_win_metrics['f1-score']

# Example: accuracy
accuracy = report['accuracy']

# Prepare data for HTML
metrics_data = [
    {'Class': 'Away Win', 'Precision': away_precision, 'Recall': away_recall, 'F1-Score': away_f1},
    {'Class': 'Home Win', 'Precision': home_win_metrics['precision'], 'Recall': home_win_metrics['recall'], 'F1-Score': home_win_metrics['f1-score']},
    {'Class': 'Overall Accuracy', 'Precision': '', 'Recall': '', 'F1-Score': accuracy},
]

if __name__ == '__main__':
    print("\nLogistic Regression (All Features)")
    print("=" * 55)
    print(f"Accuracy: {accuracy:.2%}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 2]))

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 2],
        target_names=['Home Loss (0)', 'Home Win (2)']
    ))

    print("\nSample Predictions:")
    print(results_df.head(10))

    # Model coefficients
    print("\nModel Coefficients:")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    for feature_name, coef in zip(features, model.coef_[0]):
        print(f"{feature_name}: {coef:.4f}")
    predict_match_logistic("2024-05-24", "Manchester City", "Manchester Utd")