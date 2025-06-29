import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import base64
import io

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

# Train-test split for goals
X_train_goals = train_df[features]
X_test_goals = test_df[features]


# Train logistic regression model
model = LogisticRegression(
    class_weight='balanced',

    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# Home goals model
home_goals_model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
home_goals_model.fit(X_train_goals, train_df['HomeGoals'])

# Away goals model
away_goals_model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
away_goals_model.fit(X_train_goals, train_df['AwayGoals'])

# funciton to predict match with score
def predict_match_score_logistic(match_date_str, home_team, away_team,
                                 home_model=home_goals_model, away_model=away_goals_model,
                                 le=le_team, df_all=df):
    """
    Predict the scores for a match using logistic regression.
    """
    match_date = pd.to_datetime(match_date_str)
    df_all = df_all.copy()
    df_all['Date'] = pd.to_datetime(df_all['Date'])

    def get_division(team):
        matches = df_all[((df_all['Home'] == team) | (df_all['Away'] == team)) & (df_all['Date'] < match_date)]
        if matches.empty:
            print(f"⚠️ No division data for {team} before {match_date_str}")
            return np.nan
        latest = matches.sort_values(by='Date', ascending=False).iloc[0]
        return latest['HomeDivision'] if latest['Home'] == team else latest['AwayDivision']

    def get_form(team):
        matches = df_all[
            ((df_all['Home'] == team) | (df_all['Away'] == team)) & (df_all['Date'] < match_date)].sort_values(
            by='Date', ascending=False).head(5)
        wins = 0
        for _, row in matches.iterrows():
            if row['Winner'] == 2 and row['Home'] == team:
                wins += 1
            elif row['Winner'] == 0 and row['Away'] == team:
                wins += 1
        return wins

    def get_team_features(team, opponent):
        division = get_division(team)
        form = get_form(team)
        opponent_division = get_division(opponent)
        if np.isnan(division) or np.isnan(opponent_division):
            print(f"⚠️ Missing division data for {team} or {opponent}. Skipping prediction.")
            return None
        division_gap = opponent_division - division

        return {
            'HomeTeam_enc': le.transform([team])[0],
            'AwayTeam_enc': le.transform([opponent])[0],
            'HomeDivision': division,
            'AwayDivision': opponent_division,
            'NeutralVenue': 1,
            'DivisionGap': division_gap,
            'AbsoluteDivisionGap': abs(division_gap),
            'HomeLast5_Wins': form,
            'AwayLast5_Wins': get_form(opponent),
            'HomeFormWeighted': form / (abs(division_gap) + 1),
            'AwayFormWeighted': get_form(opponent) * (abs(division_gap) + 1)
        }

    # Generate features
    try:
        row = pd.DataFrame([{
            **get_team_features(home_team, away_team),
            **get_team_features(away_team, home_team)
        }])
    except ValueError as e:
        print(f"⚠️ Error generating features: {e}")
        return

    if row is None or row.empty:
        return

    # Predict outcome
    probs = model.predict_proba(row)
    home_score = round(probs[0][model.classes_ == 2][0] * 3)  # Approximation: scale probability to 0-3
    away_score = round(probs[0][model.classes_ == 0][0] * 3)  # Approximation: scale probability to 0-3

    # Print and return the prediction
    print(f"\n📅 {match_date_str}: {home_team} vs {away_team}")
    print(f"🔢 Predicted Score: {home_team} {home_score} - {away_score} {away_team}")
    return {'home_score': home_score, 'away_score': away_score, 'home_team': home_team, 'away_team': away_team}

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
    return {'win': home_win_prob, 'lose': away_win_prob, 'prediction': outcome_map[prediction]}

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
results_df = test_df[['Date', 'Home', 'Away', 'Winner','DivisionGap', 'AbsoluteDivisionGap']].copy()
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




def calculate_importance_scores(model, X_train, features):
    """
    Calculate feature importance scores based on scaled coefficients.

    Args:
        model: Trained logistic regression model.
        X_train: Training data used for scaling.
        features: List of feature names.

    Returns:
        pd.DataFrame: DataFrame containing feature names and importance scores.
    """
    # Get standard deviation of each feature from the training data
    feature_std = X_train.std(axis=0)

    # Calculate importance scores as |coefficient| * feature_std
    importance_scores = np.abs(model.coef_[0]) * feature_std

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance_scores
    }).sort_values(by='Importance', ascending=False)

    return importance_df



# Calculate importance scores
importance_df = calculate_importance_scores(model, X_train, features)

# Plot the feature importance
buf = io.BytesIO()
plt.figure(figsize=(12, 8))
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.xticks(range(len(features)), features, rotation=45)
plt.title("Logistic Regression Feature Importance (Scores)")
plt.tight_layout()

# save plot for import
plt.savefig(buf, format='png')
buf.seek(0)
plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
plt.close()

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

# Plot confusion matrix + encode confusion image as base64 to import
buf_cm = io.BytesIO()
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.savefig(buf_cm, format='png')
buf_cm.seek(0)
confusion_plot = base64.b64encode(buf_cm.getvalue()).decode('utf-8')
plt.close()

# function to get division gap variable
def show_division_gap(results_df=results_df):
    results = []  # List to store the results
    for gap_range in [(0, 1), (2, 3), (4, 6)]:
        mask = results_df['AbsoluteDivisionGap'].between(*gap_range)
        if mask.any():
            gap_acc = accuracy_score(y_test[mask], y_pred[mask])
            match_count = mask.sum()
            results.append({
                'gap_range': f"{gap_range[0]}-{gap_range[1]}",
                'accuracy': gap_acc,
                'matches': match_count
            })
    return results

# Division gap variable
division_gap=show_division_gap(results_df)

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

    # predictions
    predict_match_logistic("2024-05-24", "Manchester City", "Manchester Utd")
    predict_match_score_logistic("2024-05-24", "Manchester City", "Manchester Utd")

    # Plot the feature importance
    plt.figure(figsize=(12, 8))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.xticks(range(len(features)), features, rotation=45)
    plt.title("Logistic Regression Feature Importance (Scores)")
    plt.tight_layout()
    plt.show()
    plt.close()