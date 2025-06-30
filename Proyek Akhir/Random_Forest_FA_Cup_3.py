import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import base64
import io
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

def split_draws_into_weighted_wins(df):
    """
    Convert draws into synthetic rows: one for home win, one for away win, weighted by form. enables the multiverse
    """
    win_df = df[df['Winner'].isin([0, 2])].copy()
    win_df['sample_weight'] = 1.0

    draw_df = df[df['Winner'] == 1].copy()
    synthetic_rows = []

    for _, row in draw_df.iterrows():
        home_score = row['HomeFormWeighted']
        away_score = row['AwayFormWeighted']
        total = home_score + away_score

        if total == 0:
            continue  # Skip if no signal, learned my lesson from before lol

        home_weight = home_score / total
        away_weight = away_score / total

        home_row = row.copy()
        home_row['Winner'] = 2
        home_row['sample_weight'] = home_weight
        synthetic_rows.append(home_row)

        away_row = row.copy()
        away_row['Winner'] = 0
        away_row['sample_weight'] = away_weight
        synthetic_rows.append(away_row)

    synthetic_df = pd.DataFrame(synthetic_rows)
    combined_df = pd.concat([win_df, synthetic_df], ignore_index=True)
    return combined_df

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

train_df = split_draws_into_weighted_wins(train_df)
train_df['Winner'] = train_df['Winner'].replace({2: 1})  # Home win becomes 1 (for binary classification)
X_train = train_df[features]
y_train = train_df['Winner']
sample_weights = train_df['sample_weight']
test_df['Winner'] = test_df['Winner'].replace({2: 1})
X_test = test_df[features]
y_test = test_df['Winner']

# training and testing for score prediction
X_train_scores = train_df[features]
y_train_home_goals = train_df['HomeGoals']  # Home goals as target
y_train_away_goals = train_df['AwayGoals']  # Away goals as target

X_test_scores = test_df[features]
y_test_home_goals = test_df['HomeGoals']  # Home goals in test set
y_test_away_goals = test_df['AwayGoals']  # Away goals in test set


model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train, sample_weight=sample_weights)


# Train home goals model
home_goals_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    random_state=42
)
home_goals_model.fit(X_train_scores, y_train_home_goals)

# Train away goals model
away_goals_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    random_state=42
)

def evaluate_goal_predictions(actual_home, actual_away, pred_home, pred_away):
    """Comprehensive evaluation of goal predictions"""
    rmse_home = np.sqrt(mean_squared_error(actual_home, pred_home))
    mae_home = mean_absolute_error(actual_home, pred_home)

    rmse_away = np.sqrt(mean_squared_error(actual_away, pred_away))
    mae_away = mean_absolute_error(actual_away, pred_away)

    # Directional accuracy (home scores more/less than away)
    actual_direction = (actual_home > actual_away).astype(int)
    pred_direction = (pred_home > pred_away).astype(int)
    direction_acc = accuracy_score(actual_direction, pred_direction)

    # Exact score accuracy
    exact_score = np.sum((actual_home == pred_home) & (actual_away == pred_away))
    exact_accuracy = exact_score / len(actual_home)

    # Within 1 goal accuracy
    home_within_1 = np.abs(actual_home - pred_home) <= 1
    away_within_1 = np.abs(actual_away - pred_away) <= 1
    within_1_acc = np.sum(home_within_1 & away_within_1) / len(actual_home)

    return {
        'Home_RMSE': rmse_home,
        'Home_MAE': mae_home,
        'Away_RMSE': rmse_away,
        'Away_MAE': mae_away,
        'Direction_Accuracy': direction_acc,
        'Exact_Accuracy': exact_accuracy,
        'Within_1_Goal_Accuracy': within_1_acc
    }

# poisson to make metrics more realistic
def poisson_adjust_predictions(predictions):
    predictions = np.where(predictions < 0, 0, predictions)
    predictions = np.round(predictions)

    adjusted = []
    for pred in predictions:
        if pred > 3:  # For high predictions, add randomness
            adj = np.random.poisson(pred * 0.9)  # Slightly reduce high predictions
        else:
            adj = pred
        adjusted.append(min(adj, 6))
    return np.array(adjusted)

# Train Random Forest for score prediction
rf_home_goal_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
rf_home_goal_model.fit(X_train, y_train_home_goals)
rf_away_goal_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
rf_away_goal_model.fit(X_train, y_train_away_goals)
# Predict scores (kelanjutan dari line 93)
rf_pred_home_goals = rf_home_goal_model.predict(X_test)
rf_pred_away_goals = rf_away_goal_model.predict(X_test)

# Adjust predictions
rf_adjusted_home_goals = poisson_adjust_predictions(rf_pred_home_goals)
rf_adjusted_away_goals = poisson_adjust_predictions(rf_pred_away_goals)

away_goals_model.fit(X_train_scores, y_train_away_goals)
# Evaluate goal predictions
rf_goal_eval = evaluate_goal_predictions(
    y_test_home_goals.values, y_test_away_goals.values,
    rf_adjusted_home_goals, rf_adjusted_away_goals
)

# retrieve the evaluation as a dictionary for inject
stats_goal = {'home_RMSE': rf_goal_eval['Home_RMSE'], 'away_RMSE': rf_goal_eval['Away_RMSE'],
                  'home_MAE': rf_goal_eval['Home_MAE'], 'away_MAE': rf_goal_eval['Away_MAE'],
                  'direction_accuracy': rf_goal_eval['Direction_Accuracy'],
                  'exact_accuracy': rf_goal_eval['Exact_Accuracy'],
                  '1goal': rf_goal_eval['Within_1_Goal_Accuracy']}
def predict_match_score(match_date_str, home_team, away_team,
                        home_model=home_goals_model, away_model=away_goals_model,
                        le=le_team, df_all=df):
    """
    Predict the scores for a match.
    """
    match_date = pd.to_datetime(match_date_str)
    df_all = df_all.copy()
    df_all['Date'] = pd.to_datetime(df_all['Date'])

    def get_division(team):
        matches = df_all[((df_all['Home'] == team) | (df_all['Away'] == team)) & (df_all['Date'] < match_date)]
        if matches.empty:
            print(f"⚠ No division data for {team} before {match_date_str}")
            return np.nan
        latest = matches.sort_values(by='Date', ascending=False).iloc[0]
        return latest['HomeDivision'] if latest['Home'] == team else latest['AwayDivision']

    def get_form(team):
        matches = df_all[((df_all['Home'] == team) | (df_all['Away'] == team)) & (df_all['Date'] < match_date)].sort_values(by='Date', ascending=False).head(5)
        wins = 0
        for _, row in matches.iterrows():
            if row['Winner'] == 2 and row['Home'] == team:
                wins += 1
            elif row['Winner'] == 0 and row['Away'] == team:
                wins += 1
        return wins

    def get_team_features(team, opponent, match_date):
        division = get_division(team)
        form = get_form(team)
        opponent_division = get_division(opponent)
        if np.isnan(division) or np.isnan(opponent_division):
            print(f"⚠ Missing division data for {team} or {opponent}. Skipping prediction.")
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
        home_features = pd.DataFrame([get_team_features(home_team, away_team, match_date)])
        away_features = pd.DataFrame([get_team_features(away_team, home_team, match_date)])
    except ValueError as e:
        print(f"⚠ Error generating features: {e}")
        return

    if home_features is None or away_features is None:
        return

    # Predict scores
    home_score = round(home_model.predict(home_features)[0])
    away_score = round(away_model.predict(away_features)[0])

    print(f"\n📅 {match_date_str}: {home_team} vs {away_team}")
    print(f"🔢 Predicted Score: {home_team} {home_score:.1f} - {away_score:.1f} {away_team}")
    return {'home_score':home_score, 'away_score':away_score, 'home_team':home_team, 'away_team':away_team}

# Function to calculate confidence
def calculate_confidence(probs):
    """
    Calculate the confidence of predictions based on the highest probability.

    Args:
        probs (numpy.ndarray): Array of predicted probabilities from the model.

    Returns:
        numpy.ndarray: Array of confidence values corresponding to predictions.
    """
    return probs.max(axis=1)


# Prediction with division-aware post-processing and confidence calculation
def predict_with_confidence(model, X, division_gap_threshold=3):
    """
    Predict outcomes with division-aware rules and calculate confidence.

    Args:
        model: Trained RandomForestClassifier model.
        X (pd.DataFrame): Test dataset.
        division_gap_threshold (int): Threshold for applying division-aware rules.

    Returns:
        tuple: Predicted labels, probabilities, and confidence values.
    """
def predict_with_confidence(model, X):
    probs = model.predict_proba(X)
    y_pred_raw = np.argmax(probs, axis=1)
    y_pred = []
    division_gaps = X['AbsoluteDivisionGap'].values
    confidence_scores = []

    for i, pred in enumerate(y_pred_raw):
        away_prob = probs[i][0]
        home_prob = probs[i][1]

        # If division gap logic or override is needed, you can add here

        y_pred.append(pred)
        confidence_scores.append(max(away_prob, home_prob) * 100)

    return np.array(y_pred), probs, confidence_scores


def predict_match_rf(match_date_str, home_team, away_team, model=model, le=le_team, df_all=df):
    import warnings
    warnings.filterwarnings("ignore")

    match_date = pd.to_datetime(match_date_str)
    df = df_all.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    teams_set = set(df['Home']).union(set(df['Away']))
    if home_team not in teams_set or away_team not in teams_set:
        print("⚠ One or both teams not found in dataset.")
        return

    def get_division(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)]
        if matches.empty:
            print(f"⚠ No division data for {team} before {match_date_str}")
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
        print("⚠ Error encoding teams — likely unseen label.")
        return

    # Predict
    probs = model.predict_proba(row)
    prediction = model.predict(row)[0]
    confidence = np.max(probs) * 100

    outcome_map = {0: 'Away Win', 1: 'Home Win'}
    home_win_prob = probs[0][model.classes_ == 1][0] * 100
    away_win_prob = probs[0][model.classes_ == 0][0] * 100

    print(f"\n📅 {match_date_str}: {home_team} vs {away_team}")
    print(f"🌲 Predicted: {outcome_map[prediction]} ({confidence:.2f}% confidence)")
    print(f"📈 Home form: {home_form}/5 | 📉 Away form: {away_form}/5")
    print(f"🔢 Divisions: {home_team} (D{home_div}) vs {away_team} (D{away_div})")
    print(f"📊 Probabilities: Home Win: {home_win_prob:.1f}%, Away Win: {away_win_prob:.1f}%")
    home_win_prob = probs[0][model.classes_ == 1][0] * 100
    away_win_prob = probs[0][model.classes_ == 0][0] * 100

    print(round(home_win_prob, 2))
    print( round(away_win_prob, 2))
    print(outcome_map[prediction])
    print(home_div)
    print(away_div)
    return {
        'win': round(home_win_prob, 2),
        'lose': round(away_win_prob, 2),
        'prediction': outcome_map[prediction],
        'home_division': home_div, 'away_division': away_div
    }


# Use the updated function for prediction and confidence
y_pred, probs, confidence = predict_with_confidence(model, X_test)



# Evaluation
results_df = test_df[['Date', 'Home', 'Away', 'Winner', 'DivisionGap', 'AbsoluteDivisionGap']].copy()
results_df['Predicted'] = y_pred
results_df['Correct'] = results_df['Winner'] == results_df['Predicted']
results_df['Actual Outcome'] = results_df['Winner'].map({0: 'Away Win', 1: 'Home Win'})
results_df['Predicted Outcome'] = results_df['Predicted'].map({0: 'Away Win', 1: 'Home Win'})
# Add confidence to the results dataframe
results_df['Confidence'] = np.round(np.array(confidence), 2)


# Add probabilities
results_df[['Away Win Prob', 'Home Win Prob']] = (probs * 100).round(1)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracyFormatted = accuracy*100
confidence_df=results_df[['Date', 'Home', 'Away', 'Predicted Outcome', 'Confidence', 'Actual Outcome']].head()

# Generate classification report as a dictionary
report = classification_report(
    y_test, y_pred, labels=[0, 1],
    target_names=['Away Win', 'Home Win'], zero_division=0, output_dict=True
)

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
# Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [features[i] for i in indices]

# make plot
buf = io.BytesIO()  # Create a buffer for the plot
plt.figure(figsize=(12, 8))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(len(features)), importances[indices], align='center')
plt.xticks(range(len(features)), feature_names, rotation=45)
plt.tight_layout()

# save plot for import
plt.savefig(buf, format='png')
buf.seek(0)
plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
plt.close()

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Plot confusion matrix + encode confusion image as base64 to import
buf_cm = io.BytesIO()
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.savefig(buf_cm, format='png')
buf_cm.seek(0)
confusion_plot = base64.b64encode(buf_cm.getvalue()).decode('utf-8')
plt.close()

if __name__ == '__main__':
    print(f"\nRandom Forest Accuracy on FA Cup test set: {accuracy * 100:.2f}%\n")

    # Classification report
    print("Overall Classification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
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

    # confidence 5
    print(results_df[['Date', 'Home', 'Away', 'Predicted Outcome', 'Confidence']].head())

    # predictions
    predict_match_rf("2024-05-24", "Manchester City", "Manchester Utd")
    predict_match_score("2024-05-24", "Manchester City", "Manchester Utd")

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(len(features)), importances[indices], align='center')
    plt.xticks(range(len(features)), feature_names, rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()