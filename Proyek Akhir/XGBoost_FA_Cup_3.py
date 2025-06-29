import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
from xgboost import plot_importance
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

df = create_advanced_features(df)
def split_draws_into_weighted_wins(df):
    """
    buat draw jadi weight pake rumus ini:
    - One for Away Win with weight = AwayFormWeighted / (total)
    - One for Home Win with weight = HomeFormWeighted / (total)
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
            continue  # skip if no no data, fun fact ini buat ak pusing banget kok sering problem ternyata ada beberapa data yang just gaonok data lol

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

# Add goal columns if not already present
if 'HomeGoals' not in df.columns or 'AwayGoals' not in df.columns:
    df['HomeGoals'] = df['FTHG']  # replace if needed
    df['AwayGoals'] = df['FTAG']  # replace if needed

# Filter training and testing data
train_df = df[~((df['Type'] == 'FA Cup') & (df['Season'] == 2023))]
# Filter test data and convert Winner = 2 (Home) to 1 (binary label)
test_df = df[(df['Type'] == 'FA Cup') & (df['Season'] == 2023) & (df['Winner'] != 1)].copy()
test_df['Winner'] = test_df['Winner'].replace({2: 1})


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

# form_features = [
#     'HomeLast5_GoalsFor', 'AwayLast5_GoalsFor',
#     'HomeLast5_GoalsAgainst', 'AwayLast5_GoalsAgainst',
#     'HomeLast5_Wins', 'AwayLast5_Wins',
#     'HomeLast5_Losses', 'AwayLast5_Losses',
#     'HomeLast5_Draws', 'AwayLast5_Draws',
#     'HomeLast5_CleanSheets', 'AwayLast5_CleanSheets',
#     'HomeFormWeighted', 'AwayFormWeighted'
# ]

features = base_features + form_features

train_df = split_draws_into_weighted_wins(train_df)

# Relabel 0 (Away) -> 0, 2 (Home) -> 1
train_df['Winner'] = train_df['Winner'].replace({2: 1})
X_train = train_df[features]
y_train = train_df['Winner']
sample_weights = train_df['sample_weight']
X_test = test_df[features]
y_test = test_df['Winner']



# Train classification model
model = XGBClassifier(
    objective='binary:logistic',  # Only 2 classes
    eval_metric='logloss',
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=1000
)

model.fit(X_train, y_train, sample_weight=sample_weights)


# Function to calculate confidence levels
def get_prediction_confidence(probs):
    """
    Calculate the confidence for predictions.
    Parameters:
    probs (numpy array): Array of probabilities for each class.
    Returns:
    numpy array: Array of confidence levels for each prediction.
    """
    confidence = np.max(probs, axis=1) * 100  # Max probability as percentage
    return confidence



def redistribute_draw_probabilities(probs): #depracted, cuman tak keep buat dokum + progress reportn nanti
    """
  function baru untuk nge bagi draw prob nya
    """
    redistributed = []
    for row in probs:
        away, draw, home = row
        total_non_draw = home + away
        if total_non_draw == 0:
            # supaya no div by zero tom foolery
            new_home = new_away = 0.5
        else:
            home_weight = home / total_non_draw
            away_weight = away / total_non_draw
            redistributed_draw_home = draw * home_weight
            redistributed_draw_away = draw * away_weight
            new_home = home + redistributed_draw_home
            new_away = away + redistributed_draw_away

        redistributed.append([new_away * 100, new_home * 100])  # Return as percentages
    return np.array(redistributed)


# Prediction with division-aware logic
def predict_with_division_rules(model, X, division_gap_threshold=3):
    probs = model.predict_proba(X)
    y_pred_raw = np.argmax(probs, axis=1)
    y_pred = []
    division_gaps = X['AbsoluteDivisionGap'].values

    for i, (pred, gap) in enumerate(zip(y_pred_raw, division_gaps)):
        # Binary classifier: 0 = Away, 1 = Home
        away_prob = probs[i][0]
        home_prob = probs[i][1]

        if gap >= division_gap_threshold:
            if away_prob > 0.7:
                y_pred.append(0)
            elif home_prob > 0.7:
                y_pred.append(1)
            else:
                y_pred.append(0 if X.iloc[i]['DivisionGap'] > 0 else 1)
        else:
            y_pred.append(pred)

    return np.array(y_pred), probs


def predict_match(match_date_str, home_team, away_team, model=model, le=le_team, df_all=df):
    import warnings
    warnings.filterwarnings("ignore")

    # Convert string date to datetime
    match_date = pd.to_datetime(match_date_str)
    df = df_all.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Check if teams exist
    teams_set = set(df['Home']).union(set(df['Away']))
    if home_team not in teams_set or away_team not in teams_set:
        print(f"⚠ One or both teams not found in the dataset.")
        return

    # Get the most recent division for both teams before match date
    def get_division(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)]
        if matches.empty:
            print(f"⚠ No matches found for {team} before {match_date_str}")
            return np.nan
        latest = matches.sort_values(by='Date', ascending=False).iloc[0]
        if latest['Home'] == team:
            return latest['HomeDivision']
        else:
            return latest['AwayDivision']

    home_div = get_division(home_team)
    away_div = get_division(away_team)

    # Get last 5 results
    def get_form(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)].sort_values(by='Date',ascending=False).head(5)
        if matches.empty:
            print(f"⚠ No recent matches for {team}")
            return 0
        wins = 0
        for _, row in matches.iterrows():
            if row['Winner'] == 2 and row['Home'] == team:
                wins += 1
            elif row['Winner'] == 0 and row['Away'] == team:
                wins += 1
        return wins

    home_form = get_form(home_team)
    away_form = get_form(away_team)

    # Build feature row
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
        print("⚠ Failed to encode team names — likely due to unseen label.")
        return

    # Predict
    probs = model.predict_proba(row)
    confidence = np.max(probs) * 100
    predicted = np.argmax(probs)

    outcome = {0: 'Away Win', 1: 'Home Win'}
    print(f"📅 {match_date_str}: {home_team} vs {away_team}")
    print(f"🏆 Predicted: {outcome.get(predicted)} ({confidence:.2f}% confidence)")
    print(f"📈 Home form: {home_form} wins in last 5")
    print(f"📉 Away form: {away_form} wins in last 5")
    print(f"🔢 Divisions: {home_team} (D{home_div}) vs {away_team} (D{away_div})")
    home_prob = probs[0][1] * 100
    away_prob = probs[0][0] * 100
    print(f"📊 Probabilities: Home Win: {home_prob:.1f}%, Away Win: {away_prob:.1f}%")
    return {'win': home_prob, 'lose': away_prob}


y_pred, probs = predict_with_division_rules(model, X_test)


confidence = np.max(probs, axis=1)


# Result DataFrame
results_df = test_df[['Date', 'Home', 'Away', 'Winner', 'DivisionGap', 'AbsoluteDivisionGap']].copy()
results_df['Predicted'] = y_pred
results_df['Correct'] = results_df['Winner'] == results_df['Predicted']
results_df['Actual Outcome'] = results_df['Winner'].map({0: 'Away Win', 1: 'Home Win'})
results_df['Predicted Outcome'] = results_df['Predicted'].map({0: 'Away Win', 1: 'Home Win'})
results_df['Away Win Prob'] = probs[:, 0] * 100
results_df['Home Win Prob'] = probs[:, 1] * 100
results_df['Confidence (%)'] = confidence.round(2)


# Evaluate classification
accuracy = accuracy_score(y_test, y_pred)
accuracyFormatted = accuracy * 100


# Fix confidence to be percentage
results_df['Confidence (%)'] = results_df['Confidence (%)'] * 100

# Extract top 5
confidence_df = results_df[['Date', 'Home', 'Away', 'Predicted Outcome', 'Confidence (%)', 'Actual Outcome']].head()

# Generate classification report as a dictionary
report = classification_report(y_test, y_pred, labels=[0, 1], target_names=['Away Win', 'Home Win'], zero_division=0, output_dict=True)


# Extract metrics for each class
away_win_metrics = report['Away Win']  # Dict for 'Away Win'
home_win_metrics = report['Home Win']  # Dict for 'Home Win'

# Example: precision, recall, and F1-score for 'Away Win'
away_precision = away_win_metrics['precision']
away_recall = away_win_metrics['recall']
away_f1 = away_win_metrics['f1-score']
print("Unique values in y_test:", np.unique(y_test))
print("Unique values in y_pred:", np.unique(y_pred))

# Example: accuracy
accuracy = report['accuracy']

# Prepare data for HTML
metrics_data = [
    {'Class': 'Away Win', 'Precision': away_precision, 'Recall': away_recall, 'F1-Score': away_f1},
    {'Class': 'Home Win', 'Precision': home_win_metrics['precision'], 'Recall': home_win_metrics['recall'], 'F1-Score': home_win_metrics['f1-score']},
    {'Class': 'Overall Accuracy', 'Precision': '', 'Recall': '', 'F1-Score': accuracy},
]

# Generate classification report as a dictionary
report = classification_report(y_test, y_pred, labels=[0, 2], target_names=['Away Win', 'Home Win'], zero_division=0,
                               output_dict=True)

# -------------------------------
# 🎯 Goal Prediction Functions
# -------------------------------

def create_goal_features(df):
    """Create features specifically optimized for goal prediction"""
    df = df.copy()

    # Calculate division average goals from your data
    division_avg_goals = df.groupby('HomeDivision')['HomeGoals'].mean().to_dict()

    # Team offensive/defensive strength metrics
    df['HomeOffensiveStrength'] = df['HomeLast5_GoalsFor'] / (df['HomeLast5_GoalsAgainst'] + 1)
    df['AwayOffensiveStrength'] = df['AwayLast5_GoalsFor'] / (df['AwayLast5_GoalsAgainst'] + 1)

    df['HomeDefensiveWeakness'] = df['HomeLast5_GoalsAgainst'] / (df['HomeLast5_GoalsFor'] + 1)
    df['AwayDefensiveWeakness'] = df['AwayLast5_GoalsAgainst'] / (df['AwayLast5_GoalsFor'] + 1)

    # Recent form metrics
    df['HomeGoalMomentum'] = df['HomeLast5_GoalsFor'] - df['HomeLast5_GoalsAgainst']
    df['AwayGoalMomentum'] = df['AwayLast5_GoalsFor'] - df['AwayLast5_GoalsAgainst']

    # Division-based adjustments
    df['HomeDivisionFactor'] = df['HomeDivision'].map(division_avg_goals)
    df['AwayDivisionFactor'] = df['AwayDivision'].map(division_avg_goals)

    return df

# THIS MIGHT DESTROY THE ENTIRE CODEBASE (score prediction)
train_df = create_goal_features(train_df)
test_df = create_goal_features(test_df)

goal_features = [
    'HomeTeam_enc', 'AwayTeam_enc',
    'HomeDivision', 'AwayDivision',
    'AbsoluteDivisionGap',
    'HomeOffensiveStrength', 'AwayOffensiveStrength',
    'HomeDefensiveWeakness', 'AwayDefensiveWeakness',
    'HomeGoalMomentum', 'AwayGoalMomentum',
    'HomeDivisionFactor', 'AwayDivisionFactor',
    'HomeLast5_CleanSheets', 'AwayLast5_CleanSheets'
]

X_train_goals = train_df[goal_features]
X_test_goals = test_df[goal_features]
y_train_home = train_df['HomeGoals']
y_train_away = train_df['AwayGoals']
y_test_home = test_df['HomeGoals']
y_test_away = test_df['AwayGoals']

# Train goal prediction models
goal_model_params = {
    'learning_rate': 0.05,
    'max_depth': 3,
    'n_estimators': 300,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'count:poisson',
    'n_jobs': -1,
    'random_state': 42,
    'eval_metric': 'rmse'
}
# home goal model for score prediction
home_goal_model = XGBRegressor(**goal_model_params)
home_goal_model.fit(X_train_goals, y_train_home)

# away goal model for score prediction
away_goal_model = XGBRegressor(**goal_model_params)
away_goal_model.fit(X_train_goals, y_train_away)

# Score prediction function
def predict_match_score(match_date_str, home_team, away_team, home_goal_model=home_goal_model, away_goal_model=away_goal_model, le=le_team, df_all=df):
    match_date = pd.to_datetime(match_date_str)
    df = df_all.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    def get_division(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)]
        if matches.empty:
            return np.nan
        latest = matches.sort_values(by='Date', ascending=False).iloc[0]
        return latest['HomeDivision'] if latest['Home'] == team else latest['AwayDivision']

    def get_clean_sheets(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)].sort_values(by='Date', ascending=False).head(5)
        if matches.empty:
            return 0
        clean_sheets = 0
        for _, row in matches.iterrows():
            if row['Home'] == team and row['AwayGoals'] == 0:
                clean_sheets += 1
            elif row['Away'] == team and row['HomeGoals'] == 0:
                clean_sheets += 1
        return clean_sheets

    home_div = get_division(home_team)
    away_div = get_division(away_team)
    home_clean_sheets = get_clean_sheets(home_team)
    away_clean_sheets = get_clean_sheets(away_team)

    row = pd.DataFrame([{
        'HomeTeam_enc': le.transform([home_team])[0],
        'AwayTeam_enc': le.transform([away_team])[0],
        'HomeDivision': home_div,
        'AwayDivision': away_div,
        'AbsoluteDivisionGap': abs(away_div - home_div),
        'HomeOffensiveStrength': 1,  # Replace with actual logic
        'AwayOffensiveStrength': 1,
        'HomeDefensiveWeakness': 1,
        'AwayDefensiveWeakness': 1,
        'HomeGoalMomentum': 0,
        'AwayGoalMomentum': 0,
        'HomeDivisionFactor': df[df['HomeDivision'] == home_div]['HomeGoals'].mean(),
        'AwayDivisionFactor': df[df['AwayDivision'] == away_div]['AwayGoals'].mean(),
        'HomeLast5_CleanSheets': home_clean_sheets,
        'AwayLast5_CleanSheets': away_clean_sheets
    }])

    pred_home_goals = round(home_goal_model.predict(row)[0])
    pred_away_goals = round(away_goal_model.predict(row)[0])

    # return
    return {'home_score': pred_home_goals, 'away_score': pred_away_goals, 'home_team': home_team, 'away_team': away_team, 'home_division': home_div, 'away_division': away_div}



def poisson_adjust_predictions(predictions):
    """Adjust predictions to better match real-world goal distributions"""
    predictions = np.where(predictions < 0, 0, predictions)  # No negative goals
    predictions = np.round(predictions)  # Goals are integers

    # Apply Poisson-inspired adjustments
    adjusted = []
    for pred in predictions:
        if pred > 3:  # For high predictions, add some randomness
            adj = np.random.poisson(pred * 0.9)  # Slightly reduce very high predictions
        else:
            adj = pred
        adjusted.append(min(adj, 6))  # Cap at 6 goals
    return np.array(adjusted)


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


def run_goal_prediction(train_df, test_df, results_df):
    """Run the complete goal prediction pipeline"""
    # Create goal-specific features
    train_df = create_goal_features(train_df)
    test_df = create_goal_features(test_df)

    # Goal prediction features
    goal_features = [
        'HomeTeam_enc', 'AwayTeam_enc',
        'HomeDivision', 'AwayDivision',
        'AbsoluteDivisionGap',
        'HomeOffensiveStrength', 'AwayOffensiveStrength',
        'HomeDefensiveWeakness', 'AwayDefensiveWeakness',
        'HomeGoalMomentum', 'AwayGoalMomentum',
        'HomeDivisionFactor', 'AwayDivisionFactor',
        'HomeLast5_CleanSheets', 'AwayLast5_CleanSheets'
    ]

    X_train_goals = train_df[goal_features]
    X_test_goals = test_df[goal_features]
    y_train_home = train_df['HomeGoals']
    y_train_away = train_df['AwayGoals']
    y_test_home = test_df['HomeGoals']
    y_test_away = test_df['AwayGoals']

    # Optimized model parameters
    goal_model_params = {
        'learning_rate': 0.05,
        'max_depth': 3,
        'n_estimators': 300,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'count:poisson',
        'n_jobs': -1,
        'random_state': 42,
        'eval_metric': 'rmse'  # For regression
    }

    # Initialize and train models
    print("\nTraining home goals model...")
    home_goal_model = XGBRegressor(**goal_model_params)
    home_goal_model.fit(X_train_goals, y_train_home)

    print("Training away goals model...")
    away_goal_model = XGBRegressor(**goal_model_params)
    away_goal_model.fit(X_train_goals, y_train_away)

    # Make predictions
    pred_home_goals = home_goal_model.predict(X_test_goals)
    pred_away_goals = away_goal_model.predict(X_test_goals)

    # Adjust predictions
    adjusted_home_goals = poisson_adjust_predictions(pred_home_goals)
    adjusted_away_goals = poisson_adjust_predictions(pred_away_goals)

    # Add to results
    results_df['Predicted Home Goals'] = adjusted_home_goals
    results_df['Predicted Away Goals'] = adjusted_away_goals
    results_df['Actual Home Goals'] = y_test_home.values
    results_df['Actual Away Goals'] = y_test_away.values

    # Evaluate
    eval_results = evaluate_goal_predictions(
        y_test_home.values, y_test_away.values,
        adjusted_home_goals, adjusted_away_goals
    )

    print("\nEnhanced Goal Prediction Evaluation:")
    print(f"🏠 Home Goals - RMSE: {eval_results['Home_RMSE']:.3f}, MAE: {eval_results['Home_MAE']:.3f}")
    print(f"🛫 Away Goals - RMSE: {eval_results['Away_RMSE']:.3f}, MAE: {eval_results['Away_MAE']:.3f}")
    print(f"🧭 Direction Accuracy: {eval_results['Direction_Accuracy']:.2%}")
    print(f"🎯 Exact Score Accuracy: {eval_results['Exact_Accuracy']:.2%}")
    print(f"✅ Within 1 Goal Accuracy: {eval_results['Within_1_Goal_Accuracy']:.2%}")

    # Show feature importance
    plt.figure(figsize=(12, 8))
    plot_importance(home_goal_model, max_num_features=15)
    plt.title('Home Goal Prediction Feature Importance')
    plt.show()

    return results_df

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


# Feature importance plot for outcome prediction
plt.figure(figsize=(12, 8))
plot_importance(model, max_num_features=15, importance_type='weight')
plt.title('Outcome Prediction Feature Importance')


# encode image as base64 to import
buf = io.BytesIO()  # Create a buffer for the plot
plt.figure(figsize=(12, 8))
plot_importance(model, max_num_features=15, importance_type='weight')
plt.title('Outcome Prediction Feature Importance')
plt.tight_layout()
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

# # Convert to DataFrame for readability
# labels = np.unique(y_test)  # Use unique values as labels
# cm_df = pd.DataFrame(cm, index=labels, columns=labels)
#
# # Convert DataFrame to HTML table
# cm_html = cm_df.to_html(classes="table table-bordered table-hover", border=0)

# predict score
prediction = predict_match_score("2024-05-24", "Manchester City", "Manchester Utd")
pred_home_goals, pred_away_goals, home_team, away_team = prediction['home_score'], prediction['away_score'], prediction['home_team'], prediction['away_team']

if __name__ == '__main__':
    print(f"\nXGBoost Accuracy on FA Cup test set: {accuracy * 100:.2f}%\n")
    print("Overall Classification Report:")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Away Win', 'Home Win'], zero_division=0))

    print("\nPerformance by Division Gap:")
    for gap_range in [(0, 1), (2, 3), (4, 6)]:
        mask = results_df['AbsoluteDivisionGap'].between(*gap_range)
        if mask.any():
            gap_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"Division gap {gap_range[0]}-{gap_range[1]}: {gap_acc:.2%} accuracy ({mask.sum()} matches)")

    print(f"Classes present in test data: {np.unique(y_test)}")
    print("Confusion Matrix:")
    print(cm)

    # Display confidence levels
    print("\nTop 5 Predictions with Confidence Levels:")
    print(results_df[['Date', 'Home', 'Away', 'Predicted Outcome', 'Confidence (%)']].head(), "\n")
    predict_match("2024-05-24", "Manchester City", "Manchester Utd")
    print(f"Predicted Score: {home_team} {pred_home_goals:.1f} - {pred_away_goals:.1f} {away_team}")

    # Run goal prediction
    results_df = run_goal_prediction(train_df, test_df, results_df)

    # Show combined results
    print("\nCombined Predictions with Scores:")
    print(results_df[['Date', 'Home', 'Away', 'Predicted Outcome', 'Confidence (%)',
                      'Predicted Home Goals', 'Predicted Away Goals',
                      'Actual Outcome', 'Actual Home Goals', 'Actual Away Goals']].head(10))

    # Feature importance plot for outcome prediction
    plt.figure(figsize=(16, 10))
    plot_importance(model, max_num_features=15, importance_type='weight')
    plt.title('Outcome Prediction Feature Importance')
    plt.tight_layout()
    plt.show()