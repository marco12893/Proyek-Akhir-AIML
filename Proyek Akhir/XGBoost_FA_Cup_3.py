import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
from xgboost import plot_importance

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

# Add goal columns if not already present
if 'HomeGoals' not in df.columns or 'AwayGoals' not in df.columns:
    df['HomeGoals'] = df['FTHG']  # replace if needed
    df['AwayGoals'] = df['FTAG']  # replace if needed

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

# X and y for classification
X_train = train_df[features]
y_train = train_df['Winner']
X_test = test_df[features]
y_test = test_df['Winner']

# Train classification model
model = XGBClassifier(
    eval_metric='logloss',
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=1000
)
model.fit(X_train, y_train)

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



# Prediction with division-aware logic
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

# Use the function to calculate confidence
confidence = get_prediction_confidence(probs)

# Add confidence to the results DataFrame

# Result DataFrame
results_df = test_df[['Date', 'Home', 'Away', 'Winner', 'DivisionGap', 'AbsoluteDivisionGap']].copy()
results_df['Predicted'] = y_pred
results_df['Correct'] = results_df['Winner'] == results_df['Predicted']
results_df['Actual Outcome'] = results_df['Winner'].map({0: 'Away Win', 2: 'Home Win'})
results_df['Predicted Outcome'] = results_df['Predicted'].map({0: 'Away Win', 2: 'Home Win'})
results_df[['Home Win Prob', 'Draw Prob', 'Away Win Prob']] = (probs * 100).round(1)
results_df['Confidence (%)'] = confidence.round(2)

# Evaluate classification
accuracy = accuracy_score(y_test, y_pred)
accuracyFormatted = accuracy*100
confidence_df=results_df[['Date', 'Home', 'Away', 'Predicted Outcome', 'Confidence (%)', 'Actual Outcome']].head()

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

if __name__ =='__main__':
    print(f"\nXGBoost Accuracy on FA Cup test set: {accuracy * 100:.2f}%\n")
    print("Overall Classification Report:")
    print(classification_report(y_test, y_pred, labels=[0, 2], target_names=['Away Win', 'Home Win'], zero_division=0))

    print("\nPerformance by Division Gap:")
    for gap_range in [(0, 1), (2, 3), (4, 6)]:
        mask = results_df['AbsoluteDivisionGap'].between(*gap_range)
        if mask.any():
            gap_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"Division gap {gap_range[0]}-{gap_range[1]}: {gap_acc:.2%} accuracy ({mask.sum()} matches)")

    print(f"Classes present in test data: {np.unique(y_test)}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=np.unique(y_test)))

    # Display confidence levels
    print("\nTop 5 Predictions with Confidence Levels:")
    print(results_df[['Date', 'Home', 'Away', 'Predicted Outcome', 'Confidence (%)']].head())

    # Feature importance plot
    plt.figure(figsize=(12, 8))
    plot_importance(model, max_num_features=15, importance_type='weight')
    plt.title('Feature Importance (Weight)')
    plt.show()


# # -------------------------------
# # 🎯 Score Prediction Starts Here
# # -------------------------------
# y_train_home = train_df['HomeGoals']
# y_train_away = train_df['AwayGoals']
# y_test_home = test_df['HomeGoals']
# y_test_away = test_df['AwayGoals']
# X_train_score = train_df[features]
# X_test_score = test_df[features]
#
# home_goal_model = XGBRegressor(
#     learning_rate=0.01,
#     max_depth=4,
#     n_estimators=200,
#     subsample=0.8,
#     colsample_bytree=0.8
# )
# away_goal_model = XGBRegressor(
#     learning_rate=0.01,
#     max_depth=4,
#     n_estimators=200,
#     subsample=0.8,
#     colsample_bytree=0.8
# )
#
# home_goal_model.fit(X_train_score, y_train_home)
# away_goal_model.fit(X_train_score, y_train_away)
#
# pred_home_goals = home_goal_model.predict(X_test_score)
# pred_away_goals = away_goal_model.predict(X_test_score)
#
# rounded_home_goals = np.round(pred_home_goals).astype(int)
# rounded_away_goals = np.round(pred_away_goals).astype(int)
#
# # Add to result DataFrame
# results_df['Predicted Home Goals'] = rounded_home_goals
# results_df['Predicted Away Goals'] = rounded_away_goals
# results_df['Actual Home Goals'] = y_test_home.values
# results_df['Actual Away Goals'] = y_test_away.values
#
# # Evaluation
# home_rmse = mean_squared_error(y_test_home, pred_home_goals, squared=False)
# away_rmse = mean_squared_error(y_test_away, pred_away_goals, squared=False)
# home_mae = mean_absolute_error(y_test_home, pred_home_goals)
# away_mae = mean_absolute_error(y_test_away, pred_away_goals)
#
# print(f"\nScore Prediction Evaluation:")
# print(f"Home Goals - RMSE: {home_rmse:.3f}, MAE: {home_mae:.3f}")
# print(f"Away Goals - RMSE: {away_rmse:.3f}, MAE: {away_mae:.3f}")
#
# exact_score_correct = (
#     (rounded_home_goals == y_test_home.values) &
#     (rounded_away_goals == y_test_away.values)
# ).sum()
# exact_score_accuracy = exact_score_correct / len(y_test_home)
# print(f"Exact Score Prediction Accuracy: {exact_score_accuracy:.2%} ({exact_score_correct}/{len(y_test_home)})")
