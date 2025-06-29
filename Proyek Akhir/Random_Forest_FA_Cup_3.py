import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
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

    confidence = calculate_confidence(probs)
    return np.array(y_pred), probs, confidence

def predict_match_rf(match_date_str, home_team, away_team, model=model, le=le_team, df_all=df):
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
    print(f"🌲 Predicted: {outcome_map[prediction]} ({confidence:.2f}% confidence)")
    print(f"📈 Home form: {home_form}/5 | 📉 Away form: {away_form}/5")
    print(f"🔢 Divisions: {home_team} (D{home_div}) vs {away_team} (D{away_div})")
    print(f"📊 Probabilities: Home Win: {probs[0][2]*100:.1f}%, Draw: {probs[0][1]*100:.1f}%, Away Win: {probs[0][0]*100:.1f}%")



# Use the updated function for prediction and confidence
y_pred, probs, confidence = predict_with_confidence(model, X_test)



# Evaluation
results_df = test_df[['Date', 'Home', 'Away', 'Winner', 'DivisionGap', 'AbsoluteDivisionGap']].copy()
results_df['Predicted'] = y_pred
results_df['Correct'] = results_df['Winner'] == results_df['Predicted']
results_df['Actual Outcome'] = results_df['Winner'].map({0: 'Away Win', 2: 'Home Win'})
results_df['Predicted Outcome'] = results_df['Predicted'].map({0: 'Away Win', 2: 'Home Win'})
# Add confidence to the results dataframe
results_df['Confidence'] = (confidence * 100).round(1)

# Add probabilities
results_df[['Away Win Prob', 'Draw Prob', 'Home Win Prob']] = (probs * 100).round(1)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracyFormatted = accuracy*100
confidence_df=results_df[['Date', 'Home', 'Away', 'Predicted Outcome', 'Confidence', 'Actual Outcome']].head()

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

if __name__ == '__main__':
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

    # confidence 5
    print(results_df[['Date', 'Home', 'Away', 'Predicted Outcome', 'Confidence']].head())


    predict_match_rf("2024-05-24", "Manchester City", "Manchester Utd")

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(len(features)), importances[indices], align='center')
    plt.xticks(range(len(features)), feature_names, rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()