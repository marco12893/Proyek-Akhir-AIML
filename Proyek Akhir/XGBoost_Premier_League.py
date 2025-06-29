import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("clean_data/English_Football_2018-2023_With_Form.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Encode teams
le_team = LabelEncoder()
le_team.fit(pd.concat([df['Home'], df['Away']]))
df['HomeTeam_enc'] = le_team.transform(df['Home'])
df['AwayTeam_enc'] = le_team.transform(df['Away'])

# Add division gap and form weights
df['DivisionGap'] = df['AwayDivision'] - df['HomeDivision']
df['AbsoluteDivisionGap'] = df['DivisionGap'].abs()
df['HomeFormWeighted'] = df['HomeLast5_Wins'] / (df['AbsoluteDivisionGap'] + 1)
df['AwayFormWeighted'] = df['AwayLast5_Wins'] * (df['AbsoluteDivisionGap'] + 1)

# Train on all league matches except Premier League 2023
train_df = df[
    (~((df['Type'] == 'League') & (df['Season'] == 2023) & (df['HomeDivision'] == 1) & (df['AwayDivision'] == 1))) &
    (df['Type'] == 'League')
]

# Test on Premier League 2023 only
test_df = df[
    (df['Type'] == 'League') &
    (df['Season'] == 2023) &
    (df['HomeDivision'] == 1) &
    (df['AwayDivision'] == 1)
]

# Features
features = [
    'HomeTeam_enc', 'AwayTeam_enc',
    'HomeDivision', 'AwayDivision',
    'NeutralVenue', 'DivisionGap', 'AbsoluteDivisionGap',
    'HomeLast5_Wins', 'AwayLast5_Wins',
    'HomeFormWeighted', 'AwayFormWeighted'
]

X_train = train_df[features]
y_train = train_df['Winner']
X_test = test_df[features]
y_test = test_df['Winner']

# Train model
model = XGBClassifier(
    eval_metric='mlogloss',
    learning_rate=0.05,
    max_depth=5,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
accuracyFormatted=accuracy*100

def predict_match_premier_league(date_str, home_team, away_team, model=model, le=le_team, df_all=df):
    match_date = pd.to_datetime(date_str)
    df = df_all.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Cek keberadaan tim
    teams_set = set(df['Home']).union(set(df['Away']))
    if home_team not in teams_set or away_team not in teams_set:
        print(f"⚠️ Tim tidak ditemukan dalam data historis.")
        return

    # Fungsi bantu ambil divisi terakhir sebelum pertandingan
    def get_division(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)]
        if matches.empty:
            print(f"⚠️ Tidak ada data historis divisi untuk {team}.")
            return np.nan
        latest = matches.sort_values(by='Date', ascending=False).iloc[0]
        return latest['HomeDivision'] if latest['Home'] == team else latest['AwayDivision']

    # Fungsi bantu ambil jumlah menang dari 5 pertandingan terakhir
    def get_form_wins(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)].sort_values(by='Date', ascending=False).head(5)
        wins = 0
        for _, row in matches.iterrows():
            if row['Winner'] == 2 and row['Home'] == team:
                wins += 1
            elif row['Winner'] == 0 and row['Away'] == team:
                wins += 1
        return wins

    # Ambil fitur
    home_div = get_division(home_team)
    away_div = get_division(away_team)
    home_form = get_form_wins(home_team)
    away_form = get_form_wins(away_team)
    division_gap = away_div - home_div
    abs_gap = abs(division_gap)

    try:
        input_data = pd.DataFrame([{
            'HomeTeam_enc': le.transform([home_team])[0],
            'AwayTeam_enc': le.transform([away_team])[0],
            'HomeDivision': home_div,
            'AwayDivision': away_div,
            'NeutralVenue': 0,
            'DivisionGap': division_gap,
            'AbsoluteDivisionGap': abs_gap,
            'HomeLast5_Wins': home_form,
            'AwayLast5_Wins': away_form,
            'HomeFormWeighted': home_form / (abs_gap + 1),
            'AwayFormWeighted': away_form * (abs_gap + 1)
        }])
    except:
        print("⚠️ Gagal encode tim. Mungkin ada tim baru yang belum dikenal.")
        return

    # Prediksi
    probs = model.predict_proba(input_data)[0]
    predicted_class = np.argmax(probs)
    confidence = probs[predicted_class] * 100

    label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    print(f"📅 Match: {date_str} — {home_team} vs {away_team}")
    print(f"🏆 Prediction: {label_map[predicted_class]} ({confidence:.2f}% confidence)")
    print(f"📈 Home Form (last 5): {home_form} wins | 🧾 Division: {home_div}")
    print(f"📉 Away Form (last 5): {away_form} wins | 🧾 Division: {away_div}")
    print(f"📊 Probabilities — Home: {probs[2]*100:.1f}%, Draw: {probs[1]*100:.1f}%, Away: {probs[0]*100:.1f}%")
    return {'win': probs[2]*100, 'draw': probs[1]*100, 'lose': probs[0]*100, 'prediction': label_map[predicted_class], 'home_team': home_team, 'away_team': away_team}

# prediction score
def predict_match_score_premier_league(date_str, home_team, away_team, model=model, le=le_team, df_all=df):
    """
    Predict the score for a Premier League match using XGBoost.

    Args:
        date_str (str): Match date in string format.
        home_team (str): Home team name.
        away_team (str): Away team name.
        model (XGBClassifier): Trained XGBoost model.
        le (LabelEncoder): Encoder for team names.
        df_all (pd.DataFrame): Dataset with historical match data.

    Returns:
        dict: Predicted home and away scores.
    """
    match_date = pd.to_datetime(date_str)
    df = df_all.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Check if teams exist in historical data
    teams_set = set(df['Home']).union(set(df['Away']))
    if home_team not in teams_set or away_team not in teams_set:
        print("⚠️ Teams not found in the historical dataset.")
        return None

    # Helper functions to get division and form
    def get_division(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)]
        if matches.empty:
            print(f"⚠️ No historical division data for {team}.")
            return np.nan
        latest = matches.sort_values(by='Date', ascending=False).iloc[0]
        return latest['HomeDivision'] if latest['Home'] == team else latest['AwayDivision']

    def get_form_wins(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)].sort_values(by='Date',
                                                                                                            ascending=False).head(
            5)
        wins = 0
        for _, row in matches.iterrows():
            if row['Winner'] == 2 and row['Home'] == team:
                wins += 1
            elif row['Winner'] == 0 and row['Away'] == team:
                wins += 1
        return wins

    # Extract features
    home_div = get_division(home_team)
    away_div = get_division(away_team)
    home_form = get_form_wins(home_team)
    away_form = get_form_wins(away_team)
    division_gap = away_div - home_div
    abs_gap = abs(division_gap)

    try:
        input_data = pd.DataFrame([{
            'HomeTeam_enc': le.transform([home_team])[0],
            'AwayTeam_enc': le.transform([away_team])[0],
            'HomeDivision': home_div,
            'AwayDivision': away_div,
            'NeutralVenue': 0,
            'DivisionGap': division_gap,
            'AbsoluteDivisionGap': abs_gap,
            'HomeLast5_Wins': home_form,
            'AwayLast5_Wins': away_form,
            'HomeFormWeighted': home_form / (abs_gap + 1),
            'AwayFormWeighted': away_form * (abs_gap + 1)
        }])
    except ValueError as e:
        print(f"⚠️ Error encoding teams: {e}")
        return None

    # Predict probabilities
    probs = model.predict_proba(input_data)[0]

    # Approximate scores based on probabilities
    home_score = round(probs[2] * 3)
    away_score = round(probs[0] * 3)

    print(f"\n📅 Match: {date_str} — {home_team} vs {away_team}")
    print(f"🔢 Predicted Score: {home_team} {home_score} - {away_score} {away_team}")
    return {'home_score': home_score, 'away_score': away_score}


if __name__ == '__main__':
    print(f"XGBoost Accuracy on Premier League 2023 test set: {accuracy * 100:.2f}%\n")

    classes = [0, 1, 2]
    target_names = ['Away Win', 'Draw', 'Home Win']

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=classes))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=classes, target_names=target_names, zero_division=0))

    # Show predictions
    results_df = test_df[['Date', 'Home', 'Away', 'Winner']].copy()
    results_df['Predicted'] = y_pred
    results_df['Correct'] = results_df['Winner'] == results_df['Predicted']

    outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    results_df['Actual Outcome'] = results_df['Winner'].map(outcome_map)
    results_df['Predicted Outcome'] = results_df['Predicted'].map(outcome_map)

    print("\nSample Predictions:")
    print(results_df[['Date', 'Home', 'Away', 'Actual Outcome', 'Predicted Outcome', 'Correct']].head(20))

    print("\n🔮 Prediksi Pertandingan Baru:")
    predict_match_premier_league("2023-08-27", "Fulham", "Liverpool")
    predict_match_score_premier_league("2023-08-27", "Fulham", "Liverpool")

