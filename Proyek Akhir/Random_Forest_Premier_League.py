import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("clean_data/English_Football_2018-2023_With_Form.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Encode teams
le_team = LabelEncoder()
le_team.fit(pd.concat([df['Home'], df['Away']]))
df['HomeTeam_enc'] = le_team.transform(df['Home'])
df['AwayTeam_enc'] = le_team.transform(df['Away'])

# --- Add form and division features ---
def add_form_and_division_features(df):
    df = df.copy()

    df['DivisionGap'] = df['AwayDivision'] - df['HomeDivision']
    df['AbsoluteDivisionGap'] = abs(df['DivisionGap'])
    df['HomeFormWeighted'] = df['HomeLast5_Wins'] / (df['AbsoluteDivisionGap'] + 1)
    df['AwayFormWeighted'] = df['AwayLast5_Wins'] * (df['AbsoluteDivisionGap'] + 1)

    return df

df = add_form_and_division_features(df)

# Filter data
train_df = df[
    (~((df['Type'] == 'League') & (df['Season'] == 2023) & (df['HomeDivision'] == 1) & (df['AwayDivision'] == 1))) &
    (df['Type'] == 'League')
]
test_df = df[(df['Type'] == 'League') & (df['Season'] == 2023) & (df['HomeDivision'] == 1) & (df['AwayDivision'] == 1)]

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

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
accuracyFormatted=accuracy*100

def predict_match_premier_league_rf(date_str, home_team, away_team, model=model, le=le_team, df_all=df):
    match_date = pd.to_datetime(date_str)
    df = df_all.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Cek apakah tim valid
    known_teams = set(df['Home']).union(set(df['Away']))
    if home_team not in known_teams or away_team not in known_teams:
        print(f"⚠️ Salah satu tim tidak ditemukan di dataset.")
        return

    # Fungsi bantu ambil divisi terakhir
    def get_division(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)]
        if matches.empty:
            print(f"⚠️ Tidak ada data historis divisi untuk {team}.")
            return np.nan
        last_match = matches.sort_values(by='Date', ascending=False).iloc[0]
        return last_match['HomeDivision'] if last_match['Home'] == team else last_match['AwayDivision']

    # Fungsi bantu form (5 laga terakhir menang)
    def get_last5_wins(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)].sort_values(by='Date', ascending=False).head(5)
        wins = 0
        for _, row in matches.iterrows():
            if row['Winner'] == 2 and row['Home'] == team:
                wins += 1
            elif row['Winner'] == 0 and row['Away'] == team:
                wins += 1
        return wins

    # Hitung fitur
    home_div = get_division(home_team)
    away_div = get_division(away_team)
    home_wins = get_last5_wins(home_team)
    away_wins = get_last5_wins(away_team)
    div_gap = away_div - home_div
    abs_gap = abs(div_gap)

    try:
        feature_row = pd.DataFrame([{
            'HomeTeam_enc': le.transform([home_team])[0],
            'AwayTeam_enc': le.transform([away_team])[0],
            'HomeDivision': home_div,
            'AwayDivision': away_div,
            'NeutralVenue': 0,
            'DivisionGap': div_gap,
            'AbsoluteDivisionGap': abs_gap,
            'HomeLast5_Wins': home_wins,
            'AwayLast5_Wins': away_wins,
            'HomeFormWeighted': home_wins / (abs_gap + 1),
            'AwayFormWeighted': away_wins * (abs_gap + 1)
        }])
    except:
        print("⚠️ Gagal encode tim. Cek nama tim atau label encoder.")
        return

    # Prediksi
    pred = model.predict(feature_row)[0]
    label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

    # Probabilitas (jika ingin ditampilkan)
    try:
        probs = model.predict_proba(feature_row)[0]
        confidence = probs[pred] * 100
        print(f"📅 Match: {date_str} — {home_team} vs {away_team}")
        print(f"🏆 Prediction: {label_map[pred]} ({confidence:.2f}% confidence)")
        print(f"📈 Home Form: {home_wins} wins | Division: {home_div}")
        print(f"📉 Away Form: {away_wins} wins | Division: {away_div}")
        print(f"📊 Probabilities — Home Win: {probs[2]*100:.1f}%, Draw: {probs[1]*100:.1f}%, Away Win: {probs[0]*100:.1f}%")
        return {'win': probs[2] * 100, 'draw': probs[1] * 100, 'lose': probs[0] * 100,
                'prediction': label_map[pred], 'home_team': home_team, 'away_team': away_team}
    except:
        print(f"🏆 Prediction: {label_map[pred]}")
        print("⚠️ Probabilities not available (model might not support them).")

# predict score
def predict_match_score_premier_league_rf(date_str, home_team, away_team, model=model, le=le_team, df_all=df):
    """
    Predict scores for a match using a trained Random Forest model.

    Args:
        date_str (str): Match date in string format.
        home_team (str): Home team name.
        away_team (str): Away team name.
        model: Trained Random Forest model.
        le: Label encoder for team names.
        df_all (pd.DataFrame): Dataset with historical data.

    Returns:
        dict: Predicted home and away scores.
    """
    match_date = pd.to_datetime(date_str)
    df = df_all.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Check if teams are valid
    known_teams = set(df['Home']).union(set(df['Away']))
    if home_team not in known_teams or away_team not in known_teams:
        print(f"⚠️ One or both teams not found in the dataset.")
        return None

    # Helper functions for division and form
    def get_division(team):
        matches = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < match_date)]
        if matches.empty:
            print(f"⚠️ No historical division data for {team}.")
            return np.nan
        last_match = matches.sort_values(by='Date', ascending=False).iloc[0]
        return last_match['HomeDivision'] if last_match['Home'] == team else last_match['AwayDivision']

    def get_last5_wins(team):
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

    # Compute features
    home_div = get_division(home_team)
    away_div = get_division(away_team)
    home_wins = get_last5_wins(home_team)
    away_wins = get_last5_wins(away_team)
    div_gap = away_div - home_div
    abs_gap = abs(div_gap)

    try:
        feature_row = pd.DataFrame([{
            'HomeTeam_enc': le.transform([home_team])[0],
            'AwayTeam_enc': le.transform([away_team])[0],
            'HomeDivision': home_div,
            'AwayDivision': away_div,
            'NeutralVenue': 0,
            'DivisionGap': div_gap,
            'AbsoluteDivisionGap': abs_gap,
            'HomeLast5_Wins': home_wins,
            'AwayLast5_Wins': away_wins,
            'HomeFormWeighted': home_wins / (abs_gap + 1),
            'AwayFormWeighted': away_wins * (abs_gap + 1)
        }])
    except Exception as e:
        print(f"⚠️ Error encoding teams or creating features: {e}")
        return None

    # Predict probabilities and classes
    probs = model.predict_proba(feature_row)[0]
    home_win_prob = probs[2]
    away_win_prob = probs[0]

    # Approximate scores based on probabilities
    home_score = round(home_win_prob * 3)  # Scale probabilities to goals
    away_score = round(away_win_prob * 3)

    print(f"\n📅 Match: {date_str} — {home_team} vs {away_team}")
    print(f"🔢 Predicted Score: {home_team} {home_score} - {away_score} {away_team}")
    return {'home_score': home_score, 'away_score': away_score}


if __name__ == '__main__':
    print(f"Random Forest Accuracy on Premier League 2023 test set: {accuracy * 100:.2f} %\n")

    classes = [0, 1, 2]
    target_names = ['Away Win', 'Draw', 'Home Win']

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=classes))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=classes, target_names=target_names, zero_division=0))

    # Display results
    results_df = test_df[['Date', 'Home', 'Away', 'Winner']].copy()
    results_df['Predicted'] = y_pred
    results_df['Correct'] = results_df['Winner'] == results_df['Predicted']

    outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    results_df['Actual Outcome'] = results_df['Winner'].map(outcome_map)
    results_df['Predicted Outcome'] = results_df['Predicted'].map(outcome_map)

    print("\nSample Predictions:")
    print(results_df[['Date', 'Home', 'Away', 'Actual Outcome', 'Predicted Outcome', 'Correct']].head(20))
    print("\n🔮 Prediksi Match Interaktif:")
    predict_match_premier_league_rf("2023-09-17", "Arsenal", "Manchester Utd")
    predict_match_score_premier_league_rf("2023-09-17", "Arsenal", "Manchester Utd")
