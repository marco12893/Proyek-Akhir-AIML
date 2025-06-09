import pandas as pd
import numpy as np
from tqdm import tqdm

# ======================
# DATA CLEANING PHASE
# ======================
print("Starting data cleaning pipeline...")

# Load datasets
df = pd.read_csv('data/English_Football_2018-2025_With_Form.csv')
team_divisions = pd.read_csv('data/League Division 2.csv')

# 1. Remove rows containing 'Attendance' (duplicate headers)
df = df[~df.apply(lambda row: row.astype(str).str.contains('Attendance', case=False, na=False)).any(axis=1)]

# 2. Remove remaining header rows
df_cleaned = df[df['Wk'] != 'Wk']

# 3. Drop sparse rows (<3 valid values)
df_cleaned = df_cleaned.dropna(thresh=3)


# 5. Convert xG to numeric
df_cleaned['xG'] = pd.to_numeric(df_cleaned['xG'], errors='coerce')
df_cleaned['xG.1'] = pd.to_numeric(df_cleaned['xG.1'], errors='coerce')

# 6. Extract goals from Score column
df_cleaned[['HomeGoals', 'AwayGoals']] = df_cleaned['Score'].str.extract(r'(\d+)–(\d+)').astype(float)

# 7. Convert dates and sort
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
df_cleaned = df_cleaned.sort_values('Date').reset_index(drop=True)

# 8. Update divisions for FA Cup matches
print("Updating team divisions...")
home_div_lookup = team_divisions.set_index(['Squad', 'Season'])['Division'].to_dict()
df_cleaned['HomeDivision'] = df_cleaned.apply(
    lambda row: home_div_lookup.get((row['Home'], row['Season']), 6), axis=1
)
df_cleaned['AwayDivision'] = df_cleaned.apply(
    lambda row: home_div_lookup.get((row['Away'], row['Season']), 6), axis=1
)

# ======================
# FEATURE ENGINEERING PHASE
# ======================
print("Calculating team form features...")


def calculate_team_form(df, team_name, current_date, current_index):
    """Calculate form metrics using only prior matches"""
    team_matches = df[((df['Home'] == team_name) | (df['Away'] == team_name)) &
                      (df['Date'] < current_date)]

    last_5 = team_matches.tail(5)
    if len(last_5) < 5:
        return None

    form = {
        'GoalsFor': 0, 'GoalsAgainst': 0,
        'Wins': 0, 'Draws': 0, 'Losses': 0,
        'CleanSheets': 0
    }

    for _, match in last_5.iterrows():
        is_home = match['Home'] == team_name
        gf = match['HomeGoals'] if is_home else match['AwayGoals']
        ga = match['AwayGoals'] if is_home else match['HomeGoals']

        form['GoalsFor'] += gf
        form['GoalsAgainst'] += ga
        form['CleanSheets'] += 1 if ga == 0 else 0

        if match['Winner'] == (0 if is_home else 2):
            form['Wins'] += 1
        elif match['Winner'] == 1:
            form['Draws'] += 1
        else:
            form['Losses'] += 1

    return form


# Add form features with progress bar
form_features = [
    'HomeLast5_GoalsFor', 'AwayLast5_GoalsFor',
    'HomeLast5_GoalsAgainst', 'AwayLast5_GoalsAgainst',
    'HomeLast5_Wins', 'AwayLast5_Wins',
    'HomeLast5_Draws', 'AwayLast5_Draws',
    'HomeLast5_Losses', 'AwayLast5_Losses',
    'HomeLast5_CleanSheets', 'AwayLast5_CleanSheets'
]

for feat in form_features:
    df_cleaned[feat] = np.nan

for i, row in tqdm(df_cleaned.iterrows(), total=len(df_cleaned)):
    # Home team form
    if home_form := calculate_team_form(df_cleaned.iloc[:i], row['Home'], row['Date'], i):
        df_cleaned.loc[i, 'HomeLast5_GoalsFor'] = home_form['GoalsFor']
        df_cleaned.loc[i, 'HomeLast5_GoalsAgainst'] = home_form['GoalsAgainst']
        df_cleaned.loc[i, 'HomeLast5_Wins'] = home_form['Wins']
        df_cleaned.loc[i, 'HomeLast5_Draws'] = home_form['Draws']
        df_cleaned.loc[i, 'HomeLast5_Losses'] = home_form['Losses']
        df_cleaned.loc[i, 'HomeLast5_CleanSheets'] = home_form['CleanSheets']

    # Away team form
    if away_form := calculate_team_form(df_cleaned.iloc[:i], row['Away'], row['Date'], i):
        df_cleaned.loc[i, 'AwayLast5_GoalsFor'] = away_form['GoalsFor']
        df_cleaned.loc[i, 'AwayLast5_GoalsAgainst'] = away_form['GoalsAgainst']
        df_cleaned.loc[i, 'AwayLast5_Wins'] = away_form['Wins']
        df_cleaned.loc[i, 'AwayLast5_Draws'] = away_form['Draws']
        df_cleaned.loc[i, 'AwayLast5_Losses'] = away_form['Losses']
        df_cleaned.loc[i, 'AwayLast5_CleanSheets'] = away_form['CleanSheets']

# Fill NA values for teams with insufficient history
df_cleaned[form_features] = df_cleaned[form_features].fillna(0)

# ======================
# FINAL OUTPUT
# ======================
print("Saving cleaned dataset with form features...")
df_cleaned.to_csv('clean_data/English_Football_2018-2023_With_Form 2.csv', index=False)
print("Pipeline completed successfully!")