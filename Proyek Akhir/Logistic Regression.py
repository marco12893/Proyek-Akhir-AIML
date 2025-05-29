import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === Load Data ===
df = pd.read_csv("data/Premier League 2018-2023 Update.csv")

# === Make Sure All Required Columns Exist ===
required_cols = ['Date', 'Home', 'Away', 'xG', 'xG.1', 'Winner']  # Winner is your label column
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# === Extract Year from Date Column ===
df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
df = df.dropna(subset=['Year'])  # Drop rows where date conversion failed

# === Split into Train and Test Sets ===
train_df = df[df['Year'].between(2018, 2023)]
test_df = df[df['Year'] == 2023]  # Use 2023–2024 season for testing

# === Label Encode Teams ===
le_team = LabelEncoder()
all_teams = pd.concat([train_df['Home'], train_df['Away'], test_df['Home'], test_df['Away']])
le_team.fit(all_teams)

train_df['HomeTeam_enc'] = le_team.transform(train_df['Home'])
train_df['AwayTeam_enc'] = le_team.transform(train_df['Away'])

test_df = test_df.copy()  # add this before encoding
test_df['HomeTeam_enc'] = le_team.transform(test_df['Home'])
test_df['AwayTeam_enc'] = le_team.transform(test_df['Away'])

# === Feature Selection ===
X_train = train_df[['xG', 'xG.1', 'HomeTeam_enc', 'AwayTeam_enc']]
y_train = train_df['Winner']

X_test = test_df[['xG', 'xG.1', 'HomeTeam_enc', 'AwayTeam_enc']]
y_test = test_df['Winner']

# === Train Logistic Regression ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === Predict ===
preds = model.predict(X_test)

# === Evaluate ===
acc = accuracy_score(y_test, preds)
print("Logistic Regression Accuracy:", round(acc * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))
