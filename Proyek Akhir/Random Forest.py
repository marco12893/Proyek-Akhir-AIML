import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/Premier League 2018-2023 Update.csv")

# Drop rows with missing values
df = df.dropna(subset=['Home', 'Away', 'xG', 'xG.1', 'Winner'])

# Encode teams
le_team = LabelEncoder()
df['HomeTeam_enc'] = le_team.fit_transform(df['Home'])
df['AwayTeam_enc'] = le_team.transform(df['Away'])

# Feature columns
features = ['HomeTeam_enc', 'AwayTeam_enc', 'xG', 'xG.1']
target = 'Winner'

# Split by season
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
train_df = df[df['Date'].dt.year < 2023]
test_df = df[df['Date'].dt.year == 2023]

# Ensure test_df copy (to avoid SettingWithCopyWarning)
test_df = test_df.copy()
test_df['HomeTeam_enc'] = le_team.transform(test_df['Home'])
test_df['AwayTeam_enc'] = le_team.transform(test_df['Away'])

# Prepare training and testing data
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f} %\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
