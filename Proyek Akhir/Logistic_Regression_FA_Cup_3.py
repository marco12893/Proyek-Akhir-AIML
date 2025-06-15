import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Train logistic regression model
model = LogisticRegression(
    class_weight='balanced',

    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)


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
results_df = test_df[['Date', 'Home', 'Away', 'Winner']].copy()
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
