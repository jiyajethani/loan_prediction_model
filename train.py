import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("../data/dataset.csv")


# ----------------------------
# Preprocessing
# ----------------------------
df.fillna({
    'Gender': df['Gender'].mode()[0],
    'Married': df['Married'].mode()[0],
    'Dependents': df['Dependents'].mode()[0],
    'Self_Employed': df['Self_Employed'].mode()[0],
    'LoanAmount': df['LoanAmount'].median(),
    'Credit_History': df['Credit_History'].mode()[0]
}, inplace=True)


# ----------------------------
# Features & Target
# ----------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"].map({'Approved': 1, 'Rejected': 0})


# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)


# ----------------------------
# Pipeline
# ----------------------------
categorical_cols = [
    'Gender', 'Married', 'Dependents',
    'Education', 'Self_Employed', 'Property_Area'
]

numerical_cols = ['Income', 'LoanAmount', 'Credit_History']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight='balanced',
    random_state=42
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', model)
])


# ----------------------------
# Train Model
# ----------------------------
pipeline.fit(X_train, y_train)


# ----------------------------
# Evaluate Model
# ----------------------------
y_pred = pipeline.predict(X_test)

print("\n📊 Model Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ----------------------------
# Save Model
# ----------------------------
joblib.dump(pipeline, "../model.pkl")
print("\n✅ Model saved as model.pkl")
