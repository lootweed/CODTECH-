from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# -------------------------------
# Step 1: Load and Preprocess Data
# -------------------------------
data_path = "C:/Users/viswa/OneDrive/Downloads/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(data_path)

# Drop customerID and rows with missing TotalCharges
df.drop("customerID", axis=1, inplace=True)
df = df[df["TotalCharges"] != " "]
df["TotalCharges"] = df["TotalCharges"].astype(float)

# Encode target label manually to ensure consistency
df["Churn"] = df["Churn"].replace({'Yes': 1, 'No': 0})

# Encode other categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and preprocessors
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
joblib.dump(le_dict, "label_encoders.pkl")

# -------------------------------
# Step 2: Create Flask Web App
# -------------------------------
app = Flask(__name__)
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
le_dict = joblib.load("label_encoders.pkl")

# HTML template
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Churn Prediction</title>
</head>
<body style="font-family:sans-serif;">
    <h2>Telco Customer Churn Predictor</h2>
    <form action="/predict" method="post">
        {% for f in features %}
            <label>{{ f }}</label><br>
            <input name="{{ f }}" type="text" required><br><br>
        {% endfor %}
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
        <h3>Prediction: <span style="color:blue;">{{ prediction }}</span></h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML, features=features, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = []
        for f in features:
            val = request.form[f].strip()
            if f in le_dict:
                le = le_dict[f]
                # Ensure the input exactly matches one of the classes
                if val not in le.classes_:
                    return f"<p style='color:red;'>Invalid input for {f}: {val}<br>Allowed: {list(le.classes_)}</p>"
                val = le.transform([val])[0]
            else:
                val = float(val)
            input_data.append(val)

        input_scaled = scaler.transform([input_data])
        pred = model.predict(input_scaled)[0]
        label = "Churn" if pred == 1 else "Not Churn"
        return render_template_string(HTML, features=features, prediction=label)
    except Exception as e:
        return f"<p style='color:red;'>Error: {str(e)}</p>"

if __name__ == "__main__":
    app.run(debug=True)
