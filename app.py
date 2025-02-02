from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)

# Load dataset
df = pd.read_csv("covid_toy.csv")
df = df.dropna()

# Label Encoding for Categorical Features
label_gender = LabelEncoder()
label_city = LabelEncoder()
label_yes_no = LabelEncoder()  # For "Yes/No" values

df['gender'] = label_gender.fit_transform(df['gender'])  # Male/Female → 0/1
df['city'] = label_city.fit_transform(df['city'])  # Encode city names
df['fever'] = label_yes_no.fit_transform(df['fever'])  # Yes/No → 0/1
df['cough'] = label_yes_no.fit_transform(df['cough'])  # Yes/No → 0/1
df['has_covid'] = label_yes_no.fit_transform(df['has_covid'])  # Encode target variable

# Features & Target
X = df[['age', 'gender', 'fever', 'cough', 'city']]
y = df['has_covid']

# Check class balance
print("COVID Positive:", sum(y == 1))
print("COVID Negative:", sum(y == 0))

# Ensure training data is balanced
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize Only Numerical Data (Age)
scaler = StandardScaler()
X_train[['age']] = scaler.fit_transform(X_train[['age']])
X_test[['age']] = scaler.transform(X_test[['age']])

# Train Logistic Regression Model with Class Balancing
# model = LogisticRegression(class_weight='balanced', random_state=42, solver='liblinear')
# model.fit(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Save Model & Encoders
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump({'gender': label_gender, 'city': label_city, 'yes_no': label_yes_no, 'scaler': scaler}, f)

# Function to Predict COVID Status
def predict_covid(data):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    data_df = pd.DataFrame([data])
    data_df['gender'] = encoders['gender'].transform(data_df['gender'])
    data_df['city'] = encoders['city'].transform(data_df['city'])
    data_df['fever'] = encoders['yes_no'].transform(data_df['fever'])
    data_df['cough'] = encoders['yes_no'].transform(data_df['cough'])
    data_df[['age']] = encoders['scaler'].transform(data_df[['age']])

    probability = model.predict_proba(data_df)[0][1]  # Probability of COVID Positive

    THRESHOLD = 0.70  # Adjusted threshold (Increase for more negatives)
    prediction = 1 if probability > THRESHOLD else 0

    print(f"Prediction: {prediction}, Probability: {probability:.2f}")

    return "COVID Positive" if prediction == 1 else "COVID Negative"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = int(request.form["age"])
        gender = request.form["gender"]
        fever = request.form["fever"]
        cough = request.form["cough"]
        city = request.form["city"]

        user_data = {"age": age, "gender": gender, "fever": fever, "cough": cough, "city": city}
        prediction = predict_covid(user_data)

        return render_template("result.html", prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=8000)
