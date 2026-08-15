from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("voting_model.pkl", "rb"))

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get form values
    flow_duration = float(request.form["flow_duration"])
    fwd_packets = float(request.form["fwd_packets"])
    bwd_packets = float(request.form["bwd_packets"])
    fwd_length = float(request.form["fwd_length"])
    fwd_max = float(request.form["fwd_max"])
    idle_mean = float(request.form["idle_mean"])

    # Base features
    base_features = [
        flow_duration,
        fwd_packets,
        bwd_packets,
        fwd_length,
        fwd_max,
        idle_mean
    ]

    # Additional engineered features
    mean_feature = np.mean(base_features)

    std_feature = np.std(base_features)

    max_feature = np.max(base_features)

    min_feature = np.min(base_features)

    entropy_feature = -np.sum(
        np.array(base_features) *
        np.log1p(np.abs(base_features) + 1)
    )

    traffic_complexity = (
        std_feature * mean_feature
    )

    # Threat score
    threat_score = 0

    if traffic_complexity > 1000:
        threat_score += 40

    if entropy_feature < -5000:
        threat_score += 25

    if std_feature > 100:
        threat_score += 15

    if max_feature > 5000:
        threat_score += 20

    # Final 13 features
    final_features = [
        flow_duration,
        fwd_packets,
        bwd_packets,
        fwd_length,
        fwd_max,
        idle_mean,
        mean_feature,
        std_feature,
        max_feature,
        min_feature,
        entropy_feature,
        traffic_complexity,
        threat_score
    ]

    # Convert to numpy array
    features_array = np.array([final_features])

    # Scale features
    scaled_features = scaler.transform(features_array)

    # Prediction
    prediction = model.predict(scaled_features)[0]

    # Probability
    probability = model.predict_proba(scaled_features)[0]

    # Result
    if prediction == 0:
        result = "BENIGN NETWORK TRAFFIC"
        confidence = probability[0]
    else:
        result = "MALICIOUS CYBER ATTACK DETECTED"
        confidence = probability[1]

    return render_template(
        "result.html",
        prediction=result,
        confidence=round(confidence * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
    