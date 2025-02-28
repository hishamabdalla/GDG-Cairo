import numpy as np
from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize label encoders
label_encoder_category = LabelEncoder()
label_encoder_street = LabelEncoder()
label_encoder_city = LabelEncoder()
label_encoder_state = LabelEncoder()
label_encoder_job = LabelEncoder()
label_encoder_gender = LabelEncoder()

# Define a threshold for unusually high transactions
HIGH_AMOUNT_THRESHOLD = 1000  # Adjust this as needed

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Fraud Detection API! Use the /predict endpoint to send data and get predictions."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON input
        data = request.json

        # Extract values from JSON
        category = data['category']
        street = data['street']
        city = data['city']
        state = data['state']
        job = data['job']
        gender = data['gender']
        hour = int(data['hour'])
        amt = float(data['amt'])

        # Encode categorical data
        category_encoded = label_encoder_category.fit_transform([category])[0]
        street_encoded = label_encoder_street.fit_transform([street])[0]
        city_encoded = label_encoder_city.fit_transform([city])[0]
        state_encoded = label_encoder_state.fit_transform([state])[0]
        job_encoded = label_encoder_job.fit_transform([job])[0]
        gender_encoded = label_encoder_gender.fit_transform([gender])[0]

        # Prepare data as a NumPy array
        final_features = np.array([category_encoded, street_encoded, city_encoded, 
                                   state_encoded, job_encoded, gender_encoded, 
                                   hour, amt]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)[0]

        # Determine the result message
        fraud_message = "⚠️ Potential Fraudulent Transaction!" if prediction == 1 else "✅ Transaction is Safe."
        
        # Check for high transaction amount
        high_amount_warning = ""
        if amt > HIGH_AMOUNT_THRESHOLD:
            high_amount_warning = "⚠️ WARNING: This transaction amount is unusually high compared to normal spending behavior."

        return jsonify({
            "prediction": int(prediction),
            "message": fraud_message,
            "high_amount_warning": high_amount_warning if high_amount_warning else None,
            "statusCode": 200
        }), 200

    except KeyError as e:
        return jsonify({"message": f"⚠️ Missing or invalid data: {str(e)}", "statusCode": 400}), 400

    except Exception as e:
        return jsonify({"message": str(e), "statusCode": 500}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

