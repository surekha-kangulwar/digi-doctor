from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load models
with open("savemodels/Diabetes.sav", 'rb') as file:
    dia_model = pickle.load(file)
with open("savemodels/Heart.sav", 'rb') as file:
    heart_model = pickle.load(file)
with open("savemodels/BreastCancer.sav", 'rb') as file:
    breast_cancer_model = pickle.load(file)

# @app.route('/surekhaadidu/hello', methods=['GET'])
# def hello():
#     return "hi didiiii"
    
    
@app.route('/api/predict-diabetes', methods=['POST'])
def predict_diabetes():
    data = request.json
    prediction = dia_model.predict([[data['preg'], data['glucose'], data['bp'], data['skin'], data['insulin'], data['bmi'], data['dpf'], data['age']]])
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return jsonify({'result': result})

@app.route('/api/predict-heart-disease', methods=['POST'])
def predict_heart_disease():
    data = request.json
    sex = 0 if data['sex'] == 'Male' else 1
    cp = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(data['cp'])
    fbs = 0 if data['fbs'] == '>120 mg/dl' else 1
    restecg = ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'].index(data['restecg'])
    exang = 1 if data['exang'] == 'Yes' else 0
    slope = ['Upsloping', 'Flat', 'Downsloping'].index(data['slope'])
    thal = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(data['thal'])

    prediction = heart_model.predict([[data['age'], sex, cp, data['trestbps'], data['chol'], fbs, restecg, data['thalach'], exang, data['oldpeak'], slope, data['ca'], thal]])
    result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
    return jsonify({'result': result})

@app.route('/api/predict-breast-cancer', methods=['POST'])
def predict_breast_cancer():
    data = request.json
    prediction = breast_cancer_model.predict([[data['radius_mean'], data['texture_mean'], data['perimeter_mean'], data['area_mean'], data['smoothness_mean'], data['compactness_mean'], data['concavity_mean'], data['concave_points_mean'], data['symmetry_mean'], data['fractal_dimension_mean']]])
    result = "Malignant" if prediction[0] == 1 else "Benign"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
