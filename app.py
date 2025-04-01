import eventlet
eventlet.monkey_patch()




from flask import Flask, render_template, jsonify,send_file
from flask_socketio import SocketIO
import io
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report,roc_curve,auc

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load your pre-trained model (ensure model1.pkl is in your project directory)
pdm_model = joblib.load('model1.pkl')
preprocessor = joblib.load('ct.pkl')

# Global storage for simulation results
latest_results = {
    'true_labels': None,
    'predictions': None,
    'probabilities': None
}

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/simulate', methods=['GET'])
def simulate():
    global latest_results
    n_samples = 50  # Number of simulated samples

    # Generate random sensor data (adjust ranges as needed)
    air_temp = np.random.uniform(297, 299, n_samples)            # Air Temperature (°C)
    process_temp = np.random.uniform(307, 313, n_samples)        # Process Temperature (°C)
    rpm = np.random.uniform(1200, 1800, n_samples)                # RPM
    torque = np.random.uniform(10, 80, n_samples)               # Torque (arbitrary units)
    toolwear = np.random.uniform(0, 500, n_samples)                # Tool Wear (normalized 0-1)
    model_type = np.random.choice(['M','L','H'],n_samples)
    # Create a DataFrame with the sensor data
    data = pd.DataFrame({
        'Type': model_type,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rpm,
        'Torque [Nm]': torque,
        'Tool wear [min]': toolwear,
        # Adding dummy columns for those expected by the preprocessor
        'Machine failure': np.zeros(n_samples), # This is the target column
        'TWF': np.zeros(n_samples),
        'HDF': np.zeros(n_samples),
        'PWF': np.zeros(n_samples),
        'OSF': np.zeros(n_samples),
        'RNF': np.zeros(n_samples)
        
    })
    transformed_data = preprocessor.transform(data) 
    ohe_type = preprocessor.named_transformers_['model_type'].get_feature_names_out(['Type'])
    new_dataset_columns = list(ohe_type) + ['Air_temp'] + ['Process_temp'] + ['Rotational_speed'] + ['Torque'] + ['Tool_wear'] + ['Machine_failure'] + ['TWF'] + ['HDF'] + ['PWF'] + ['OSF'] +['RNF']
    pdm_dataset_transformed = pd.DataFrame(transformed_data, columns = new_dataset_columns)
    pdm_dataset_transformed.drop(['Machine_failure','TWF','HDF','PWF','OSF','RNF'],axis = 1,inplace = True)
    # Run model predictions
    probabilities = pdm_model.predict_proba(pdm_dataset_transformed)[:, 1]  # Probability for class 1
    predictions = pdm_model.predict(pdm_dataset_transformed)

    # For demonstration, simulate true labels (in a real scenario, use your true labels)
    true_labels = np.random.randint(0, 2, n_samples)

    # Store the latest simulation results globally.
    latest_results['true_labels'] = true_labels
    latest_results['predictions'] = predictions
    latest_results['probabilities'] = probabilities


    # Calculate performance metrics
    roc_auc = roc_auc_score(true_labels, probabilities) if len(set(true_labels)) > 1 else None

    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions)

    # Prepare the results dictionary
    results = {
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),  # Convert numpy array to list for JSON serialization
        'classification_report': report,
        'data': data.to_dict(orient='records'),
        'predictions': predictions.tolist()
    }

    # Emit results via Socket.IO for real-time dashboard updates
    socketio.emit('simulation_update', results)

    return jsonify(results)  # Return results as JSON response

@app.route('/plot_confusion_matrix')
def plot_confusion_matrix():
    if latest_results['true_labels'] is None:
        return "No simulation data available", 400

    cm = confusion_matrix(latest_results['true_labels'], latest_results['predictions'])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype='image/png')

@app.route('/classification_report')
def get_classification_report():
    if latest_results['true_labels'] is None:
        return "No simulation data available", 400

    report = classification_report(latest_results['true_labels'], latest_results['predictions'])
    return "<pre>" + report + "</pre>"

@app.route('/plot_roc')
def plot_roc():
    if latest_results['true_labels'] is None:
        return "No simulation data available", 400

    fpr, tpr, _ = roc_curve(latest_results['true_labels'], latest_results['probabilities'])
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.2f)" % roc_auc_val, color="blue")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype='image/png')


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')
