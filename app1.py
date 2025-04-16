import eventlet
eventlet.monkey_patch()
import numpy as np

from flask import Flask, request, jsonify, send_file,render_template
from flask_socketio import SocketIO
import joblib
import pandas as pd
import io
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


# ——— Load model, preprocessor, and validation data ———
model        = joblib.load('model3.pkl')
preprocessor = joblib.load('ct.pkl')

X_val = pd.read_csv('X_val.csv')
y_val = pd.read_csv('y_val.csv')['Machine failure']

# Global storage for latest validation results
latest_results = {
    'y_true': None,
    'y_pred': None,
    'y_prob': None,
    'metrics': None
}

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/')
def index():
    return "Predictive Maintenance Server Running."


@app.route('/validate', methods=['GET'])
def validate():
    # 1. Transform & predict
    
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # 2. Compute metrics
    roc_auc = roc_auc_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred).tolist()
    report = classification_report(y_val, y_pred, output_dict=True)

    # 3. Store for plotting endpoints
    latest_results.update({
        'y_true':  y_val,
        'y_pred':  y_pred,
        'y_prob':  y_prob,
        'metrics': {
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report
        }
    })

    # 4. Emit for any dashboard
    socketio.emit('validation_results', latest_results['metrics'])

    return jsonify(latest_results['metrics'])


@app.route('/plot_confusion_matrix', methods=['GET'])
def plot_confusion_matrix():
    if latest_results['y_true'] is None:
        return "Run /validate first", 400

    cm = confusion_matrix(latest_results['y_true'], latest_results['y_pred'])

    # Plot with Matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)

    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=['Neg', 'Pos'],
        yticklabels=['Neg', 'Pos'],
        ylabel='Actual',
        xlabel='Predicted',
        title='Confusion Matrix'
    )
    # annotate counts
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
     for j in range(cm.shape[1]):
        ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)',
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > 0.5 else "black")


    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/plot_roc_curve', methods=['GET'])
def plot_roc_curve():
    if latest_results['y_true'] is None:
        return "Run /validate first", 400

    fpr, tpr, _ = roc_curve(latest_results['y_true'], latest_results['y_prob'])
    auc_score = roc_auc_score(latest_results['y_true'], latest_results['y_prob'])

    # Plot ROC
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set(
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.05],
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        title='Receiver Operating Characteristic'
    )
    ax.legend(loc="lower right")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/classification_report', methods=['GET'])
def classification_report_view():
    if latest_results['metrics'] is None:
        return "Run /validate first", 400

    report = latest_results['metrics']['classification_report']
    df = pd.DataFrame(report).transpose()

    # Format as an HTML table
    return df.to_html(classes='table table-bordered', border=0)



@app.route('/sensor-data', methods=['POST'])
def sensor_data():
    try:
        data = request.json
        df = pd.DataFrame([data])
        X_t = preprocessor.transform(df)
        prob = model.predict_proba(X_t)[0, 1]
        pred = model.predict(X_t)[0]

        result = {'prediction': int(pred), 'probability': float(prob)}
        socketio.emit('realtime_prediction', result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

