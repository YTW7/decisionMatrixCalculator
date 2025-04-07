from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import numpy as np
import pandas as pd
import os
import time
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

df, feature_columns, target_column, X_train, X_test, Y_train, Y_test, scaler = None, None, None, None, None, None, None, None

classifiers = {
    "Logistic Regression": LogisticRegression(),
    "SVM": svm.SVC(kernel='linear', random_state=2),
    "Decision Tree": DecisionTreeClassifier(random_state=2),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=2),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=900, random_state=2),
    "Naïve Bayes": GaussianNB(),
}

@app.route('/')
def home():
    return render_template('index.html', classifiers=classifiers.keys(), results=[])

@app.route('/upload', methods=['POST'])
def upload_file():
    global df, feature_columns, target_column, X_train, X_test, Y_train, Y_test, scaler
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        df = pd.read_csv(filepath)
        feature_columns = df.columns[:-1].tolist()
        target_column = df.columns[-1]
        
        return jsonify({"message": "Dataset uploaded successfully", "features": feature_columns, "target": target_column})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    global X_train, X_test, Y_train, Y_test, scaler
    if df is None:
        return jsonify({"error": "No dataset uploaded"}), 400
    
    try:
        training_percentage = float(request.form['training_percentage'])
        test_size = 1 - (training_percentage / 100)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_columns])
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled, df[target_column], test_size=test_size, stratify=df[target_column], random_state=2
        )

        results = []
        for name, classifier in classifiers.items():
            start_time = time.time()
            classifier.fit(X_train, Y_train)
            training_time = (time.time() - start_time) * 1000
            test_accuracy = accuracy_score(Y_test, classifier.predict(X_test))
            
            results.append({
                "name": name,
                "test_accuracy": round(test_accuracy, 2),
                "training_time": round(training_time, 4)
            })

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dmatrix/input', methods=['GET', 'POST'])
def dmatrix_input():
    if request.method == 'POST':
        session['attributes'] = request.form.getlist("attributes")
        session['alternatives'] = request.form.getlist("alternatives")
        session['attribute_types'] = [request.form.get(f"attribute_type_{i}") for i in range(len(session['attributes']))]
        return redirect(url_for('dmatrix_normalize'))
    return render_template('dmatrix_input.html')

@app.route('/dmatrix/normalize', methods=['GET', 'POST'])
def dmatrix_normalize():
    if request.method == 'POST':
        matrix_values = list(map(float, request.form.getlist("matrix[]")))
        num_alternatives = len(session['alternatives'])
        num_attributes = len(session['attributes'])
        matrix = np.array(matrix_values).reshape(num_alternatives, num_attributes)
        session['matrix'] = matrix.tolist()

        column_sums = np.sqrt(np.sum(matrix**2, axis=0))
        normalized_matrix = matrix / column_sums
        session['normalized_matrix'] = normalized_matrix.tolist()

        return redirect(url_for('dmatrix_weight'))

    return render_template(
        'dmatrix_normalize.html',
        attributes=session['attributes'],
        alternatives=session['alternatives']
    )

@app.route('/dmatrix/weight', methods=['GET', 'POST'])
def dmatrix_weight():
    if request.method == 'POST':
        weights = list(map(float, request.form.getlist("weights[]")))
        session['weights'] = weights

        normalized_matrix = np.array(session['normalized_matrix'])
        weighted_matrix = normalized_matrix * weights
        session['weighted_matrix'] = weighted_matrix.tolist()

        return redirect(url_for('dmatrix_final'))

    return render_template(
        'dmatrix_weight.html',
        attributes=session.get('attributes', []),
        alternatives=session.get('alternatives', []),
        attribute_types=session.get('attribute_types', []),
        normalized_matrix=session['normalized_matrix']
    )

@app.route('/dmatrix/final', methods=['GET'])
def dmatrix_final():
    attributes = session['attributes']
    alternatives = session['alternatives']
    attribute_types = session['attribute_types']
    weighted_matrix = np.array(session['weighted_matrix'])

    PIS = []
    NIS = []

    for j in range(len(attribute_types)):
        col = weighted_matrix[:, j]
        if attribute_types[j] == "positive":
            PIS.append(np.max(col))
            NIS.append(np.min(col))
        else:
            PIS.append(np.min(col))
            NIS.append(np.max(col))

    s_plus = np.sqrt(np.sum((weighted_matrix - PIS) ** 2, axis=1))
    s_minus = np.sqrt(np.sum((weighted_matrix - NIS) ** 2, axis=1))
    relative_closeness = s_minus / (s_plus + s_minus)
    ranking = np.argsort(-relative_closeness).tolist()

    return render_template(
        'dmatrix_final.html',
        attributes=attributes,
        alternatives=alternatives,
        weighted_matrix=weighted_matrix.tolist(),
        PIS=PIS,
        NIS=NIS,
        s_plus=s_plus.tolist(),
        s_minus=s_minus.tolist(),
        relative_closeness=relative_closeness.tolist(),
        ranking=ranking
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)
