from flask import Flask, request, jsonify, render_template, send_file
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Load training and testing data from CSVs
train_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('testing_data.csv')

# Clean column names (remove extra whitespace)
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

@app.route('/')
def index():
    # Pass the columns to the HTML template for populating the dropdowns
    columns = train_df.columns.tolist()
    return render_template('index.html', columns=columns)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get query parameters
        feature_col = request.args.get('feature', 'YEAR')
        target_col = request.args.get('target', 'QLD_total')
        degree = int(request.args.get('degree', 2))
        value = float(request.args.get('value'))
        
        if feature_col not in train_df.columns or target_col not in train_df.columns:
            return jsonify({'error': 'Invalid column name'}), 400

        # Prepare training and testing data
        X_train = train_df[[feature_col]].values
        y_train = train_df[target_col].values
        X_test = test_df[[feature_col]].values
        y_test = test_df[target_col].values

        # Create polynomial features and train model
        poly = PolynomialFeatures(degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Make prediction for the given input value
        input_poly = poly.transform([[value]])
        prediction = model.predict(input_poly)[0]
        test_score = model.score(X_test_poly, y_test)

        return jsonify({
            'feature_column': feature_col,
            'target_column': target_col,
            'input_value': value,
            'predicted_value': round(prediction, 2),
            'degree': degree,
            'test_score_r2': round(test_score, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plot', methods=['GET'])
def plot():
    try:
        # Get the query parameters from the URL for plotting
        feature_col = request.args.get('feature', 'YEAR')
        target_col = request.args.get('target', 'QLD_total')
        degree = int(request.args.get('degree', 2))
        
        if feature_col not in train_df.columns or target_col not in train_df.columns:
            return jsonify({'error': 'Invalid column name'}), 400

        # Prepare the training data
        X_train = train_df[[feature_col]].values
        y_train = train_df[target_col].values

        # Train the model using polynomial features
        poly = PolynomialFeatures(degree)
        X_train_poly = poly.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Create a range of feature values for a smooth regression curve
        x_min, x_max = X_train.min(), X_train.max()
        x_plot = np.linspace(x_min, x_max, 300).reshape(-1, 1)
        x_plot_poly = poly.transform(x_plot)
        y_plot = model.predict(x_plot_poly)

        # Create the plot with matplotlib
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, color='blue', label='Training Data')
        plt.plot(x_plot, y_plot, color='red', linewidth=2, label='Polynomial Regression')
        plt.xlabel(feature_col)
        plt.ylabel(target_col)
        plt.title(f'Polynomial Regression (Degree = {degree})')
        plt.legend()
        plt.grid(True)

        # Save the plot into an in-memory buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()  # Close the plot to free memory
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
