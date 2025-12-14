from flask import Flask, render_template, request
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load model and preprocessor
model = pickle.load(open('artifacts/model.pkl', 'rb'))
preprocessor = pickle.load(open('artifacts/preprocessor.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        gender = request.form['gender']
        race_ethnicity = request.form['race_ethnicity']
        parental_education = request.form['parental_education']
        lunch = request.form['lunch']
        test_prep = request.form['test_prep']
        reading_score = float(request.form['reading_score'])
        writing_score = float(request.form['writing_score'])

        # Create a DataFrame from user input
        input_df = pd.DataFrame({
            'gender': [gender],
            'race_ethnicity': [race_ethnicity],
            'parental_level_of_education': [parental_education],
            'lunch': [lunch],
            'test_preparation_course': [test_prep],
            'reading_score': [reading_score],
            'writing_score': [writing_score]
        })

        # Preprocess and predict
        processed_input = preprocessor.transform(input_df)
        prediction = model.predict(processed_input)[0]

        return render_template('home.html', result=round(prediction, 2))

    except Exception as e:
        return render_template('home.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
