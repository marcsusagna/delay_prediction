from flask import redirect  # Importing the class flask
from flask import Flask, url_for, render_template, request
from src.serve_app_utils import predict_contrafactual_test
from src.serve_app_utils import summarize_model_metrics
import os
# Instantiate API
app = Flask(__name__, template_folder='templates')


def process_all_increase_input(user_input):
    all_increase = [float(x) for x in user_input.split(" ")]
    return all_increase


@app.route('/predict/<name>', methods=['GET', 'POST'])
# Creating a function named predict
def predict(name):
    all_increases = process_all_increase_input(name)
    model_results = predict_contrafactual_test(all_increases)
    table = model_results.to_html(index=False)
    return render_template("table_template.html", table=table)

@app.route('/metrics/<name>', methods=['GET', 'POST'])
# Creating a function named metrics
def metrics(name):
    my_df = summarize_model_metrics()
    table = my_df.to_html(index=False)
    return render_template("table_template.html", table=table)

@app.route('/login', methods=['GET', 'POST'])
# Creating a function named login
def login():
    if request.method == 'POST':
        input_ = request.form['increases']
        if (input_ == "model metrics"):
            return redirect(url_for('metrics', name=input_))
        else:
            return redirect(url_for('predict', name=input_))
    else:
        return "INVALID"


# Programs executes from here in a development server (locally on your system)
# with debugging enabled.

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
