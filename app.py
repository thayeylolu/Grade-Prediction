import sys
sys.path.append('./mlscripts')

import model as md

#   imported the needed packages (libraries) 
import numpy as np
from flask import Flask, request, render_template

#   Initialize the flask App
app = Flask(__name__) 

#   Assigned the pickled model (m.pkl) to a var name :model
#   it is read (rb), opened(open()) and loaded (load())

#   created a route ; which is the first pagr you see
#   when it launches. 

@app.route('/')
#   The default method here is 'Get'

#   The home function (home())
def home():
    '''
        The home() function does the following
        - Renders the main.html which is in the templates folder
    '''
    return render_template('main.html')

#   The '/predict' route uses the post method
#   it renders the main html and also displaces the predicted response
#   render_template() actually use a positional argument (context) which is assigned solution()

@app.route('/predict',methods=['POST'])

#   The predict function
def predict():
    '''
        This renders the result.
        It checks the values in the form value then converts them to integers.
        Stores them in a list 'int_features'
        The final feature is a numpy representation of the int_features
        The model's predict function is called, 
        final_features passed as the parameter.
    '''

    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features)
    prediction = 'Course Grade: {}'.format(md.ClassifyGrade(final_features))

    return render_template('main.html', prediction_text = prediction)

#   This is the main global functionwhich runs the flask app
if __name__ == "__main__":
    app.run(debug=True)