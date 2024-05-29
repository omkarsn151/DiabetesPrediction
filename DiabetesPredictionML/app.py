from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  

# Load the trained model and scaler
model = joblib.load('diabetes_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    result = session.pop('result', None)  
    # form_data = session.get('form_data')
    form_data = session.pop('form_data',None)
    return render_template('index.html', 
                           result=result,
                           form_data = form_data
                           )

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data

        form_data = {
            'pregnancy': int(request.form['pregnancy']),
            'glucose' : float(request.form['glucose']),
            'blood_pressure' : float(request.form['blood_pressure']),
            'skin_thickness' : float(request.form['skin_thickness']),
            'insulin' : float(request.form['insulin']),
            'bmi' : float(request.form['bmi']),
            'diabetes_pedigree_function' : float(request.form['diabetes_pedigree_function']),
            'age' : float(request.form['age']),


        }

        session['form_data'] = form_data
        
        


        form_data = {key: float(value) for key, value in form_data.items()}


        # Create a numpy array for the input
        user_input = np.array([[
                        form_data['pregnancy'], 
                        form_data['glucose'], 
                        form_data['blood_pressure'], 
                        form_data['skin_thickness'], 
                        form_data['insulin'], 
                        form_data['bmi'], 
                        form_data['diabetes_pedigree_function'], 
                        form_data['age'], 
                        ]])

        # Standardize the user input
        user_input_scaled = scaler.transform(user_input)

        # Make a prediction
        prediction = model.predict(user_input_scaled)

        # Prepare the result message
        if prediction[0] == 1:
            result = "The model predicts that you have Diabetes"
        else:
            result = "The model predicts that you do not have Diabetes"

        
        session['result'] = result

        

        # Redirect to home
        return redirect(url_for('home'))
    


if __name__ == '__main__':
    app.run(debug=True)






































# ---------------------------------------------------------------------




# from flask import Flask, render_template, request
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load the trained model and scaler
# model = joblib.load('diabetes_prediction_model.pkl')
# scaler = joblib.load('scaler.pkl')

# @app.route('/')
# def home():
#     return render_template('index.html', 
#                            result = None,
#                         #    pregnancy = "",
#                         #    glucose = "",
#                         #    blood_pressure = "",
#                         #    skin_thickness = "",
#                         #    insulin = "",
#                         #    bmi = "",
#                         #    diabetes_pedigree_function = "",
#                         #    age = "",
#                            )

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
        
#         # Get the form data
#         pregnancy = int(request.form['pregnancy'])
#         glucose = float(request.form['glucose'])
#         blood_pressure = float(request.form['blood_pressure'])
#         skin_thickness = float(request.form['skin_thickness'])
#         insulin = float(request.form['insulin'])
#         bmi = float(request.form['bmi'])
#         diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
#         age = float(request.form['age'])

#         # Create a numpy array for the input
#         user_input = np.array([[pregnancy, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

#         # Standardize the user input
#         user_input_scaled = scaler.transform(user_input)

#         # Make a prediction
#         prediction = model.predict(user_input_scaled)

#         # Prepare the result message
#         if prediction[0] == 1:
#             result = "The model predicts that you have Diabetes"

#         else:
#             result = "The model predicts that you do not have Diabetes"

#         return render_template(
#                         'index.html', 
#                         result=result,
#                         # pregnancy=pregnancy, 
#                         # glucose=glucose, 
#                         # blood_pressure=blood_pressure, 
#                         # skin_thickness=skin_thickness, 
#                         # insulin=insulin, 
#                         # bmi=bmi, 
#                         # diabetes_pedigree_function=diabetes_pedigree_function, 
#                         # age=age
#                         )
    
  



# if __name__ == '__main__':
#     app.run(debug=True, port=5002)




# ------------------------------------------------------------------------------------------

# impppppppp



# from flask import Flask, render_template, request, redirect, url_for, session
# import joblib
# import numpy as np

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Change this to a random secret key

# # Load the trained model and scaler
# model = joblib.load('diabetes_prediction_model.pkl')
# scaler = joblib.load('scaler.pkl')

# @app.route('/')
# def home():
#     result = session.pop('result', None)  # Get and remove the result from session
#     return render_template('index.html', 
#                            result=result,
#                            pregnancy = "",
#                            glucose = "",
#                            blood_pressure = "",
#                            skin_thickness = "",
#                            insulin = "",
#                            bmi = "",
#                            diabetes_pedigree_function = "",
#                            age = "",
#                            )

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get the form data
#         pregnancy = int(request.form['pregnancy'])
#         glucose = float(request.form['glucose'])
#         blood_pressure = float(request.form['blood_pressure'])
#         skin_thickness = float(request.form['skin_thickness'])
#         insulin = float(request.form['insulin'])
#         bmi = float(request.form['bmi'])
#         diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
#         age = float(request.form['age'])

#         # Create a numpy array for the input
#         user_input = np.array([[pregnancy, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

#         # Standardize the user input
#         user_input_scaled = scaler.transform(user_input)

#         # Make a prediction
#         prediction = model.predict(user_input_scaled)

#         # Prepare the result message
#         if prediction[0] == 1:
#             result = "The model predicts that you have Diabetes"
#         else:
#             result = "The model predicts that you do not have Diabetes"

#         # Store the result in session
#         session['result'] = result

        

#         # Redirect to home
#         return redirect(url_for('home'))

# if __name__ == '__main__':
#     app.run(debug=True, port=5002)







# ----------------------------------------------------------------------------------------


















# -------------------------------------------






# from flask import Flask, render_template, request, redirect, url_for
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load the trained model and scaler
# model = joblib.load('diabetes_prediction_model.pkl')
# scaler = joblib.load('scaler.pkl')

# @app.route('/')
# def home():
#     result = request.args.get('result', None)
#     return render_template('index.html', result=result)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get the form data
#         pregnancy = int(request.form['pregnancy'])
#         glucose = float(request.form['glucose'])
#         blood_pressure = float(request.form['blood_pressure'])
#         skin_thickness = float(request.form['skin_thickness'])
#         insulin = float(request.form['insulin'])
#         bmi = float(request.form['bmi'])
#         diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
#         age = float(request.form['age'])

#         # Create a numpy array for the input
#         user_input = np.array([[pregnancy, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

#         # Standardize the user input
#         user_input_scaled = scaler.transform(user_input)

#         # Make a prediction
#         prediction = model.predict(user_input_scaled)

#         # Prepare the result message
#         if prediction[0] == 1:
#             result = "The model predicts that you have Diabetes"
#         else:
#             result = "The model predicts that you do not have Diabetes"

#         # Redirect to home with result as query parameter
#         return redirect(url_for('home', result=result))

# if __name__ == '__main__':
#     app.run(debug=True, port=5002)


