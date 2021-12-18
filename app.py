from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import joblib
app = Flask(__name__)
liver = pd.read_csv("indian_liver_patient.csv")
model=joblib.load(open('liver-svc.pkl','rb'))
# Pagination
@app.route("/<name>")
def hello_world(name):
    if name == "breast":
        return render_template("./Breast.html")
    if name == "heart":
        return render_template("./Heart.html")
    if name == "liver":
        return render_template("./profile.html")

# Root file of dashboard
@app.route("/")
def index():
    return render_template("./index.html")
# Login route
@app.route("/login")
def login_page():
    return render_template("./Login/login.html")

# Liver prediction form
@app.route("/profile")
def profile_page():
    totalBilirubin=sorted(liver['Total_Bilirubin'].unique())
    direct_Bilirubin=sorted(liver['Direct_Bilirubin'].unique())
    alkaline_Phosphotase=sorted(liver['Alkaline_Phosphotase'].unique())
    alamine_Aminotransferase = sorted(liver['Alamine_Aminotransferase'].unique())
    aspartate_Aminotransferase = sorted(liver['Aspartate_Aminotransferase'].unique())
    total_Protiens = sorted(liver['Total_Protiens'].unique())
    albumin = sorted(liver['Albumin'].unique())
    albumin_and_Globulin_Ratio = sorted(liver['Albumin_and_Globulin_Ratio'].unique())
    return render_template("index.html",
                        totalBilirubin=totalBilirubin,
                         direct_Bilirubin = direct_Bilirubin,
                         alkaline_Phosphotase = alkaline_Phosphotase,
                         alamine_Aminotransferase = alamine_Aminotransferase,
                         aspartate_Aminotransferase = aspartate_Aminotransferase,
                         total_Protiens = total_Protiens,
                         albumin = albumin,
                         albumin_and_Globulin_Ratio = albumin_and_Globulin_Ratio
                        )
# Liver prediction form result
@app.route("/getData",methods=['POST'])
def getData():
    age=request.form.get('age')
    gender=request.form.get('gender')
    totalBilirubin=request.form.get('totalBilirubin')
    Direct_Bilirubin=request.form.get('Direct_Bilirubin')
    alkaline_Phosphotase=request.form.get('alkaline_Phosphotase')
    alamine_Aminotransferase=request.form.get('alamine_Aminotransferase')
    aspartate_Aminotransferase=request.form.get('aspartate_Aminotransferase')
    total_Protiens=request.form.get('total_Protiens')
    albumin_and_Globulin_Ratio=request.form.get('albumin_and_Globulin_Ratio')
    albumin=request.form.get('albumin')
    print(age,gender,totalBilirubin,Direct_Bilirubin,alkaline_Phosphotase,alamine_Aminotransferase,total_Protiens,albumin,albumin_and_Globulin_Ratio)
    prediction=model.predict(pd.DataFrame(columns=["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"],
                              data=np.array([age,gender,totalBilirubin,Direct_Bilirubin,alkaline_Phosphotase,
                               alamine_Aminotransferase,aspartate_Aminotransferase,total_Protiens,albumin,albumin_and_Globulin_Ratio
                              ]).reshape(1, 10)))
    # prediction=model.predict(age,gender,totalBilirubin,Direct_Bilirubin,alkaline_Phosphotase,
    #                            alamine_Aminotransferase,total_Protiens,albumin,albumin_and_Globulin_Ratio)
    print(prediction[0])
    condition = str(prediction[0])
    if condition == "0":
        return "Not Sick"
    elif condition == "1":
        return "Sick"

if __name__ == "__main__":
    app.run(debug=True,port=8000)