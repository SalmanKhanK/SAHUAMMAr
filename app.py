from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import os
import joblib
app = Flask(__name__)
liver = pd.read_csv("indian_liver_patient.csv")
breast_cancer = pd.read_csv("Breast_Cancer.csv")
heart_cancer_csv = pd.read_csv("heart.csv")
hyper_tension_csv = pd.read_csv('FinalData.csv')
diabetes_csv = pd.read_csv('diabetes.csv')

model=joblib.load(open('liver-svc.pkl','rb'))
breastmodel=joblib.load(open('breast-cancer-logreg.pkl','rb'))
heartmodel=joblib.load(open('heart-decTree.pkl','rb'))
hypertension_model=joblib.load(open('hyper-ten-svc.pkl','rb'))
diabetes_model=joblib.load(open('diabetes-logreg.pkl','rb'))

PEOPLE_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
# Pagination
@app.route("/<name>")
def hello_world(name):
    if name == "breast":
        return render_template("./Breast.html")
    if name == "heart":
        return render_template("./Heart.html")
    if name == "liver":
        return render_template("./index.html")
    if name == "hypertension":
        return render_template("./Hypertension.html")
    if name == "diabetes":
        return render_template("./Diabetes.html")

# Root file of dashboard
@app.route("/")
def index():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'heart.jpg')
    return render_template("./Home/home.html", user_image = full_filename)
# Login route
@app.route("/login")
def login_page():
    return render_template("./Login/login.html")

# Liver prediction form
@app.route("/liver")
def profile_page():
    totalBilirubin=sorted(liver['Total_Bilirubin'].unique())
    direct_Bilirubin=sorted(liver['Direct_Bilirubin'].unique())
    alkaline_Phosphotase=sorted(liver['Alkaline_Phosphotase'].unique())
    alamine_Aminotransferase = sorted(liver['Alamine_Aminotransferase'].unique())
    aspartate_Aminotransferase = sorted(liver['Aspartate_Aminotransferase'].unique())
    total_Protiens = sorted(liver['Total_Protiens'].unique())
    albumin = sorted(liver['Albumin'].unique())
    albumin_and_Globulin_Ratio = sorted(liver['Albumin_and_Globulin_Ratio'].unique())
    liverImg = os.path.join(app.config['UPLOAD_FOLDER'], 'heart.jpg')
    return render_template("index.html",
                        totalBilirubin=totalBilirubin,
                         direct_Bilirubin = direct_Bilirubin,
                         alkaline_Phosphotase = alkaline_Phosphotase,
                         alamine_Aminotransferase = alamine_Aminotransferase,
                         aspartate_Aminotransferase = aspartate_Aminotransferase,
                         total_Protiens = total_Protiens,
                         albumin = albumin,
                         albumin_and_Globulin_Ratio = albumin_and_Globulin_Ratio,
                         liverImg=liverImg,
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
#Breast Cancer Prediction 
@app.route("/breast")
def breast_page():
    id=sorted(breast_cancer['id'].unique())
    diagnosis=breast_cancer['diagnosis'].unique()
    radius_mean=sorted(breast_cancer['radius_mean'].unique())
    texture_mean = sorted(breast_cancer['texture_mean'].unique())
    perimeter_mean = sorted(breast_cancer['perimeter_mean'].unique())
    area_mean = sorted(breast_cancer['area_mean'].unique())
    smoothness_mean = sorted(breast_cancer['smoothness_mean'].unique())
    compactness_mean = sorted(breast_cancer['compactness_mean'].unique())
    concavity_mean = sorted(breast_cancer['concavity_mean'].unique())
    concave_points_mean = sorted(breast_cancer['concave points_mean'].unique())
    symmetry_mean = sorted(breast_cancer['symmetry_mean'].unique())
    fractal_dimension_mean = sorted(breast_cancer['fractal_dimension_mean'].unique())
    radius_se = sorted(breast_cancer['radius_se'].unique())
    texture_se = sorted(breast_cancer['texture_se'].unique())
    perimeter_se = sorted(breast_cancer['perimeter_se'].unique())
    area_se = sorted(breast_cancer['area_se'].unique())
    smoothness_se = sorted(breast_cancer['smoothness_se'].unique())
    compactness_se = sorted(breast_cancer['compactness_se'].unique())
    concavity_se = sorted(breast_cancer['concavity_se'].unique())
    concave_points_se = sorted(breast_cancer['concave points_se'].unique())
    symmetry_se = sorted(breast_cancer['symmetry_se'].unique())
    fractal_dimension_se = sorted(breast_cancer['fractal_dimension_se'].unique())
    radius_worst = sorted(breast_cancer['radius_worst'].unique())
    texture_worst = sorted(breast_cancer['texture_worst'].unique())
    perimeter_worst = sorted(breast_cancer['perimeter_worst'].unique())
    area_worst = sorted(breast_cancer['area_worst'].unique())
    smoothness_worst = sorted(breast_cancer['smoothness_worst'].unique())
    compactness_worst = sorted(breast_cancer['compactness_worst'].unique())
    concavity_worst = sorted(breast_cancer['concavity_worst'].unique())
    concave_points_worst = sorted(breast_cancer['concave points_worst'].unique())
    symmetry_worst = sorted(breast_cancer['symmetry_worst'].unique())
    fractal_dimension_worst = sorted(breast_cancer['fractal_dimension_worst'].unique())
    print(diagnosis,"Diagonosis")
    return render_template("Breast.html",
                        id=id,
                        diagnosis=diagnosis,
                        radius_mean=radius_mean,
                        texture_mean= texture_mean,
                        perimeter_mean = perimeter_mean,
                        area_mean = area_mean,
                        smoothness_mean = smoothness_mean,
                        compactness_mean = compactness_mean,
                        concavity_mean = concavity_mean,
                        concave_points_mean = concave_points_mean,
                        symmetry_mean = symmetry_mean,
                        fractal_dimension_mean = fractal_dimension_mean,
                        radius_se = radius_se,
                        texture_se = texture_se,
                        perimeter_se = perimeter_se,
                        area_se = area_se,
                        smoothness_se = smoothness_se,
                        compactness_se = compactness_se,
                        concavity_se = concavity_se,
                        concave_points_se = concave_points_se,
                        symmetry_se = symmetry_se,
                        fractal_dimension_se = fractal_dimension_se,
                        radius_worst = radius_worst,
                        texture_worst = texture_worst,
                        perimeter_worst = perimeter_worst,
                        area_worst = area_worst,
                        smoothness_worst = smoothness_worst,
                        compactness_worst = compactness_worst,
                        concavity_worst = concavity_worst,
                        concave_points_worst = concave_points_worst,
                        symmetry_worst = symmetry_worst,
                        fractal_dimension_worst = fractal_dimension_worst 
                        )

# Heart Prediction 
@app.route("/heart")
def heart_page():
    cp=heart_cancer_csv['cp'].unique()
    trestbps=sorted(heart_cancer_csv['trestbps'].unique())
    chol=sorted(heart_cancer_csv['chol'].unique())
    fbs=sorted(heart_cancer_csv['fbs'].unique())
    restecg=sorted(heart_cancer_csv['restecg'].unique())
    thalach=sorted(heart_cancer_csv['thalach'].unique())
    exang=sorted(heart_cancer_csv['exang'].unique())
    oldpeak=sorted(heart_cancer_csv['oldpeak'].unique())
    slope=sorted(heart_cancer_csv['slope'].unique())
    ca=sorted(heart_cancer_csv['ca'].unique())
    thal=sorted(heart_cancer_csv['thal'].unique())
    target=sorted(heart_cancer_csv['target'].unique())
    print(cp,trestbps,chol,"Hearted")
    return render_template("Heart.html",
                        cp=cp,
                        trestbps=trestbps, 
                        chol=chol, 
                        fbs=fbs,
                        restecg=restecg, 
                        thalach=thalach,
                        exang=exang, 
                        oldpeak=oldpeak, 
                        slope=slope, 
                        ca=ca, 
                        thal=thal, 
                        target=target
                        )

@app.route("/getHeartPred",methods=['POST'])
def getHeartData():
    age=request.form.get('age')
    sex=request.form.get('sex')
    cp=request.form.get('cp')
    trestbps=request.form.get('trestbps')
    chol=request.form.get('chol')
    fbs=request.form.get('fbs')
    restecg=request.form.get('restecg')
    thalach=request.form.get('thalach')
    exang=request.form.get('exang')
    oldpeak=request.form.get('oldpeak')
    slope=request.form.get('slope')
    ca=request.form.get('ca')
    thal=request.form.get('thal')
    target=request.form.get('target')
    print(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,
                    oldpeak,slope,ca,thal,target)
    predictionH=heartmodel.predict(pd.DataFrame(
                    columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang",
                    "oldpeak","slope","ca","thal","target"],
                              data=np.array([age,sex,cp,trestbps,chol,
                               fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
                              ]).reshape(1, 14)))
    # prediction=model.predict(age,gender,totalBilirubin,Direct_Bilirubin,alkaline_Phosphotase,
    #                            alamine_Aminotransferase,total_Protiens,albumin,albumin_and_Globulin_Ratio)
    print("+++++++++++++++++++++++++++++++++++++++++")
    print(predictionH,"Predict The Modal")
    print("+++++++++++++++++++++++++++++++++++++++++")
    print("+++++++++++++++++++++++++++++++++++++++++")
    # condition = str(prediction[0])
    # if condition == "0":
    #     return "Not Sick"
    # elif condition == "1":
    #     return "Sick"
# Gender  Age  Severity  BreathShortness  VisualChanges  NoseBleeding
# Whendiagnoused  Systolic  Diastolic 
@app.route("/hypertension")
def hypertension_page():
    Severity=hyper_tension_csv['Severity'].unique()
    BreathShortness=sorted(hyper_tension_csv['BreathShortness'].unique())
    VisualChanges=sorted(hyper_tension_csv['VisualChanges'].unique())
    NoseBleeding=sorted(hyper_tension_csv['NoseBleeding'].unique())
    Whendiagnoused=sorted(hyper_tension_csv['Whendiagnoused'].unique())
    Systolic=sorted(hyper_tension_csv['Systolic'].unique())
    Diastolic=sorted(hyper_tension_csv['Diastolic'].unique())
    hypertensionImg = os.path.join(app.config['UPLOAD_FOLDER'], 'hypertension.jpg')
    print(Severity,BreathShortness,"HYper")
    return render_template("Hypertension.html",
                    Severity=Severity, 
                    BreathShortness=BreathShortness,  
                    VisualChanges=VisualChanges,  
                    NoseBleeding=NoseBleeding,
                    Whendiagnoused=Whendiagnoused,  
                    Systolic=Systolic,  
                    Diastolic=Diastolic,
                    hypertensionImg=hypertensionImg,
                )

@app.route("/getHyperPred",methods=['POST'])
def getHyperPred():
    Gender=request.form.get('Gender')
    Age=request.form.get('Age')
    Severity=request.form.get('Severity')
    BreathShortness=request.form.get('BreathShortness')
    VisualChanges=request.form.get('VisualChanges')
    NoseBleeding=request.form.get('NoseBleeding')
    Whendiagnoused=request.form.get('Whendiagnoused')
    Systolic=request.form.get('Systolic')
    Diastolic=request.form.get('Diastolic')
   
    print(Gender,Age,Severity,BreathShortness,VisualChanges,NoseBleeding,Whendiagnoused,Systolic,Diastolic)
    predictionHy=hypertension_model.predict(pd.DataFrame(
                    columns=["Gender","Age","Severity","BreathShortness",
                    "VisualChanges","NoseBleeding","Whendiagnoused","Systolic","Diastolic"],
                              data=np.array([Gender,Age,Severity,BreathShortness,VisualChanges,
                               NoseBleeding,Whendiagnoused,Systolic,Diastolic]).reshape(1, 9)))
    print("+++++++++++++++++++++++++++++++++++++++++")
    condition = str(predictionHy[0])
    if condition == "0":
        return "Not Sick"
    elif condition == "1":
        return "Sick"

@app.route("/diabetes")
def diabetes_page():
    # Pregnancies=diabetes_csv['Pregnancies'].unique()
    # Glucose=sorted(diabetes_csv['Glucose'].unique())
    # BloodPressure=sorted(diabetes_csv['BloodPressure'].unique())
    # SkinThickness=sorted(diabetes_csv['SkinThickness'].unique())
    # Insulin=sorted(diabetes_csv['Insulin'].unique())
    # BMI=sorted(diabetes_csv['BMI'].unique())
    # DiabetesPedigreeFunction=sorted(diabetes_csv['DiabetesPedigreeFunction'].unique())
    DiabImg = os.path.join(app.config['UPLOAD_FOLDER'], 'hypertension.jpg')
    return render_template("Diabetes.html",DiabImg=DiabImg)
@app.route("/getDiabetesPred",methods=['POST'])
def getDiabetesPred():
    Age=request.form.get('Age')
    Pregnancies=request.form.get('Pregnancies')
    Glucose=request.form.get('Glucose')
    BloodPressure=request.form.get('BloodPressure')
    SkinThickness=request.form.get('SkinThickness')
    Insulin=request.form.get('Insulin')
    BMI=request.form.get('BMI')
    DiabetesPedigreeFunction=request.form.get('DiabetesPedigreeFunction')
    # print("+++++++++++++++++++++++++++++++++++++++++")
    # print(Age,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction)
    # print("+++++++++++++++++++++++++++++++++++++++++")
    predictionDiabetes=diabetes_model.predict(pd.DataFrame(
                    columns=["Age","Pregnancies","Glucose",
                    "BloodPressure","SkinThickness","Insulin","BMI","Diastolic"],
                              data=np.array([Age,Pregnancies,Glucose,DiabetesPedigreeFunction,
                               SkinThickness,Insulin,BMI,DiabetesPedigreeFunction]).reshape(1, 8)))
    print("+++++++++++++++++++++++++++++++++++++++++")
    condition = str(predictionDiabetes[0])
    print(condition)
    print("++++++++++++++++++++++++++++++++++++++++")
    if condition == "0":
        return "NO"
    elif condition == "1":
        return "Yes"
if __name__ == "__main__":
    app.run(debug=True,port=8000)