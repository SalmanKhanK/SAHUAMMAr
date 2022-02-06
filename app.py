from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import os
import joblib
app = Flask(__name__)
liver = pd.read_csv("indian_liver_patient.csv")
breast_cancer = pd.read_csv("Breast_Cancer.csv")
heart_cancer_csv = pd.read_csv("heart1.csv")
hyper_tension_csv = pd.read_csv('FinalData.csv')
diabetes_csv = pd.read_csv('diabetes.csv')

model=joblib.load(open('liver-svc.pkl','rb'))
heartmodel=joblib.load(open('heart-decTree.pkl','rb'))
hypertension_model=joblib.load(open('hyper-ten-svc.pkl','rb'))
diabetes_model=joblib.load(open('diabetes-logreg.pkl','rb'))
breastcancer_model=joblib.load(open('breast-cancer.pkl','rb'))

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
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'webimg.jpg') 
    logo_img = os.path.join(app.config['UPLOAD_FOLDER'], 'final-logo.PNG')

    return render_template("./Home/home.html", user_image = full_filename, logo_img = logo_img)

@app.route("/about")
def about():
    logo_img = os.path.join(app.config['UPLOAD_FOLDER'], 'final-logo.PNG')
    about_big_img = os.path.join(app.config['UPLOAD_FOLDER'], 'about-big-img.jpg')
    four_top_img = os.path.join(app.config['UPLOAD_FOLDER'], 'four-top-img.jpg')
    four_bottom_img = os.path.join(app.config['UPLOAD_FOLDER'], 'four-bottom-img.jpg')
    footer = os.path.join(app.config['UPLOAD_FOLDER'], 'footer.jpg')

    return render_template("./about.html", 
    logo_img=logo_img,
    about_big_img = about_big_img, 
    four_top_img = four_top_img,
    four_bottom_img=four_bottom_img,
    footer=footer

    
    )
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
    brCancerImg = os.path.join(app.config['UPLOAD_FOLDER'], 'heart.jpg')
    return render_template("Breast.html",brCancerImg=brCancerImg)

# Heart Prediction
# id', 'diagnosis', 'radius_mean', 'perimeter_mean', 'area_mean',
#        'smoothness_mean', 'concavity_mean', 'concave points_mean',
#        'symmetry_mean', 'area_se', 'radius_worst', 'perimeter_worst',
#        'area_worst', 'smoothness_worst', 'concavity_worst',
#        'concave points_worst', 'symmetry_worst
@app.route("/getBreastPred",methods=['POST'])
def getBreastPred():
    id=request.form.get('id')
    radius_mean=request.form.get('radius_mean')
    perimeter_mean=request.form.get('perimeter_mean')
    area_mean=request.form.get('area_mean')
    smoothness_mean=request.form.get('smoothness_mean')
    concavity_mean=request.form.get('concavity_mean')
    concave_points_mean=request.form.get('concave points_mean')
    symmetry_mean=request.form.get('symmetry_mean')
    area_se=request.form.get('area_se') 
    radius_worst=request.form.get('radius_worst')   
    perimeter_worst=request.form.get('perimeter_worst')
    area_worst=request.form.get('area_worst')   
    smoothness_worst=request.form.get('smoothness_worst')
    concavity_worst=request.form.get('concavity_worst')
    concave_points_worst=request.form.get('concave points_worst')   
    symmetry_worst=request.form.get('symmetry_worst')
    print("++++++++++++++++++++++++++++++++++++++++++++")
    print(id,radius_mean,perimeter_mean,area_mean,smoothness_mean,concavity_mean,
    concave_points_mean,symmetry_mean,area_se,radius_worst,perimeter_worst,area_worst,
    smoothness_worst,concavity_worst,concave_points_worst,symmetry_worst)
    predictionBr=breastcancer_model.predict(pd.DataFrame(
                    columns=["id","radius_mean","perimeter_mean","area_mean","smoothness_mean",
                    "concavity_mean","concave_points_mean","symmetry_mean","area_se",'radius_worst',
                    "perimeter_worst","area_worst","smoothness_worst","concavity_worst","concave_points_worst","symmetry_worst"],
                              data=np.array([id,radius_mean,perimeter_mean,area_mean,
                              smoothness_mean,concavity_mean,
                               concave_points_mean,symmetry_mean,area_se,radius_worst,perimeter_worst,area_worst,
                               smoothness_worst,concavity_worst,concave_points_worst,symmetry_worst]).reshape(1, 16)))
    print("+++++++++++++++++++++++++++++++++++++++++")
    print(predictionBr,"Predict The Modal")
    print("+++++++++++++++++++++++++++++++++++++++++")
    condition = str(predictionBr[0])
    if condition == "1":
        return "Breast Cancer"
    elif condition == "0":
        return "No Breast Cancer"

@app.route("/heart")
def heart_page():
    Sex=heart_cancer_csv['Sex'].unique()
    ChestPainType=sorted(heart_cancer_csv['ChestPainType'].unique())
    RestingECG=sorted(heart_cancer_csv['RestingECG'].unique())
    ExerciseAngina=sorted(heart_cancer_csv['ExerciseAngina'].unique())
    ST_Slope=sorted(heart_cancer_csv['ST_Slope'].unique())
    FastingBS=sorted(heart_cancer_csv['FastingBS'].unique())
    heartImg = os.path.join(app.config['UPLOAD_FOLDER'], 'heart.jpg')
    return render_template("Heart.html",
                        Sex=Sex,
                        ChestPainType=ChestPainType, 
                        RestingECG=RestingECG, 
                        ExerciseAngina=ExerciseAngina,
                        ST_Slope=ST_Slope, 
                        FastingBS=FastingBS,
                        heartImg=heartImg
                        )

@app.route("/getHeartPred",methods=['POST'])
def getHeartPred():
    Age=request.form.get('Age')
    Sex=request.form.get('Sex')
    ChestPainType=request.form.get('ChestPainType')
    RestingBP=request.form.get('RestingBP')
    Cholesterol=request.form.get('Cholesterol')
    FastingBS=request.form.get('FastingBS')
    RestingECG=request.form.get('RestingECG')
    MaxHR=request.form.get('MaxHR')
    ExerciseAngina=request.form.get('ExerciseAngina') 
    Oldpeak=request.form.get('Oldpeak')   
    ST_Slope=request.form.get('ST_Slope')

    print(Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,
                    ST_Slope)
    predictionH=heartmodel.predict(pd.DataFrame(
                    columns=["Age","Sex","ChestPainType","RestingBP","Cholesterol",
                    "FastingBS","RestingECG","MaxHR","ExerciseAngina",'Oldpeak',
                    "ST_Slope"],
                              data=np.array([Age,Sex,ChestPainType,RestingBP,Cholesterol,
                               FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]).reshape(1, 11)))
    print("+++++++++++++++++++++++++++++++++++++++++")
    print(predictionH,"Predict The Modal")
    print("+++++++++++++++++++++++++++++++++++++++++")
    # print("+++++++++++++++++++++++++++++++++++++++++")
    condition = str(predictionH[0])
    if condition == "1":
        return "Yes Diaseas"
    elif condition == "0":
        return "No Diaseas"
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