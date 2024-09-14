from flask import Flask, render_template, request, session, redirect
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from keras.models import model_from_json
import random
import os
from werkzeug.utils import secure_filename
import shutil
from flask_pymongo import PyMongo
import requests
import tensorflow as tf
# import json

app = Flask(__name__, static_folder="Static", template_folder="Templates")
app.secret_key = 'super-secret-key'

# Journal Setup from MongoDB

app.config["MONGO_URI"] = "mongodb+srv://prameyamohanty10:PrAmEyAmOhAnTy%40100808@mydatabase.gyir6sh.mongodb.net/FemCare?retryWrites=true&w=majority&appName=MyDatabase/FemCare"
mongo = PyMongo(app)

# ========================================================================================

# Journal Setup from JSON

# with open("FemCare.articles.json", "r") as journal:
#     data = json.load(journal)

# ========================================================================================

# Cervical Cancer Setup

model = joblib.load("models\cervical_cancer.joblib")
features = [["Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies", "Smokes (years)", "Smokes (packs/year)", "Hormonal Contraceptives (years)", "IUD (years)", "STDs: Number of diagnosis"], ["condylomatosis", "cervical condylomatosis", "vaginal condylomatosis", "vulvo-perineal condylomatosis", "syphilis", "pelvic inflammatory disease", "genital herpes", "molluscum contagiosum", "AIDS", "HIV", "Hepatitis B", "HPV"], ["Cancer", "CIN", "hpv"]]

mainsfeatures = ["Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies", "Smokes (years)", "Smokes (packs/year)", "Hormonal Contraceptives (years)", "IUD (years)", "STDs", "STDs (number)", "condylomatosis", "cervical condylomatosis", "vaginal condylomatosis", "vulvo-perineal condylomatosis", "syphilis", "pelvic inflammatory disease", "genital herpes", "molluscum contagiosum", "AIDS", "HIV", "Hepatitis B", "HPV", "STDs: Number of diagnosis", "Cancer", "CIN", "HPV"]

# ========================================================================================

# Breast Cancer Setup

json_file = open(r"models\breast_cancer.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(r"models\breast_cancer.h5")
loaded_model.compile(
    loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

def save_and_get_pred_img(image):
    defrance = str(random.randint(1, 100000))
    file = r"C:\Users\prame\OneDrive - 45ktw4\Documents\Intel India AI Impact Festival 2024\FemCare"
    file_path = os.path.join(file, defrance)
    os.makedirs(file_path)
    filename = secure_filename(image.filename)
    next_file_path = os.path.join(file_path, defrance)
    os.makedirs(next_file_path)
    UPLOAD_FOLDER = next_file_path
    app.config["IMAGE_UPLOADS"] = UPLOAD_FOLDER
    image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
    return file_path

class Api_service:
    def __init__(self, img_file_path):
        self.img_file_path = img_file_path

    def prediction_function(self):
        data_generation = ImageDataGenerator(rescale=1.0 / 255)
        predict_generation = data_generation.flow_from_directory(
            self.img_file_path,
            target_size=(25, 25),
            batch_size=10,
            class_mode="categorical",
        )

        prediction = loaded_model.predict_generator(predict_generation)
        has_no_cancer = (
            "The percentage of no cancer : "
            + str(round(prediction[0][1] * 100, 2))
            + "%"
        )
        has_cancer = (
            "The percentage of being affected by Breast Cancer is " + str(round(prediction[0][0] * 100, 2)) + "%."
        )
        shutil.rmtree(self.img_file_path)
        return has_cancer, has_no_cancer
    
# ========================================================================================

# PCOS Setup

pcos_features = ['Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Blood Group', 'Pulse rate(bpm)', 'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)', 'Marraige Status (Yrs)', 'Pregnant(Y/N)', 'No. of aborptions', 'I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)']

values = [33, 89.0, 163.0, 33.5, 12, 78, 22, 10.8, 4, 7, 6.0, 1, 0, 381.99, 15.0, 3.45, 0.72, 4.791666667, 46, 40, 0.8695652174, 0.73, 0.7, 15.23, 16.7, 0.28, 100.0, 1, 0, 1, 1, 1, 1.0, 0, 120, 80, 13, 12, 19.0, 18.0, 7.9]

# ========================================================================================

# Ovarian Cancer Setup

target_names = {
    'CC': "Clear Cell Carcinoma",
    'EC': "Endometrioid Carcinoma",
    'HGSC': "High Grade Serous Carcinoma",
    'LGSC': "Low Grade Serous Carcinoma",
    'MC': 'Mucinous Carcinoma'
}

model = tf.keras.models.load_model("models/ovarian-cancer.h5")

def load_and_prep_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size = [img_shape, img_shape])
    img = img/255.
    return img

def pred_and_plot(model, filename, target_names):
    tn = list(target_names.keys())
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))
    pred_class = tn[int(tf.round(pred)[0][0])]
    return f"The class of Ovarian Cancer is {target_names[pred_class]}"

# ========================================================================================

@app.route("/")
def home():
    return render_template("index.html", session=session)

@app.route("/cervical-cancer", methods=["GET", "POST"])
def cervical_cancer():
    values = []
    if request.method == "POST":
        for item in mainsfeatures:
            resp = request.form.get(item)
            if resp != None:
                values.append(resp)
        sum = 0
        for item in features[1]:
            sum += (int(request.form.get(item)))
        if sum>0:
            values.insert(8, "1")
            values.insert(9, str(sum))
        else:
            values.insert(8, "0")
            values.insert(9, str(sum))
        vals = [float(value) for value in values]
        prediction = model.predict(np.array([vals]))
        
        return render_template("cervical-cancer.html", features=features, values=[], cc_result=str(prediction[0]*25) + "%", session=session)
    return render_template("cervical-cancer.html", features=features, values=values, cc_result="", session=session)

@app.route("/pcos", methods=["GET", "POST"])
def pcos():
    if request.method == "POST":
        pred = [[]]
        for item in pcos_features:
            pred[0].append(request.form.get(item))

        pcos_model = joblib.load(r"models\pcos.joblib")
        prediction = list(pcos_model.predict(pred))[0]
        if prediction == 1:
            return render_template("pcos.html", pcos_features=pcos_features, values=values, pcos_result="There, session=session is a high chance of PCOS for these parameters.")
        else:
            return render_template("pcos.html", pcos_features=pcos_features, values=values, pcos_result="There, session=session is a low chance of PCOS for these parameters.")
    return render_template("pcos.html", pcos_features=pcos_features, values=values, pcos_result="", session=session)

@app.route("/breast-cancer", methods=["GET", "POST"])
def breast_cancer():
    if request.method == "POST":
        image = request.files["img"]
        img_file_path = save_and_get_pred_img(image)
        predict_img = Api_service(img_file_path)
        has_cancer, has_no_cancer = predict_img.prediction_function()
        return render_template("breast-cancer.html", bc_result=has_cancer, session=session)
    return render_template("breast-cancer.html", bc_result="", session=session)
    
@app.route("/ovarian-cancer", methods=["GET", "POST"])
def ovarian_cancer():
    if request.method == "POST":
        image = request.files["img"]
        filename = secure_filename(image.filename)
        file_path = os.path.join('Uploads', filename)
        image.save(file_path)
        result = pred_and_plot(model, file_path, target_names)
        os.remove(file_path)
        return render_template("ovarian-cancer.html", oc_result=result, session=session)
    return render_template("ovarian-cancer.html", oc_result="", session=session)
    
@app.route("/journal", methods=["GET", "POST"])
def journal():
    # From MongoDB
    articles = mongo.db.articles.find({})
    # From JSON
    # articles = data
    return render_template("journal.html", articles=articles, session=session)

@app.route("/article/<article_slug>")
def article(article_slug):
    # From MongoDB
    article = mongo.db.articles.find_one({ "slug": article_slug })
    # From JSON
    # article = None
    # for item in data:
    #     if item["slug"] == article_slug:
    #         article = item
    return render_template("article.html", article=article, session=session)

@app.route("/add", methods=["POST"])
def add():
    article = request.json
    mongo.db.articles.insert_one(article)
    return "Article added successfully."

@app.route("/get", methods=["GET", "POST"])
def get():
    reqUrl = "https://femcare-chatbot.vercel.app/get"
    msg = request.form["msg"]

    headersList = {
    "Accept": "*/*",
    "User-Agent": "Thunder Client (https://www.thunderclient.com)",
    "Content-Type": "multipart/form-data; boundary=kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A" 
    }

    payload = f"--kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A\r\nContent-Disposition: form-data; name=\"msg\"\r\n\r\n{msg}\r\n--kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A--\r\n"

    response = requests.request("POST", reqUrl, data=payload,  headers=headersList)
    return response.text

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        data = dict(request.form)
        user = mongo.db.users.find_one(data)
        if (user != None):
            session['user'] = {"username": user["username"], "email": user["email"]}
        return redirect("/")

    print(session)
    return render_template("login.html", session=session)

@app.route("/logout")
def logout():
    session.pop('user')
    return redirect('/signin')

if __name__ == "__main__":
    app.run(debug=True, host="192.168.0.105", port=5001)
    
