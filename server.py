from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import CountVectorizer
import pickle

app = Flask(__name__)

sugar_model = None
carbs_model = None
tfidf_vect = None

def init():
    global inputs_dc, prediction_dc
    from sklearn.externals import joblib
 
    features = ['_269 Sugars, total (g)', '_205 Carbohydrate (g)']

    # Load models
    global sugar_model
    global carbs_model
    global tfidf_vect
    sugar_model = joblib.load('./nutritionSearch/ipython/output/{0}_model.pkl'.format(features[0]))
    carbs_model = joblib.load('./nutritionSearch/ipython/output/{0}_model.pkl'.format(features[1]))
    tfidf_vect = CountVectorizer(decode_error="replace",vocabulary=pickle.load(
                                open("./nutritionSearch/ipython/output/feature.pkl", "rb")))

    
def run(input_text):    
    x_tfidf = tfidf_vect.transform(input_text)

    sugar_pred = sugar_model.predict(x_tfidf)
    carbs_pred = carbs_model.predict(x_tfidf)

    return [sugar_pred, carbs_pred]

@app.route('/')
def hello_world():
    return 'Welcome to EatGeek!'

@app.route('/nutrition')
def nutrition():
    init()

    description = request.args.get('description') or request.args.get('desciption')

    sugar_pred, carbs_pred = run([description])
    
    sugar_ranges = {1: '0-7 g', 2: '8-19 g', 3: '20-35 g', 4: '36-58 g', 5: '59-100 g'}
    carbs_ranges = {1: '0-10 g', 2: '11-24 g', 3: '25-43 g', 4: '44-66 g', 5: '67-100 g'}

    sugar_pred = sugar_ranges[sugar_pred[0]]
    carbs_pred = carbs_ranges[carbs_pred[0]]

    return jsonify({"sugar": sugar_pred, "carbohydrates": carbs_pred})
