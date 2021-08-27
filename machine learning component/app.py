from flask import Flask, request, redirect, url_for, flash, jsonify
from keras.models import load_model
import pandas as pd


app = Flask(__name__)


@app.route('/api/', methods=['POST'])
def makecalc():
    json_ = request.json()
    data = pd.DataFrame(json_)
    model = load_model("my_model.h5")
    prediction = model.predict(data,batch_size=1)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')