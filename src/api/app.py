"""
Simple API to query the model
"""


from pathlib import Path
from flask import Flask, jsonify, request, render_template
import os
import pandas as pd

import numpy as np

from src.utils.modelling import ArtifactSaverLoader

PROJECT_DIR = Path(__file__).resolve().parents[2]


artifact_loader = ArtifactSaverLoader(models_filepath=PROJECT_DIR.joinpath('models'))

model = artifact_loader.load_artifact('linear-sdg-reg-2020-11-22T17H-22M.p')

app = Flask(__name__)
port = int(os.getenv('PORT', '3000'))


@app.route('/', methods=['GET'])
def index():
    msg = """
    curl -X POST -H "Content-Type: application/json" -d @./example_query_api.json localhost:3000/get_house_price
            """
    return msg


@app.route('/get_house_price', methods=['POST'])
def get_house_price():
    if request.method == 'POST':
        try:
            data = request.get_json(force=1)
            df = pd.DataFrame({'CRIM': [data.get('CRIM'), ],
                               'ZN': [data.get('ZN'), ],
                               'INDUS': [data.get('INDUS'), ],
                               'CHAS': [data.get('CHAS'), ],
                               'NOX': [data.get('NOX'), ],
                               'RM': [data.get('RM'), ],
                               'AGE': [data.get('AGE'), ],
                               'DIS': [data.get('DIS'), ],
                               'RAD': [data.get('RAD'), ],
                               'TAX': [data.get('TAX'), ],
                               'PTRATIO': [data.get('PTRATIO'), ],
                               'B': [data.get('B'), ],
                               'LSTAT': [data.get('LSTAT'), ],}, index=[0, ])
            predicted_price = model.predict(df)[0]
            predicted_price = np.round(predicted_price)
            prediction = {'predicted_price': predicted_price}
        except:
            prediction = {'It was not possible to get the prediction'}
        return jsonify(result=prediction)


if __name__ == '__main__':
    app.config["SECRET_KEY"] = "ITSASECRET"
    app.run(host='0.0.0.0', port=port, debug=True)
