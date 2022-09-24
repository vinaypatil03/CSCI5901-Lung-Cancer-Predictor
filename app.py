from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from numpy import unique
from numpy import argmax
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from absl import logging
import time
from actions import save_data, get_data_records
logging.set_verbosity(logging.INFO)
import tempfile
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

run = neptune.init(
    project=os.getenv('NEPTUNE_PROJECT'),
    api_token=os.getenv('NEPTUNE_API'),
)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/train-data', methods=['POST'])
def train_data():
    data = request.files['file']
    data.save(secure_filename(data.filename))
    bs = request.form.get('batch_size')
    ep = request.form.get('ephocs')
    ts = request.form.get('test_size')

    df = pd.read_csv(data.filename)

    df = df.dropna()
    df = df.drop_duplicates()

    le_gender = LabelEncoder()
    le_lung_cancer = LabelEncoder()
    df['GENDER'] = le_gender.fit_transform(df['GENDER'])
    df['LUNG_CANCER'] = le_lung_cancer.fit_transform(df['LUNG_CANCER'])

    DATA_ROOT = tempfile.mkdtemp(prefix='tfx-data')  # Create a temporary directory.
    df.to_csv(os.path.join(DATA_ROOT, 'data.csv'), index=False)
    _data_filepath = os.path.join(DATA_ROOT, "data.csv")

    X = df.drop(['LUNG_CANCER'], axis=1)
    y = df['LUNG_CANCER']

    n_class = len(unique(y))
    n_features = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(ts), random_state=42)

    model = Sequential()
    model.add(Dense(20, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(20, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(n_class, activation='softmax'))

    st = time.time()

    neptune_cbk = NeptuneCallback(run=run, base_namespace="training")

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.fit(X_train, y_train, epochs=int(ep), batch_size=int(bs), callbacks=[neptune_cbk])

    et = time.time()
    total_time = et - st

    yhat = model.predict(X_test)
    yhat = argmax(yhat, axis=-1).astype('int')
    acc = accuracy_score(y_test, yhat)

    save_data(bs, ep, acc, total_time)

    model.save('final_model.h5')
    data = get_data_records()

    return render_template('metrics.html', data=data)


@app.route('/metrics')
def metrics():
    data = get_data_records()
    return render_template('metrics.html', data=data)


@app.route('/get-data', methods=['POST'])
def get_data():
    lm = load_model('final_model.h5')
    data = request.form
    data = dict(data)
    inputData = [int(value) for key, value in data.items()]
    df = pd.DataFrame([inputData], columns=['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                                            'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING',
                                            'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'])
    x = lm.predict(df)
    x = argmax(x, axis=-1).astype('int')
    if x[0] == 1:
        return "Person Has Cancer"
    else:
        return "Person Does Not Have Cancer"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
