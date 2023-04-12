from flask import Flask,jsonify,render_template,request
from flask_cors import CORS
from prophet import Prophet
from pandas import read_csv
from pandas import to_datetime
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
app=Flask(__name__)
CORS(app)
@app.route('/')
def predict():
    path="datasets.csv"
    data = pd.read_csv(path)
    data = data[['ORDERDATE', 'SALES']]
    data.columns = ['ds', 'y']
    start_date = pd.to_datetime('2005-07-01')
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=365)
    future = future.loc[future['ds'] >= start_date]
    forecast = model.predict(future)
    model.plot(forecast)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Sales Prediction')
    plt.legend()
    plt.show()
    return 'hello'
    
if __name__ == '__main__':
    app.run(debug=True)