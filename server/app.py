from flask import Flask,request,send_file
from flask_cors import CORS
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import numpy as np
app=Flask(__name__)
CORS(app)
@app.route('/',methods=['POST','GET'])
def predict():
    path="datasets.csv"
    data = pd.read_csv(path)
    number = request.json['number']
    print('Received number:', number)
    periodicity = request.json['periodicity']
    print('Received periodicity:', periodicity)
    if periodicity=='week':
       periodicity='W'
    elif periodicity=='month':
        periodicity='M'
    elif periodicity=='year':
        periodicity='Y'
    else:
        periodicity='D'
    print('Received number:', number)
    data = data[['ORDERDATE', 'SALES']]
    data.columns = ['ds', 'y']
    data['ds'] = pd.to_datetime(data['ds'])
    data = data.sort_values('ds')
    start_date = pd.to_datetime('2005-05-31')
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=number ,freq=periodicity)
    future = future.loc[future['ds'] >= start_date]
    forecast = model.predict(future)
    actual = data[['ds', 'y']]
    predicted = forecast[['ds', 'yhat']]
    mse = np.mean((actual['y'] - predicted['yhat']) ** 2)  # Mean Squared Error
    mae = np.mean(np.abs(actual['y'] - predicted['yhat']))  # Mean Absolute Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print('Mean Squared Error (MSE):', mse)
    print('Mean Absolute Error (MAE):', mae)
    print('Root Mean Squared Error (RMSE):', rmse)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='ds', y='y', data=actual, label='Actual')
    sns.lineplot(x='ds', y='yhat', data=predicted, label='Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_title('Sales Prediction with Prophet\nMSE: {:.2f}, MAE: {:.2f}, RMSE: {:.2f}'.format(mse, mae, rmse))
    ax.legend()
    predicted.to_csv('prediction.csv', index=False)
    plt.savefig('plot.png')
   # plt.show()
    return 'helo'
@app.route('/getImage', methods=['GET'])
def get_image():
    image_file = open(r'C:\\Users\\Technopark Computers\\Desktop\\KAARPRO\\plot.png', 'rb')
    return send_file(image_file, mimetype='image/png')
if __name__ == '__main__':
    app.run(debug=True)