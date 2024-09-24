import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import methods
from models import *
from pylab import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import  mean_absolute_error

mpl.rcParams['font.sans-serif'] = ['SimHei']
file_path = './91-Site_1A-Trina_5W.csv'
data = pd.read_csv(file_path, header=0, low_memory=False, index_col=0)
data = data.rename(columns={
    u'1A Trina - Active Energy Delivered-Received (kWh)': 'AE_Power',
    u'1A Trina - Current Phase Average (A)': 'Current',  # current power
    u'1A Trina - Wind Speed (m/s)': 'Wind_speed',  # wind speed
    u'1A Trina - Active Power (kW)': 'Power',  # power
    u'1A Trina - Weather Relative Humidity (%)': 'Humidity',  # humidity
    u'1A Trina - Weather Temperature Celsius (\xb0C)': 'Temp',  # temperature
    u'1A Trina - Global Horizontal Radiation (W/m\xb2)': 'GHI',  # ghi
    u'1A Trina - Diffuse Horizontal Radiation (W/m\xb2)': 'DHI',  # dhi
    u'1A Trina - Wind Direction (Degrees)': 'Wind_dir',  # wind direction
    u'1A Trina - Weather Daily Rainfall (mm)': 'Rainfall'  # rainfall
})

feature = ['Current', 'Wind_speed', 'Power', 'Humidity', 'Temp', 'GHI', 'DHI', 'Wind_dir']

# set input feature
input_feature = ['Wind_speed', 'Humidity', 'Temp', 'GHI']
input_feature_num = 4

# set target feature == output neurons num
target_feature = ['Power']

# assign NAN to 0
data = data.fillna(0)
data[data < 0] = 0

# set the example num
data = data[23500:24364]

# normalization
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
data[input_feature] = x_scaler.fit_transform(data[input_feature].to_numpy())
data[target_feature] = y_scaler.fit_transform(data[target_feature].to_numpy())

# dataset allocation
test_x, test_y = methods.create_dataset(data, target_feature, input_feature)

# import the models
BP_Net = torch.load('model_bp.pth')
RNN = torch.load('model_rnn.pth')
BiLSTM = torch.load('model_bi_lstm.pth')


def prediction(model, series_x, series_y, name):
    model = model.eval()
    pred = model(series_x)
    pred[pred < 0] = 0
    length = len(series_y)
    for i in range(length):
        if series_y[i] == 0:
            pred[i] = 0
    pred = pred.view(-1).data.numpy()
    pred = y_scaler.inverse_transform(pred.reshape(-1, 1))
    series_y = y_scaler.inverse_transform(series_y.reshape(-1, 1))
    MSE = mean_squared_error(series_y, pred)
    RMSE = sqrt(MSE)
    R2 = r2_score(series_y, pred)
    MAE = mean_absolute_error(series_y, pred)

    print(name, ' :')
    print(' MSE: {:.3f}'.format(MSE))
    print(' RMSE: {:.3f}'.format(RMSE))
    print(' MAE: {:.3f}'.format(MAE))
    print(' R2: {:.3f}'.format(R2))
    return pred


pred_bp = prediction(BP_Net, test_x, test_y, 'BP_Net')
pred_rnn = prediction(RNN, test_x, test_y, 'RNN')
pred_bilstm = prediction(BiLSTM, test_x, test_y, 'BiLSTM')
test_y = y_scaler.inverse_transform(test_y.reshape(-1, 1))

print('Drawing...')
# draw
x = np.linspace(0, 72, 864)
plt.plot(x, pred_bp, 'aqua', label='BP Prediction')
plt.plot(x, pred_rnn, 'yellow', label='RNN Prediction')
plt.plot(x, pred_bilstm, 'green', label='BILSTM Prediction')
plt.plot(x, test_y, 'r', label='Actual Value')

plt.title('72h prediction compared')
plt.xlabel('time(hour)')
plt.ylabel('power(kW)')
plt.xlim(0, 73)
plt.legend(loc='upper right')
plt.show()
print('Done')
