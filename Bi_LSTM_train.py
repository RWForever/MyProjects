
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from models import BiLSTMNet


file_path = './91-Site_1A-Trina_5W.csv'
data = pd.read_csv(file_path, header=0, low_memory=False, index_col=0)
data = data.rename(columns={
    u'1A Trina - Active Energy Delivered-Received (kWh)': 'AE_Power',
    u'1A Trina - Current Phase Average (A)': 'Current',
    u'1A Trina - Wind Speed (m/s)': 'Wind_speed',
    u'1A Trina - Active Power (kW)': 'Power',
    u'1A Trina - Weather Relative Humidity (%)': 'Humidity',
    u'1A Trina - Weather Temperature Celsius (\xb0C)': 'Temp',
    u'1A Trina - Global Horizontal Radiation (W/m\xb2)': 'GHI',
    u'1A Trina - Diffuse Horizontal Radiation (W/m\xb2)': 'DHI',
    u'1A Trina - Wind Direction (Degrees)': 'Wind_dir',
    u'1A Trina - Weather Daily Rainfall (mm)': 'Rainfall'
})

feature = ['Current', 'Wind_speed', 'Power', 'Humidity', 'Temp', 'GHI', 'DHI', 'Wind_dir']
# set input feature
input_feature = ['Wind_speed', 'Humidity', 'Temp', 'GHI']
input_feature_num = 4
# set target feature
target_feature = ['Power']
# dataset = data[~data['Power'].isin([0])].dropna(axis=0)

# delete empty-power data
data = data.dropna(subset=['Power'])
# NAN==0
data = data.fillna(0)
data[data < 0] = 0


# normalization
scaler = MinMaxScaler()
data[feature] = scaler.fit_transform(data[feature].to_numpy())
# print(data['Power'].describe())
# Set the input feature dataset
data_x = data[input_feature]
# Sets the target feature dataset
data_y = data[target_feature]

train_x = data_x[:8640]
train_y = data_y[:8640]


# transfer data to numpy
train_x = torch.from_numpy(train_x.to_numpy()).float()
train_y = torch.squeeze(torch.from_numpy(train_y.to_numpy()).float())
# Dataset adjustment, X to tensor
train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
# import model
bi_lstm = BiLSTMNet(input_size=input_feature_num)


optimizer = torch.optim.Adam(bi_lstm.parameters(), lr=0.01)
loss_func = nn.MSELoss()
epochs = 100
print(bi_lstm)
print('Start training...')

for e in range(epochs):

    # The forward propagation
    y_pred = bi_lstm(train_x)
    y_pred = torch.squeeze(y_pred)
    loss = loss_func(y_pred, train_y)
    #  The back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 20 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(e, loss.item()))

plt.plot(y_pred.detach().numpy(), 'r', label='y_pred')
plt.plot(train_y.detach().numpy(), 'b', label='y_train')
plt.legend()
plt.show()

print('Model saving...')

MODEL_PATH = 'model_bi_lstm.pth'

torch.save(bi_lstm, MODEL_PATH)

print('Model saved')
