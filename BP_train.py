import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from models import NetBP
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


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

# feature_num == input neurons num
input_feature_num = 4
# set target feature
target_feature = ['Power']

# delete empty-power data
data = data.dropna(subset=['Power'])

# assign NAN and data less than 0 to 0
data = data.fillna(0)
data[data < 0] = 0

# set the example num
data = data[:8640]
# normalization
scaler = MinMaxScaler()
data[feature] = scaler.fit_transform(data[feature].to_numpy())

# Dataset adjustment, X to tensor, y to one-dimensional sequence
data_x = data[input_feature]
data_y = data[target_feature]
data_x = torch.from_numpy(data_x.to_numpy()).float()
train_x = data_x.reshape(data_x.shape[0], 1, data_x.shape[1])
train_y = torch.squeeze(torch.from_numpy(data_y.to_numpy()).float())

bp_net = NetBP(n_features=input_feature_num)

# lr:learning rate
# epochs:Number of training wheels
optimizer = torch.optim.SGD(bp_net.parameters(), lr=0.01)
loss_func = nn.MSELoss()
epochs = 2000
print(bp_net)
print('Start training...')

for e in range(epochs):
    # The forward propagation
    y_pred = bp_net(train_x)
    y_pred = torch.squeeze(y_pred)
    loss = loss_func(y_pred, train_y)
    # Back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 400 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(e, loss.item()))

plt.plot(y_pred.detach().numpy(), 'r', label='y_pred')
plt.plot(train_y.detach().numpy(), 'b', label='y_train')
plt.legend()
plt.show()

print('Done.')

print('Model saving...')
MODEL_PATH = 'model_bp.pth'
torch.save(bp_net, MODEL_PATH)
print('Model saved')
