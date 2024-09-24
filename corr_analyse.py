import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = './91-Site_1A-Trina_5W.csv'
data = pd.read_csv(file_path, header=0, low_memory=False)
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
data.index = pd.to_datetime(data.Timestamp)
# delete data which power is empty
data = data.dropna(subset=['Power'])
print(data.index)
df = data.corr()['Power']
print(df)

sns.heatmap(data.corr(), linewidths=0.1, vmax=1.0, linecolor='white', annot=True)
plt.show()

