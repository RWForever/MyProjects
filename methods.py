import torch
from sklearn import preprocessing


def create_dataset(data, target_features, input_features):
    data_x = data[input_features]
    data_y = data[target_features]
    data_x = torch.from_numpy(data_x.to_numpy()).float()
    data_x = data_x.reshape(data_x.shape[0], 1, data_x.shape[1])
    data_y = torch.squeeze(torch.from_numpy(data_y.to_numpy()).float())
    return data_x, data_y




