import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class Dataset_NASDAQ_subset(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='subset_nasdaq_data.csv',
                 scale=True, timeenc=0, freq='d'):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 252  # Example: 1 year of daily data (252 trading days)
            self.label_len = 30  # Example: 1 month
            self.pred_len = 30  # Example: 1 month
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        num_train = int(0.7 * len(df_raw))  # 70% for training
        num_val = int(0.1 * len(df_raw))  # 10% for validation
        num_test = len(df_raw) - num_train - num_val  # 20% for testing
        
        border1s = [0, num_train - self.seq_len, num_train + num_val - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = df_raw.iloc[:num_train, 1:].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_raw.iloc[:, 1:].values)
        else:
            data = df_raw.iloc[:, 1:].values
        
        df_stamp = df_raw.iloc[border1:border2, 0:1]
        df_stamp['date'] = pd.to_datetime(df_stamp['Date'])
        df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
        df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
        df_stamp['year'] = df_stamp['date'].apply(lambda row: row.year)
        data_stamp = df_stamp.drop(['date'], axis=1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)