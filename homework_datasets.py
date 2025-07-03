from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class CustomDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def normalize(self):
        scaler = MinMaxScaler()
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        return self
    
    def fill_none(self):
        self.data = self.data.fillna('NaN')
        return self
    
    def encode(self):
        self.fill_none()
        categorical_cols = self.data.select_dtypes(exclude=['number']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
        return self
    
    def __str__(self):
        return str(self.data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]