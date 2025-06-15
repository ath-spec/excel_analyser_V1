import numpy as np
import pandas as pd

def preprocess_data(file_path):
    data = pd.read_excel(file_path)
    data = data.replace(r'^\s*$', np.nan, regex=True)
    data = data.dropna(axis=1, how='all')
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:
            data[col] = data[col].fillna(data[col].median())
        elif data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
    return data
