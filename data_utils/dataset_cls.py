import os
import torch
import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import io_tools
from datetime import datetime
from utils.io_tools import load_config_from_yaml

    
class CMambaDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data,
        split,
        window_size,
        transform,
        task_type='regression',  # Added task_type parameter
        y_key='Close',           # Added y_key parameter
        threshold=0.001,         # Added threshold for classification
    ):
        self.data = data
        self.transform = transform
        self.window_size = window_size
        self.task_type = task_type
        self.y_key = y_key
        self.threshold = threshold
            
        print('{} data points loaded as {} split.'.format(len(self), split))
        
        # If classification, compute returns for downstream class label creation
        if self.task_type == 'classification':
            # Pre-compute returns if we're using classification
            self._compute_returns()

    def __len__(self):
        return max(0, len(self.data) - self.window_size - 1)

    def __getitem__(self, i: int):
        sample = self.data.iloc[i: i + self.window_size + 1]
        transformed_sample = self.transform(sample)
        
        # For classification, add returns and class labels
        if self.task_type == 'classification':
            # Add price returns
            if 'returns' not in transformed_sample:
                returns = self.returns[i]
                transformed_sample['returns'] = returns
                
            # Add class labels using returns
            target_class = self.create_class_labels(transformed_sample['returns'])
            transformed_sample['target_class'] = target_class
            
        return transformed_sample
    
    def _compute_returns(self):
        """Pre-compute returns for the entire dataset for efficiency"""
        prices = self.data[self.y_key].values
        # Calculate returns for each window
        self.returns = []
        for i in range(len(self)):
            current_price = prices[i + self.window_size]
            previous_price = prices[i + self.window_size - 1]
            
            # Calculate percentage return
            if previous_price != 0:
                ret = (current_price - previous_price) / previous_price
            else:
                ret = 0.0
                
            self.returns.append(ret)
    
    def create_class_labels(self, returns):
        """
        Convert returns into class labels:
        0: Down (negative return below threshold)
        1: Stationary (return between -threshold and threshold)
        2: Up (positive return above threshold)
        """
        if isinstance(returns, torch.Tensor):
            # If returns is a tensor
            labels = torch.zeros_like(returns, dtype=torch.long)
            labels[returns < -self.threshold] = 0  # Down
            labels[(returns >= -self.threshold) & (returns <= self.threshold)] = 1  # Stationary
            labels[returns > self.threshold] = 2  # Up
        else:
            # If returns is a scalar or numpy value
            returns = float(returns)
            if returns < -self.threshold:
                return 0  # Down
            elif returns <= self.threshold:
                return 1  # Stationary
            else:
                return 2  # Up
                
        return labels


class ClassificationTransform:
    """
    Transform class for CryptoMamba classification tasks.
    Extends the regular transform functionality with classification-specific features.
    
    This transform:
    1. Handles standard data normalization
    2. Creates feature windows
    3. Calculates returns for classification
    """
    def __init__(
        self,
        factors=None,  # Normalization factors
        y_key='Close',
        feature_keys=None,
        threshold=0.001,
    ):
        self.factors = factors
        self.y_key = y_key
        self.feature_keys = feature_keys or ['Open', 'High', 'Low', 'Close', 'Volume']
        self.threshold = threshold
        
    def __call__(self, data):
        """
        Transform a window of data for the model
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dict with features and targets
        """
        # Get the last timestamp for reference
        timestamp = data.iloc[-1]['Timestamp']
        
        # Extract features (all rows except the last one)
        features_df = data.iloc[:-1]
        
        # Extract current values and previous values for calculating returns
        current_values = data.iloc[-1]
        previous_values = data.iloc[-2]
        
        # Calculate normalized features
        features = []
        for key in self.feature_keys:
            values = features_df[key].values
            
            # Normalize if factors are provided
            if self.factors is not None:
                factor = self.factors.get(key, {})
                min_val = factor.get('min', min(values))
                max_val = factor.get('max', max(values))
                if max_val > min_val:
                    values = (values - min_val) / (max_val - min_val)
                    
            features.append(values)
            
        features = np.stack(features, axis=1)
        features = torch.tensor(features, dtype=torch.float32)
        
        # Get target value (last row)
        y = current_values[self.y_key]
        y_old = previous_values[self.y_key]
        
        # Normalize target if factors are provided
        if self.factors is not None:
            factor = self.factors.get(self.y_key, {})
            min_val = factor.get('min', 0)
            max_val = factor.get('max', 1)
            if max_val > min_val:
                y = (y - min_val) / (max_val - min_val)
                y_old = (y_old - min_val) / (max_val - min_val)
        
        # Calculate returns for classification
        returns = (y - y_old) / y_old if y_old != 0 else 0.0
        
        # Create final sample dict
        sample = {
            'features': features,
            self.y_key: torch.tensor(y, dtype=torch.float32),
            f'{self.y_key}_old': torch.tensor(y_old, dtype=torch.float32),
            'returns': torch.tensor(returns, dtype=torch.float32),
            'timestamp': torch.tensor(timestamp, dtype=torch.float32),
        }
        
        return sample


class DataConverter:
    # [Original DataConverter code remains unchanged]
    def __init__(self, config) -> None:
        self.config = config
        self.root = config.get('root')
        self.jumps = config.get('jumps') * 60
        self.end_date = config.get('end_date')
        self.data_path = config.get('data_path')
        self.start_date = config.get('start_date')
        self.folder_name = f'{self.start_date}_{self.end_date}_{self.jumps // 60}'
        self.file_path = f'{self.root}/{self.folder_name}'
    
    def process_data(self):
        data, start, stop = self.load_data()
        new_df = {}
        for key in ['Timestamp', 'High', 'Low', 'Open', 'Close', 'Volume']:
            new_df[key] = []
        for i in tqdm(range(start, stop - 3600, self.jumps)):
            high, low, open, close, vol = self.merge_data(data, i, self.jumps)
            if high is None:
                continue
            new_df.get('Timestamp').append(i)
            new_df.get('High').append(high)
            new_df.get('Low').append(low)
            new_df.get('Open').append(open)
            new_df.get('Close').append(close)
            new_df.get('Volume').append(vol)

        df = pd.DataFrame(new_df)

        return df

    def get_data(self):
        tmp = '----'
        data_path = f'{self.file_path}/{tmp}.csv'
        yaml_path = f'{self.file_path}/config.pkl'
        
        if os.path.isfile(data_path.replace(tmp, 'train')):
            train = pd.read_csv(data_path.replace(tmp, 'train'), index_col=0)
            val = pd.read_csv(data_path.replace(tmp, 'val'), index_col=0)
            test = pd.read_csv(data_path.replace(tmp, 'test'), index_col=0)
            return train, val, test
        
        df = self.process_data()

        train, val, test = self.split(df)

        if not os.path.isdir(self.file_path):
            os.mkdir(self.file_path)

        train.to_csv(data_path.replace('----', 'train'))
        val.to_csv(data_path.replace('----', 'val'))
        test.to_csv(data_path.replace('----', 'test'))
        io_tools.save_yaml(self.config, yaml_path)
        return train, val, test
    

    def split(self, data):
        if self.config.get('train_ratio') is not None:
            total = len(data)
            n_train = int(self.train_ratio * total)
            n_test = int(self.test_ratio * total)
            total = list(range(total))
            random.shuffle(total)
        
            train = sorted(total[: n_train])
            val = sorted(total[n_train: -n_test])
            test = sorted(total[-n_test: ])
            train = data.iloc[train]
            val = data.iloc[val]
            test = data.iloc[test]
        else:
            tmp_dict = {}
            for key in ['train', 'val', 'test']:
                start, stop = self.config.get(f'{key}_interval')
                start = self.generate_timestamp(f'{start}: 00-00')
                stop = self.generate_timestamp(f'{stop}: 00-00')
                tmp = data[data['Timestamp'] >= start].reset_index(drop=True)
                tmp = tmp[tmp['Timestamp'] < stop].reset_index(drop=True)
                tmp_dict[key] = tmp
            train = tmp_dict.get('train')
            val = tmp_dict.get('val')
            test = tmp_dict.get('test')
        return train, val, test


    def load_data(self):
        df = pd.read_csv(self.data_path)
        if 'Timestamp' not in df.keys():
            dates = df.get('Date').to_list()
            df['Timestamp'] = [self.generate_timestamp(x, d_format="%Y-%m-%d") for x in dates]
        if self.start_date is None:
            self.start_date = self.convert_timestamp(min(list(df.get('Timestamp')))).strftime('%Y-%d-%m')
            # raise ValueError(self.start_date)
        if self.end_date is None:
            self.end_date = self.convert_timestamp(max(list(df.get('Timestamp'))) + 24 * 60 * 60).strftime('%Y-%d-%m')
        start = self.generate_timestamp(f'{self.start_date}: 00-00')
        stop = self.generate_timestamp(f'{self.end_date}: 00-00')
        df = df[df['Timestamp'] >= start].reset_index(drop=True)
        df = df[df['Timestamp'] < stop].reset_index(drop=True)
        final_day = self.generate_timestamp(f'{self.end_date}: 00-00')
        return df, start, final_day
    
    @staticmethod
    def merge_data(data, start, jump):
        tmp = data[data['Timestamp'] >= start].reset_index(drop=True)
        tmp = tmp[tmp['Timestamp'] < start + jump].reset_index(drop=True)
        if len(tmp) == 0:
            return None, None, None, None, None
        _, high, low, open, close, _ = DataConverter.get_row_values(tmp.iloc[0])
        count = 1
        vol = 0
        for row in tmp.iterrows():
            _, h, l, _, close, v = DataConverter.get_row_values(row[1])
            high = max(high, h)
            low = min(low, l)
            vol += v
            count += 1
        return high, low, open, close, vol
    
    @staticmethod
    def get_row_values(row):
        ts = int(row.get('Timestamp'))
        high = float(row.get('High'))
        low = float(row.get('Low'))
        open = float(row.get('Open'))
        close = float(row.get('Close'))
        vol = float(row.get('Volume'))
        return ts, high, low, open, close, vol

    @staticmethod
    def generate_timestamp(date, d_format="%Y-%d-%m: %H-%M"):
        """
            Year-day-month: hour-minute (all zero padded)
        """
        return int(time.mktime(datetime.strptime(date, d_format).timetuple()))

    @staticmethod
    def convert_timestamp(timestamp):
        return datetime.fromtimestamp(timestamp)
    
    # You might want to add a method for creating transforms based on task type
    def create_transform(self, task_type='regression', y_key='Close', threshold=0.001):
        """
        Create appropriate transform based on task type
        
        Args:
            task_type: 'regression' or 'classification'
            y_key: Target column name
            threshold: Threshold for stationary class in classification
            
        Returns:
            Transform function
        """
        if task_type == 'classification':
            return ClassificationTransform(
                factors=self.get_normalization_factors(),
                y_key=y_key,
                feature_keys=['Open', 'High', 'Low', 'Close', 'Volume'],
                threshold=threshold
            )
        else:
            # Return original transform for regression
            return self.original_transform
            
    def get_normalization_factors(self):
        """
        Calculate normalization factors for features
        
        Returns:
            Dict of normalization factors
        """
        # Implementation would depend on your data structure
        # This is a placeholder
        train, _, _ = self.get_data()
        factors = {}
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            factors[col] = {
                'min': float(train[col].min()),
                'max': float(train[col].max())
            }
            
        return factors