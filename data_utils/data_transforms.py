import torch
import numpy as np


class DataTransform:
    def __init__(
        self, 
        is_train, 
        use_volume=False, 
        task_type='regression',  # Added task_type parameter
        threshold=0.0001,        # Added threshold for classification
        y_key='Close'           # Added target key parameter
    ):
        self.is_train = is_train
        self.keys = ['Timestamp', 'Open', 'High', 'Low', 'Close']
        if use_volume:
            self.keys.append('Volume')
        self.task_type = task_type
        self.threshold = threshold
        self.y_key = y_key
        print(f"Using features: {self.keys} with task type: {task_type}")

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
            return labels
        else:
            # If returns is a scalar or numpy value
            returns = float(returns)
            if returns < -self.threshold:
                return torch.tensor(0, dtype=torch.long)  # Down
            elif returns <= self.threshold:
                return torch.tensor(1, dtype=torch.long)  # Stationary
            else:
                return torch.tensor(2, dtype=torch.long)  # Up

    def __call__(self, window):
        data_list = []
        output = {}
        if 'Timestamp_orig' in window.keys():
            self.keys.append('Timestamp_orig')
        
        for key in self.keys:
            data = torch.tensor(window.get(key).tolist())
            if key == 'Volume':
                data /= 1e9
            output[key] = data[-1]
            output[f'{key}_old'] = data[-2]
            if key == 'Timestamp_orig':
                continue
            data_list.append(data[:-1].reshape(1, -1))
        
        features = torch.cat(data_list, 0)
        output['features'] = features
        
        # Add classification-specific outputs if needed
        if self.task_type == 'classification':
            # Calculate returns for the target variable
            current_price = output[self.y_key]
            previous_price = output[f'{self.y_key}_old']
            
            # Avoid division by zero
            if previous_price != 0:
                returns = (current_price - previous_price) / previous_price
            else:
                returns = torch.tensor(0.0)
                
            # Add returns to output
            output['returns'] = returns
            
            # Create class labels based on returns
            target_class = self.create_class_labels(returns)
            output['target_class'] = target_class
            
            # For debugging
            # print(f"Return: {returns:.4f}, Class: {target_class.item()}")
        
        return output

    def set_initial_seed(self, seed):
        try:
            self.rng.seed(seed)
        except AttributeError:
            # Initialize rng if not already done
            self.rng = np.random.RandomState(seed)