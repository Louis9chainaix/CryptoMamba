import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.cmamba import CMamba
from torchmetrics.regression import MeanAbsolutePercentageError as MAPE
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
    

class BaseModule(pl.LightningModule):

    def __init__(
        self,
        lr=0.0002, 
        lr_step_size=50,
        lr_gamma=0.1,
        weight_decay=0.0, 
        logger_type=None,
        window_size=14,
        y_key='Close',
        optimizer='adam',
        mode='default',
        loss='cross_entropy',  # Changed default to cross_entropy for classification
        task_type='classification',  # Added task type parameter
        num_classes=3,  # Added num_classes parameter
        threshold=0.001,  # Threshold for determining stationary price movement
    ):
        super().__init__()

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.logger_type = logger_type 
        self.y_key = y_key
        self.optimizer = optimizer
        self.batch_size = None   
        self.mode = mode
        self.window_size = window_size
        self.loss = loss
        self.task_type = task_type
        self.num_classes = num_classes
        self.threshold = threshold

        # Define losses for different tasks
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.mape = MAPE()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.normalization_coeffs = None

        # Classification metrics
        if self.task_type == 'classification':
            self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
            self.recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
            self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x, y_old=None):
        if self.task_type == 'regression':
            if self.mode == 'default':
                return self.model(x).reshape(-1)
            elif self.mode == 'diff':
                return self.model(x).reshape(-1) + y_old
        else:  # classification
            return self.model(x)  # Returns logits for num_classes
        
    def set_normalization_coeffs(self, factors):
        scale = factors.get(self.y_key).get('max') - factors.get(self.y_key).get('min')
        shift = factors.get(self.y_key).get('min')
        self.normalization_coeffs = (scale, shift)

    def denormalize(self, y, y_hat=None):
        if self.normalization_coeffs is not None:
            scale, shift = self.normalization_coeffs
            y = y * scale + shift
            if y_hat is not None:
                y_hat = y_hat * scale + shift
                return y, y_hat
        return y

    def create_class_labels(self, returns):
        """
        Convert returns into class labels:
        0: Down (negative return below threshold)
        1: Stationary (return between -threshold and threshold)
        2: Up (positive return above threshold)
        """
        labels = torch.zeros_like(returns, dtype=torch.long)
        labels[returns < -self.threshold] = 0  # Down
        labels[(returns >= -self.threshold) & (returns <= self.threshold)] = 1  # Stationary
        labels[returns > self.threshold] = 2  # Up
        return labels

    def training_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        
        if self.batch_size is None:
            self.batch_size = x.shape[0]
            
        if self.task_type == 'regression':
            # Original regression logic
            y_old = batch[f'{self.y_key}_old']
            y_hat = self.forward(x, y_old).reshape(-1)
            y, y_hat = self.denormalize(y, y_hat)
            mse = self.mse(y_hat, y)
            rmse = torch.sqrt(mse)
            mape = self.mape(y_hat, y)
            l1 = self.l1(y_hat, y)

            self.log("train/mse", mse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
            self.log("train/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("train/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("train/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)

            if self.loss == 'mse':
                return mse
            elif self.loss == 'rmse':
                return rmse
            elif self.loss == 'mae':
                return l1
            elif self.loss == 'mape':
                return mape
            
        else:  # Classification
            # Get returns
            if 'returns' in batch:
                returns = batch['returns']
            else:
                # If returns not provided, try to calculate from price
                y_old = batch.get(f'{self.y_key}_old')
                if y_old is not None:
                    # Denormalize if needed
                    y = self.denormalize(y)
                    y_old = self.denormalize(y_old)
                    returns = (y - y_old) / y_old  # Percentage returns
                else:
                    # If no reference price, use normalized values directly
                    returns = y
            
            # Convert to class labels (0=down, 1=stationary, 2=up)
            target_classes = self.create_class_labels(returns)
            
            # Forward pass to get logits
            logits = self.forward(x)
            
            # Calculate loss
            loss = self.cross_entropy(logits, target_classes)
            
            # Calculate metrics
            predictions = torch.argmax(logits, dim=1)
            acc = self.accuracy(predictions, target_classes)
            prec = self.precision(predictions, target_classes)
            rec = self.recall(predictions, target_classes)
            f1 = self.f1(predictions, target_classes)
            
            # Log metrics
            self.log("train/loss", loss.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("train/accuracy", acc, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("train/precision", prec, batch_size=self.batch_size, sync_dist=True, prog_bar=False)
            self.log("train/recall", rec, batch_size=self.batch_size, sync_dist=True, prog_bar=False)
            self.log("train/f1", f1, batch_size=self.batch_size, sync_dist=True, prog_bar=False)
            
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        
        if self.batch_size is None:
            self.batch_size = x.shape[0]
            
        if self.task_type == 'regression':
            # Original regression logic
            y_old = batch[f'{self.y_key}_old']
            y_hat = self.forward(x, y_old).reshape(-1)
            y, y_hat = self.denormalize(y, y_hat)
            mse = self.mse(y_hat, y)
            rmse = torch.sqrt(mse)
            mape = self.mape(y_hat, y)
            l1 = self.l1(y_hat, y)

            self.log("val/mse", mse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=False)
            self.log("val/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("val/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("val/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
            return {
                "val_loss": mse,
            }
            
        else:  # Classification
            # Get returns
            if 'returns' in batch:
                returns = batch['returns']
            else:
                # If returns not provided, try to calculate from price
                y_old = batch.get(f'{self.y_key}_old')
                if y_old is not None:
                    # Denormalize if needed
                    y = self.denormalize(y)
                    y_old = self.denormalize(y_old)
                    returns = (y - y_old) / y_old  # Percentage returns
                else:
                    # If no reference price, use normalized values directly
                    returns = y
            
            # Convert to class labels
            target_classes = self.create_class_labels(returns)
            
            # Forward pass
            logits = self.forward(x)
            
            # Calculate loss
            loss = self.cross_entropy(logits, target_classes)
            
            # Calculate metrics
            predictions = torch.argmax(logits, dim=1)
            acc = self.accuracy(predictions, target_classes)
            prec = self.precision(predictions, target_classes)
            rec = self.recall(predictions, target_classes)
            f1 = self.f1(predictions, target_classes)
            
            # Log metrics
            self.log("val/loss", loss.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("val/accuracy", acc, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("val/precision", prec, batch_size=self.batch_size, sync_dist=True, prog_bar=False)
            self.log("val/recall", rec, batch_size=self.batch_size, sync_dist=True, prog_bar=False)
            self.log("val/f1", f1, batch_size=self.batch_size, sync_dist=True, prog_bar=False)
            
            # Calculate class distribution for analysis
            class_counts = torch.bincount(target_classes, minlength=self.num_classes)
            class_distribution = class_counts.float() / class_counts.sum()
            self.log("val/class_0_ratio", class_distribution[0], batch_size=self.batch_size, sync_dist=True)
            self.log("val/class_1_ratio", class_distribution[1], batch_size=self.batch_size, sync_dist=True)
            self.log("val/class_2_ratio", class_distribution[2], batch_size=self.batch_size, sync_dist=True)
            
            return {
                "val_loss": loss,
                "val_acc": acc
            }
    
    def test_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        
        if self.batch_size is None:
            self.batch_size = x.shape[0]
            
        if self.task_type == 'regression':
            # Original regression logic
            y_old = batch[f'{self.y_key}_old']
            y_hat = self.forward(x, y_old).reshape(-1)
            y, y_hat = self.denormalize(y, y_hat)
            mse = self.mse(y_hat, y)
            rmse = torch.sqrt(mse)
            mape = self.mape(y_hat, y)
            l1 = self.l1(y_hat, y)

            self.log("test/mse", mse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=False)
            self.log("test/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("test/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("test/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
            return {
                "test_loss": mse,
            }
            
        else:  # Classification
            # Get returns
            if 'returns' in batch:
                returns = batch['returns']
            else:
                # If returns not provided, try to calculate from price
                y_old = batch.get(f'{self.y_key}_old')
                if y_old is not None:
                    # Denormalize if needed
                    y = self.denormalize(y)
                    y_old = self.denormalize(y_old)
                    returns = (y - y_old) / y_old  # Percentage returns
                else:
                    # If no reference price, use normalized values directly
                    returns = y
            
            # Convert to class labels
            target_classes = self.create_class_labels(returns)
            
            # Forward pass
            logits = self.forward(x)
            
            # Calculate metrics
            predictions = torch.argmax(logits, dim=1)
            loss = self.cross_entropy(logits, target_classes)
            acc = self.accuracy(predictions, target_classes)
            prec = self.precision(predictions, target_classes)
            rec = self.recall(predictions, target_classes)
            f1 = self.f1(predictions, target_classes)
            
            # Log metrics
            self.log("test/loss", loss.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("test/accuracy", acc, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("test/precision", prec, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("test/recall", rec, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            self.log("test/f1", f1, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            
            # Create confusion matrix for detailed analysis
            confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
            for t, p in zip(target_classes.cpu(), predictions.cpu()):
                confusion[t, p] += 1
                
            self.log("test/down_accuracy", confusion[0, 0].float() / confusion[0].sum().float() if confusion[0].sum() > 0 else torch.tensor(0.0), 
                    batch_size=self.batch_size, sync_dist=True)
            self.log("test/stationary_accuracy", confusion[1, 1].float() / confusion[1].sum().float() if confusion[1].sum() > 0 else torch.tensor(0.0), 
                    batch_size=self.batch_size, sync_dist=True)
            self.log("test/up_accuracy", confusion[2, 2].float() / confusion[2].sum().float() if confusion[2].sum() > 0 else torch.tensor(0.0), 
                    batch_size=self.batch_size, sync_dist=True)
            
            return {
                "test_loss": loss,
                "test_acc": acc,
                "test_f1": f1
            }
    
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optim = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == 'sgd':
            optim = torch.optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Unimplemented optimizer {self.optimizer}')
        scheduler = torch.optim.lr_scheduler.StepLR(optim, 
                                                    self.lr_step_size, 
                                                    self.lr_gamma
                                                    )
        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step()