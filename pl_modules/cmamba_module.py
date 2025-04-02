import torch.nn as nn
from models.cmamba import CMamba
from .base_module import BaseModule
    

class CryptoMambaModule(BaseModule):

    def __init__(
        self,
        num_features=5,
        hidden_dims=[14, 3],  # Changed default to output 3 classes
        norm_layer=nn.LayerNorm,
        d_conv=4,
        layer_density=1,
        expand=2, 
        mlp_ratio=0, 
        drop=0.0, 
        num_classes=3,  # Changed default to 3 for classification
        d_states=16,
        use_checkpoint=False,
        lr=0.0002, 
        lr_step_size=50,
        lr_gamma=0.1,
        weight_decay=0.0, 
        logger_type=None,
        window_size=14,
        y_key='Close',
        optimizer='adam',
        mode='default',
        loss='cross_entropy',  # Changed default to cross_entropy
        task_type='classification',  # Added task_type parameter
        threshold=0.001,  # Added threshold parameter
        cls=True,  # Added cls parameter for CMamba
        **kwargs
    ): 
        super().__init__(lr=lr,
                         lr_step_size=lr_step_size,
                         lr_gamma=lr_gamma,
                         weight_decay=weight_decay,
                         logger_type=logger_type,
                         y_key=y_key,
                         optimizer=optimizer,
                         mode=mode,
                         window_size=window_size,
                         loss=loss,
                         task_type=task_type,  # Pass new parameter to BaseModule
                         num_classes=num_classes,  # Pass new parameter to BaseModule
                         threshold=threshold,  # Pass new parameter to BaseModule
                         )
        assert window_size == hidden_dims[0]

        self.model = CMamba(
            num_features=num_features,
            hidden_dims=hidden_dims,
            norm_layer=norm_layer,
            d_conv=d_conv,
            layer_density=layer_density,
            expand=expand, 
            mlp_ratio=mlp_ratio, 
            drop=drop, 
            num_classes=num_classes,
            d_states=d_states,
            use_checkpoint=use_checkpoint,
            cls=cls,  # Pass cls parameter to CMamba
            **kwargs
        )
        
    # Add method to create class labels if you want to access it directly from this class
    def create_class_labels(self, returns):
        """
        Convert returns into class labels:
        0: Down (negative return below threshold)
        1: Stationary (return between -threshold and threshold)
        2: Up (positive return above threshold)
        """
        return super().create_class_labels(returns)
    
    # Add a prediction method specifically for market direction
    def predict_market_direction(self, features):
        """
        Predict market direction (down, stationary, up) for given features
        
        Args:
            features: Input features tensor
            
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.model(features)
            probabilities = torch.softmax(logits, dim=1)
            _, predicted_classes = torch.max(probabilities, dim=1)
            
            return predicted_classes, probabilities