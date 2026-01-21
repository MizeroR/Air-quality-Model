"""
Model Architectures for Air Quality Forecasting

This module contains various RNN/LSTM architectures optimized 
for time series forecasting of PM2.5 concentrations.

Author: Reine Mizero
Date: January 2026
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, GRU, SimpleRNN, 
    Bidirectional, Conv1D, MaxPooling1D, 
    GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
import numpy as np

def rmse_metric(y_true, y_pred):
    """Custom RMSE metric for Keras"""
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def rmse_loss(y_true, y_pred):
    """Custom RMSE loss function"""
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

class AirQualityModels:
    """
    Collection of model architectures for air quality forecasting
    """
    
    @staticmethod
    def create_simple_lstm(input_shape, units=32, dropout=0.0):
        """Simple single-layer LSTM model"""
        model = Sequential([
            LSTM(units, activation='relu', input_shape=input_shape),
            Dropout(dropout) if dropout > 0 else tf.keras.layers.Lambda(lambda x: x),
            Dense(1, activation='linear')
        ])
        return model
    
    @staticmethod
    def create_deep_lstm(input_shape, units=[64, 32], dropout=0.2):
        """Multi-layer LSTM model"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units[0], activation='relu', return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for i, unit in enumerate(units[1:]):
            return_seq = i < len(units[1:]) - 1
            model.add(LSTM(unit, activation='relu', return_sequences=return_seq))
            model.add(Dropout(dropout))
        
        model.add(Dense(1, activation='linear'))
        return model
    
    @staticmethod
    def create_bidirectional_lstm(input_shape, units=64, dropout=0.2):
        """Bidirectional LSTM model"""
        model = Sequential([
            Bidirectional(LSTM(units, activation='relu'), input_shape=input_shape),
            Dropout(dropout),
            Dense(32, activation='relu'),
            Dropout(dropout),
            Dense(1, activation='linear')
        ])
        return model
    
    @staticmethod
    def create_gru_model(input_shape, units=[64, 32], dropout=0.2):
        """GRU-based model"""
        model = Sequential()
        
        model.add(GRU(units[0], activation='relu', return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        
        for i, unit in enumerate(units[1:]):
            return_seq = i < len(units[1:]) - 1
            model.add(GRU(unit, activation='relu', return_sequences=return_seq))
            model.add(Dropout(dropout))
        
        model.add(Dense(1, activation='linear'))
        return model
    
    @staticmethod
    def create_cnn_lstm(input_shape, filters=64, kernel_size=3, lstm_units=50, dropout=0.2):
        """CNN-LSTM hybrid model"""
        model = Sequential([
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(dropout),
            LSTM(lstm_units, activation='relu'),
            Dropout(dropout),
            Dense(25, activation='relu'),
            Dense(1, activation='linear')
        ])
        return model
    
    @staticmethod
    def create_attention_lstm(input_shape, units=64, dropout=0.2):
        """LSTM with simplified attention mechanism"""
        model = Sequential([
            LSTM(units, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(units//2, activation='relu', return_sequences=True),
            Dropout(dropout),
            GlobalAveragePooling1D(),  # Simplified attention
            Dense(32, activation='relu'),
            Dropout(dropout),
            Dense(1, activation='linear')
        ])
        return model
    
    @staticmethod
    def create_regularized_lstm(input_shape, units=64, dropout=0.3, l2_reg=0.01):
        """LSTM with L2 regularization"""
        model = Sequential([
            LSTM(units, activation='relu', input_shape=input_shape, 
                 kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg)),
            Dropout(dropout),
            Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
            Dropout(dropout),
            Dense(1, activation='linear')
        ])
        return model
    
    @staticmethod
    def create_ensemble_base_models(input_shape):
        """Create multiple models for ensemble"""
        models = {
            'lstm_deep': AirQualityModels.create_deep_lstm(input_shape, units=[64, 32]),
            'lstm_bidirect': AirQualityModels.create_bidirectional_lstm(input_shape, units=64),
            'gru_model': AirQualityModels.create_gru_model(input_shape, units=[64, 32]),
            'cnn_lstm': AirQualityModels.create_cnn_lstm(input_shape, filters=64),
        }
        return models
    
    @staticmethod
    def get_model_configs():
        """Get all available model configurations"""
        return {
            'simple_lstm': {
                'function': AirQualityModels.create_simple_lstm,
                'params': {'units': 32, 'dropout': 0.0},
                'description': 'Single-layer LSTM with minimal complexity'
            },
            'deep_lstm': {
                'function': AirQualityModels.create_deep_lstm,
                'params': {'units': [64, 32], 'dropout': 0.2},
                'description': 'Multi-layer LSTM with dropout regularization'
            },
            'bidirectional_lstm': {
                'function': AirQualityModels.create_bidirectional_lstm,
                'params': {'units': 64, 'dropout': 0.2},
                'description': 'Bidirectional LSTM for improved context capture'
            },
            'gru_model': {
                'function': AirQualityModels.create_gru_model,
                'params': {'units': [64, 32], 'dropout': 0.2},
                'description': 'GRU-based alternative to LSTM'
            },
            'cnn_lstm': {
                'function': AirQualityModels.create_cnn_lstm,
                'params': {'filters': 64, 'kernel_size': 3, 'lstm_units': 50, 'dropout': 0.2},
                'description': 'Hybrid CNN-LSTM for feature extraction and temporal modeling'
            },
            'attention_lstm': {
                'function': AirQualityModels.create_attention_lstm,
                'params': {'units': 64, 'dropout': 0.2},
                'description': 'LSTM with simplified attention mechanism'
            },
            'regularized_lstm': {
                'function': AirQualityModels.create_regularized_lstm,
                'params': {'units': 64, 'dropout': 0.3, 'l2_reg': 0.01},
                'description': 'LSTM with L2 regularization for better generalization'
            }
        }

class ModelEvaluator:
    """
    Utility class for model evaluation and comparison
    """
    
    def __init__(self, target_scaler):
        self.target_scaler = target_scaler
        self.results = []
    
    def evaluate_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """
        Evaluate a single model and store results
        """
        # Make predictions
        train_pred = model.predict(X_train, verbose=0).flatten()
        val_pred = model.predict(X_val, verbose=0).flatten()
        
        # Calculate RMSE on original scale
        train_rmse = self.calculate_rmse_original(y_train, train_pred)
        val_rmse = self.calculate_rmse_original(y_val, val_pred)
        
        # Store results
        result = {
            'model_name': model_name,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'overfitting_ratio': val_rmse / train_rmse,
            'parameters': model.count_params()
        }
        
        self.results.append(result)
        return result
    
    def calculate_rmse_original(self, y_true_scaled, y_pred_scaled):
        """Calculate RMSE on original scale"""
        y_true_orig = self.target_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        y_pred_orig = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        return np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))
    
    def get_best_model(self):
        """Get the best performing model based on validation RMSE"""
        if not self.results:
            return None
        return min(self.results, key=lambda x: x['val_rmse'])
    
    def compare_models(self):
        """Compare all evaluated models"""
        if not self.results:
            return None
        
        import pandas as pd
        df = pd.DataFrame(self.results)
        return df.sort_values('val_rmse')

if __name__ == "__main__":
    print("Air Quality Model Architectures")
    print("This module provides various RNN/LSTM architectures for air quality forecasting.")
    
    # Example usage
    input_shape = (24, 50)  # 24 timesteps, 50 features
    models = AirQualityModels()
    
    print(f"\nAvailable model architectures: {len(models.get_model_configs())}")
    for name, config in models.get_model_configs().items():
        print(f"- {name}: {config['description']}")
