"""
Data Preprocessing Utilities for Air Quality Forecasting

This module contains utility functions for data preprocessing,
feature engineering, and sequence creation for time series modeling.

Author: Reine Mizero
Date: January 2026
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AirQualityPreprocessor:
    """
    Comprehensive preprocessing pipeline for air quality data
    """
    
    def __init__(self, sequence_length=24, target_column='pm2.5'):
        """
        Initialize the preprocessor
        
        Parameters:
        - sequence_length: Number of time steps for sequence creation
        - target_column: Name of the target variable column
        """
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_names = None
        
    def create_temporal_features(self, df):
        """
        Create temporal features from datetime index
        """
        df_processed = df.copy()
        
        # Extract basic temporal features
        df_processed['year'] = df_processed.index.year
        df_processed['month'] = df_processed.index.month
        df_processed['day'] = df_processed.index.day
        df_processed['hour'] = df_processed.index.hour
        df_processed['day_of_week'] = df_processed.index.dayofweek
        df_processed['day_of_year'] = df_processed.index.dayofyear
        df_processed['week_of_year'] = df_processed.index.isocalendar().week
        df_processed['is_weekend'] = (df_processed.index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding
        df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
        df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
        df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['day'] / 31)
        df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['day'] / 31)
        df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour'] / 24)
        df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour'] / 24)
        
        return df_processed
    
    def create_lag_features(self, df, features, lag_periods=[1, 2, 3, 6, 12, 24]):
        """
        Create lag features for specified columns
        """
        df_lagged = df.copy()
        
        for feature in features:
            if feature in df_lagged.columns:
                for lag in lag_periods:
                    df_lagged[f'{feature}_lag_{lag}h'] = df_lagged[feature].shift(lag)
        
        return df_lagged
    
    def create_rolling_features(self, df, features, windows=[3, 6, 12, 24]):
        """
        Create rolling statistics features
        """
        df_rolling = df.copy()
        
        for feature in features:
            if feature in df_rolling.columns:
                for window in windows:
                    # Rolling statistics
                    df_rolling[f'{feature}_rolling_mean_{window}h'] = df_rolling[feature].rolling(window=window, min_periods=1).mean()
                    df_rolling[f'{feature}_rolling_std_{window}h'] = df_rolling[feature].rolling(window=window, min_periods=1).std()
                    df_rolling[f'{feature}_rolling_min_{window}h'] = df_rolling[feature].rolling(window=window, min_periods=1).min()
                    df_rolling[f'{feature}_rolling_max_{window}h'] = df_rolling[feature].rolling(window=window, min_periods=1).max()
        
        return df_rolling
    
    def create_weather_interactions(self, df):
        """
        Create weather interaction features
        """
        df_interactions = df.copy()
        
        # Temperature-Dewpoint difference
        if all(col in df_interactions.columns for col in ['TEMP', 'DEWP']):
            df_interactions['temp_dewp_diff'] = df_interactions['TEMP'] - df_interactions['DEWP']
            
            # Relative humidity approximation
            df_interactions['relative_humidity'] = 100 * (
                np.exp((17.625 * df_interactions['DEWP']) / (243.04 + df_interactions['DEWP'])) / 
                np.exp((17.625 * df_interactions['TEMP']) / (243.04 + df_interactions['TEMP']))
            )
        
        # Wind chill effect
        if all(col in df_interactions.columns for col in ['TEMP', 'Iws']):
            df_interactions['wind_chill_effect'] = df_interactions['TEMP'] - (df_interactions['Iws'] * 2)
        
        # Pressure-temperature interaction
        if all(col in df_interactions.columns for col in ['PRES', 'TEMP']):
            df_interactions['pressure_temp_interaction'] = df_interactions['PRES'] * df_interactions['TEMP']
        
        return df_interactions
    
    def handle_missing_values(self, df, strategy='advanced'):
        """
        Handle missing values with various strategies
        """
        df_filled = df.copy()
        
        if strategy == 'advanced':
            # Time series specific imputation
            for column in df_filled.columns:
                if df_filled[column].isnull().sum() > 0:
                    # Forward fill, backward fill, then interpolation
                    df_filled[column].fillna(method='ffill', inplace=True)
                    df_filled[column].fillna(method='bfill', inplace=True)
                    df_filled[column].interpolate(method='linear', inplace=True)
                    df_filled[column].fillna(df_filled[column].mean(), inplace=True)
        elif strategy == 'mean':
            df_filled.fillna(df_filled.mean(), inplace=True)
        elif strategy == 'median':
            df_filled.fillna(df_filled.median(), inplace=True)
        
        return df_filled
    
    def create_sequences(self, X, y=None, prediction_steps=1):
        """
        Create sequences for time series modeling
        """
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(self.sequence_length, len(X) - prediction_steps + 1):
            X_seq.append(X[i-self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i + prediction_steps - 1])
        
        X_seq = np.array(X_seq)
        if y_seq is not None:
            y_seq = np.array(y_seq)
        
        return X_seq, y_seq
    
    def fit_transform(self, train_df, exclude_features=None):
        """
        Fit the preprocessor and transform training data
        """
        if exclude_features is None:
            exclude_features = [self.target_column, 'No']
        
        # Feature engineering
        train_processed = self.create_temporal_features(train_df)
        
        # Lag features (excluding target to prevent leakage)
        lag_features = [col for col in train_processed.columns if col not in [self.target_column]]
        train_processed = self.create_lag_features(train_processed, lag_features)
        
        # Rolling features
        rolling_features = ['TEMP', 'DEWP', 'PRES', 'Iws', 'Is', 'Ir']
        train_processed = self.create_rolling_features(train_processed, rolling_features)
        
        # Weather interactions
        train_processed = self.create_weather_interactions(train_processed)
        
        # Handle missing values
        train_processed = self.handle_missing_values(train_processed)
        
        # Separate features and target
        feature_cols = [col for col in train_processed.columns if col not in exclude_features]
        X = train_processed[feature_cols].values
        y = train_processed[self.target_column].values
        
        # Fit and transform scalers
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Store feature names
        self.feature_names = feature_cols
        
        return X_scaled, y_scaled, feature_cols
    
    def transform(self, df, exclude_features=None):
        """
        Transform new data using fitted preprocessor
        """
        if exclude_features is None:
            exclude_features = [self.target_column, 'No']
        
        # Apply same transformations
        df_processed = self.create_temporal_features(df)
        
        # Lag features
        lag_features = [col for col in df_processed.columns if col not in [self.target_column]]
        df_processed = self.create_lag_features(df_processed, lag_features)
        
        # Rolling features
        rolling_features = ['TEMP', 'DEWP', 'PRES', 'Iws', 'Is', 'Ir']
        df_processed = self.create_rolling_features(df_processed, rolling_features)
        
        # Weather interactions
        df_processed = self.create_weather_interactions(df_processed)
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Use same features as training
        if self.feature_names is not None:
            available_features = [col for col in self.feature_names if col in df_processed.columns]
            X = df_processed[available_features].values
        else:
            feature_cols = [col for col in df_processed.columns if col not in exclude_features]
            X = df_processed[feature_cols].values
        
        # Transform using fitted scaler
        X_scaled = self.feature_scaler.transform(X)
        
        return X_scaled

def calculate_rmse_original_scale(y_true_scaled, y_pred_scaled, scaler):
    """
    Calculate RMSE on original scale
    """
    y_true_orig = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    return np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))

if __name__ == "__main__":
    print("Air Quality Data Preprocessing Utilities")
    print("This module provides comprehensive preprocessing tools for air quality forecasting.")
