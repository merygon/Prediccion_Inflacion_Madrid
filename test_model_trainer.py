"""
Test script for ModelTrainer class to verify implementation.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from model_trainer import ModelTrainer

def create_sample_data():
    """Create sample inflation data for testing."""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start='2010-01-01', end='2023-12-01', freq='M')
    
    # Create synthetic inflation data with trend and seasonality
    n_periods = len(dates)
    trend = np.linspace(1, 3, n_periods)  # Gradual increase from 1% to 3%
    seasonal = 0.5 * np.sin(2 * np.pi * np.arange(n_periods) / 12)  # Annual seasonality
    noise = np.random.normal(0, 0.3, n_periods)
    
    inflation_rate = trend + seasonal + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'fecha': dates,
        'ipc_general': 100 + np.cumsum(inflation_rate / 12),  # Convert to index
        'inflation_rate_annual': inflation_rate
    })
    
    return data

def test_model_trainer():
    """Test ModelTrainer functionality."""
    print("Testing ModelTrainer implementation...")
    
    # Create sample data
    data = create_sample_data()
    print(f"Created sample data with {len(data)} observations")
    
    # Initialize ModelTrainer
    trainer = ModelTrainer()
    print("ModelTrainer initialized successfully")
    
    # Test ARIMA model training
    try:
        print("\n1. Testing ARIMA model training...")
        arima_result = trainer.train_arima(data, target_column='inflation_rate_annual')
        print(f"ARIMA model trained successfully")
        print(f"ARIMA order: {arima_result['parameters']['order']}")
        print(f"ARIMA validation MAE: {arima_result['validation']['mae']:.4f}")
    except Exception as e:
        print(f"ARIMA training failed: {e}")
    
    # Test Random Forest model training
    try:
        print("\n2. Testing Random Forest model training...")
        # Create features for RF
        from feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        
        # Create lag features
        data_with_features = fe.create_lag_features(data, target_columns=['inflation_rate_annual'])
        data_with_features = fe.create_rolling_features(data_with_features, target_columns=['inflation_rate_annual'])
        
        # Prepare features and target
        feature_cols = [col for col in data_with_features.columns 
                       if col not in ['fecha', 'inflation_rate_annual'] and 
                       data_with_features[col].dtype in ['float64', 'int64']]
        
        X = data_with_features[feature_cols].fillna(method='ffill').fillna(0)
        y = data_with_features['inflation_rate_annual']
        
        rf_result = trainer.train_random_forest(X, y, hyperparameter_tuning=False)
        print(f"Random Forest model trained successfully")
        print(f"RF validation MAE: {rf_result['validation']['mae']:.4f}")
        print(f"Top 3 important features: {list(rf_result['feature_importance'].keys())[:3]}")
    except Exception as e:
        print(f"Random Forest training failed: {e}")
    
    # Test LSTM model training (simplified)
    try:
        print("\n3. Testing LSTM model training...")
        lstm_result = trainer.train_lstm(data, target_column='inflation_rate_annual', sequence_length=6)
        print(f"LSTM model trained successfully")
        print(f"LSTM validation MAE: {lstm_result['validation']['mae']:.4f}")
    except Exception as e:
        print(f"LSTM training failed: {e}")
    
    # Test model evaluation
    try:
        print("\n4. Testing model evaluation...")
        evaluation_results = trainer.evaluate_models()
        print(f"Evaluated {len(evaluation_results)} models")
        
        for model_name, results in evaluation_results.items():
            if results['status'] == 'success':
                metrics = results['metrics']
                print(f"{model_name}: MAE={metrics.get('MAE', 'N/A'):.4f}, "
                      f"RMSE={metrics.get('RMSE', 'N/A'):.4f}, "
                      f"MAPE={metrics.get('MAPE', 'N/A'):.2f}%")
    except Exception as e:
        print(f"Model evaluation failed: {e}")
    
    # Test model selection
    try:
        print("\n5. Testing model selection...")
        best_model = trainer.select_best_model()
        print(f"Best model: {best_model['model_name']} ({best_model['model_type']})")
        print(f"Best {best_model['selection_metric']}: {best_model['best_score']:.4f}")
    except Exception as e:
        print(f"Model selection failed: {e}")
    
    # Test model summary
    try:
        print("\n6. Testing model summary...")
        summary = trainer.get_model_summary()
        print(f"Total models trained: {summary['total_models']}")
        print(f"Model types: {summary['model_types']}")
    except Exception as e:
        print(f"Model summary failed: {e}")
    
    print("\nModelTrainer testing completed!")

if __name__ == "__main__":
    test_model_trainer()