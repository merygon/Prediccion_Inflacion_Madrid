"""
Simple test script to verify feature engineering functionality.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append('src')

from feature_engineering import FeatureEngineer

def create_sample_data():
    """Create sample inflation data for testing."""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
    np.random.seed(42)
    
    # Create sample IPC data with realistic values
    base_ipc = 100
    ipc_values = []
    for i in range(len(dates)):
        # Add trend and seasonal components
        trend = i * 0.2
        seasonal = 2 * np.sin(2 * np.pi * i / 12)
        noise = np.random.normal(0, 0.5)
        ipc_values.append(base_ipc + trend + seasonal + noise)
    
    data = pd.DataFrame({
        'fecha': dates,
        'ipc_general': ipc_values,
        'ipc_alimentacion': [val + np.random.normal(0, 1) for val in ipc_values],
        'ipc_vivienda': [val + np.random.normal(0, 0.8) for val in ipc_values]
    })
    
    return data

def test_feature_engineering():
    """Test the FeatureEngineer class functionality."""
    print("Testing Feature Engineering Module...")
    
    # Create sample data
    data = create_sample_data()
    print(f"Created sample data with {len(data)} rows and {len(data.columns)} columns")
    
    # Initialize FeatureEngineer
    fe = FeatureEngineer()
    print("Initialized FeatureEngineer")
    
    # Test lag features
    print("\n1. Testing lag features...")
    lagged_data = fe.create_lag_features(data)
    lag_cols = [col for col in lagged_data.columns if '_lag_' in col]
    print(f"Created {len(lag_cols)} lag features: {lag_cols[:5]}...")
    
    # Test rolling features
    print("\n2. Testing rolling features...")
    rolling_data = fe.create_rolling_features(lagged_data)
    rolling_cols = [col for col in rolling_data.columns if any(x in col for x in ['_ma_', '_std_', '_min_', '_max_'])]
    print(f"Created {len(rolling_cols)} rolling features: {rolling_cols[:5]}...")
    
    # Test seasonal features
    print("\n3. Testing seasonal features...")
    seasonal_data = fe.create_seasonal_features(rolling_data)
    seasonal_cols = [col for col in seasonal_data.columns if any(x in col for x in ['month', 'quarter', 'sin', 'cos'])]
    print(f"Created {len(seasonal_cols)} seasonal features: {seasonal_cols[:5]}...")
    
    # Test economic indicators
    print("\n4. Testing economic indicators...")
    indicators_data = fe.create_economic_indicators(seasonal_data)
    indicator_cols = [col for col in indicators_data.columns if any(x in col for x in ['_trend_', '_roc_', '_volatility_'])]
    print(f"Created {len(indicator_cols)} economic indicator features: {indicator_cols[:5]}...")
    
    # Test feature selection
    print("\n5. Testing feature selection...")
    target_col = 'ipc_general'
    selected_features = fe.create_feature_selection_methods(indicators_data, target_col, method='correlation')
    print(f"Selected {len(selected_features)} features using correlation method")
    
    # Generate feature summary
    print("\n6. Generating feature summary...")
    summary = fe.get_feature_summary(indicators_data)
    print(f"Total features: {summary['total_features']}")
    print(f"Lag features: {summary['lag_features_count']}")
    print(f"Rolling features: {summary['rolling_features_count']}")
    print(f"Seasonal features: {summary['seasonal_features_count']}")
    print(f"Economic indicators: {summary['economic_indicators_count']}")
    
    print("\n✅ All feature engineering tests completed successfully!")
    return indicators_data

if __name__ == "__main__":
    test_data = test_feature_engineering()
    print(f"\nFinal dataset shape: {test_data.shape}")
    print(f"Sample columns: {list(test_data.columns)[:10]}...")