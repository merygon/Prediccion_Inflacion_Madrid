"""
Unit tests for FeatureEngineer class.
Tests feature creation, transformations, and calculation accuracy.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
import sys

# Add src directory to path
sys.path.append('src')

from feature_engineering import FeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary config
        self.temp_config = {
            'feature_engineering': {
                'lag_periods': [1, 3, 6, 12],
                'rolling_windows': [3, 6, 12],
                'seasonal_features': True,
                'economic_indicators': True
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(self.temp_config, f)
            self.config_file = f.name
        
        # Create sample data for testing
        self.sample_dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        np.random.seed(42)
        
        # Sample IPC data with realistic patterns
        base_ipc = 100
        self.sample_ipc_values = []
        for i in range(len(self.sample_dates)):
            trend = i * 0.2
            seasonal = 2 * np.sin(2 * np.pi * i / 12)
            noise = np.random.normal(0, 0.5)
            self.sample_ipc_values.append(base_ipc + trend + seasonal + noise)
        
        self.sample_data = pd.DataFrame({
            'fecha': self.sample_dates,
            'ipc_general': self.sample_ipc_values,
            'ipc_alimentacion': [val + np.random.normal(0, 1) for val in self.sample_ipc_values],
            'ipc_vivienda': [val + np.random.normal(0, 0.8) for val in self.sample_ipc_values],
            'ipc_general_annual_rate': np.random.normal(2.5, 1.5, len(self.sample_dates))
        })
        
        self.feature_engineer = FeatureEngineer(self.config_file)
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'config_file') and os.path.exists(self.config_file):
            os.unlink(self.config_file)
    
    def test_create_lag_features_default_columns(self):
        """Test creation of lag features with default target columns."""
        result = self.feature_engineer.create_lag_features(self.sample_data)
        
        # Check that lag features were created
        lag_columns = [col for col in result.columns if '_lag_' in col]
        self.assertGreater(len(lag_columns), 0)
        
        # Check specific lag features for IPC columns
        expected_lags = [1, 3, 6, 12]
        for lag in expected_lags:
            self.assertIn(f'ipc_general_lag_{lag}', result.columns)
            self.assertIn(f'ipc_alimentacion_lag_{lag}', result.columns)
            self.assertIn(f'ipc_vivienda_lag_{lag}', result.columns)
        
        # Check that original data is preserved
        self.assertEqual(len(result), len(self.sample_data))
        for col in self.sample_data.columns:
            self.assertIn(col, result.columns)
    
    def test_create_lag_features_specific_columns(self):
        """Test creation of lag features for specific target columns."""
        target_columns = ['ipc_general']
        result = self.feature_engineer.create_lag_features(
            self.sample_data, 
            target_columns=target_columns
        )
        
        # Check that only specified columns have lag features
        lag_columns = [col for col in result.columns if '_lag_' in col]
        for col in lag_columns:
            self.assertTrue(col.startswith('ipc_general_lag_'))
        
        # Should not have lag features for other columns
        self.assertNotIn('ipc_alimentacion_lag_1', result.columns)
        self.assertNotIn('ipc_vivienda_lag_1', result.columns)
    
    def test_create_lag_features_custom_lags(self):
        """Test creation of lag features with custom lag periods."""
        custom_lags = [2, 4, 8]
        result = self.feature_engineer.create_lag_features(
            self.sample_data,
            lags=custom_lags
        )
        
        # Check that custom lag features were created
        for lag in custom_lags:
            self.assertIn(f'ipc_general_lag_{lag}', result.columns)
        
        # Should not have default lag features
        self.assertNotIn('ipc_general_lag_1', result.columns)
        self.assertNotIn('ipc_general_lag_3', result.columns)
    
    def test_create_lag_features_accuracy(self):
        """Test accuracy of lag feature calculations."""
        # Use simple test data for easy verification
        simple_data = pd.DataFrame({
            'fecha': pd.date_range('2020-01-01', periods=10, freq='M'),
            'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        result = self.feature_engineer.create_lag_features(
            simple_data,
            target_columns=['value'],
            lags=[1, 2]
        )
        
        # Check lag_1 values (should be previous value)
        self.assertTrue(pd.isna(result['value_lag_1'].iloc[0]))  # First value should be NaN
        self.assertEqual(result['value_lag_1'].iloc[1], 10)      # Second value should be 10
        self.assertEqual(result['value_lag_1'].iloc[2], 20)      # Third value should be 20
        
        # Check lag_2 values (should be 2 periods back)
        self.assertTrue(pd.isna(result['value_lag_2'].iloc[0]))  # First value should be NaN
        self.assertTrue(pd.isna(result['value_lag_2'].iloc[1]))  # Second value should be NaN
        self.assertEqual(result['value_lag_2'].iloc[2], 10)      # Third value should be 10
    
    def test_create_rolling_features_default(self):
        """Test creation of rolling features with default settings."""
        result = self.feature_engineer.create_rolling_features(self.sample_data)
        
        # Check that rolling features were created
        rolling_columns = [col for col in result.columns if any(x in col for x in ['_ma_', '_std_', '_min_', '_max_'])]
        self.assertGreater(len(rolling_columns), 0)
        
        # Check specific rolling features
        expected_windows = [3, 6, 12]
        for window in expected_windows:
            self.assertIn(f'ipc_general_ma_{window}', result.columns)
            self.assertIn(f'ipc_general_std_{window}', result.columns)
            self.assertIn(f'ipc_general_min_{window}', result.columns)
            self.assertIn(f'ipc_general_max_{window}', result.columns)
    
    def test_create_rolling_features_accuracy(self):
        """Test accuracy of rolling feature calculations."""
        # Simple test data
        simple_data = pd.DataFrame({
            'fecha': pd.date_range('2020-01-01', periods=10, freq='M'),
            'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        result = self.feature_engineer.create_rolling_features(
            simple_data,
            target_columns=['value'],
            windows=[3]
        )
        
        # Check 3-period moving average
        # For index 2 (third value): (10+20+30)/3 = 20
        self.assertAlmostEqual(result['value_ma_3'].iloc[2], 20.0, places=1)
        
        # For index 3 (fourth value): (20+30+40)/3 = 30
        self.assertAlmostEqual(result['value_ma_3'].iloc[3], 30.0, places=1)
        
        # Check that first two values are NaN (not enough data for 3-period window)
        self.assertTrue(pd.isna(result['value_ma_3'].iloc[0]))
        self.assertTrue(pd.isna(result['value_ma_3'].iloc[1]))
    
    def test_create_seasonal_features(self):
        """Test creation of seasonal features."""
        result = self.feature_engineer.create_seasonal_features(self.sample_data)
        
        # Check that seasonal features were created
        seasonal_columns = [col for col in result.columns if any(x in col for x in ['month', 'quarter', 'sin', 'cos'])]
        self.assertGreater(len(seasonal_columns), 0)
        
        # Check specific seasonal features
        expected_features = ['month', 'quarter', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']
        for feature in expected_features:
            self.assertIn(feature, result.columns)
        
        # Check value ranges
        self.assertTrue(all(result['month'].between(1, 12)))
        self.assertTrue(all(result['quarter'].between(1, 4)))
        self.assertTrue(all(result['month_sin'].between(-1, 1)))
        self.assertTrue(all(result['month_cos'].between(-1, 1)))
    
    def test_create_seasonal_features_accuracy(self):
        """Test accuracy of seasonal feature calculations."""
        # Test data with known dates
        test_data = pd.DataFrame({
            'fecha': pd.to_datetime(['2020-01-01', '2020-04-01', '2020-07-01', '2020-10-01']),
            'value': [100, 101, 102, 103]
        })
        
        result = self.feature_engineer.create_seasonal_features(test_data)
        
        # Check month values
        expected_months = [1, 4, 7, 10]
        for i, expected_month in enumerate(expected_months):
            self.assertEqual(result['month'].iloc[i], expected_month)
        
        # Check quarter values
        expected_quarters = [1, 2, 3, 4]
        for i, expected_quarter in enumerate(expected_quarters):
            self.assertEqual(result['quarter'].iloc[i], expected_quarter)
    
    def test_create_economic_indicators(self):
        """Test creation of economic indicators."""
        result = self.feature_engineer.create_economic_indicators(self.sample_data)
        
        # Check that economic indicators were created
        indicator_columns = [col for col in result.columns if any(x in col for x in ['_trend_', '_roc_', '_volatility_'])]
        self.assertGreater(len(indicator_columns), 0)
        
        # Check specific indicators
        expected_indicators = [
            'ipc_general_trend_3', 'ipc_general_trend_6', 'ipc_general_trend_12',
            'ipc_general_roc_3', 'ipc_general_roc_6', 'ipc_general_roc_12',
            'ipc_general_volatility_3', 'ipc_general_volatility_6', 'ipc_general_volatility_12'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)
    
    def test_create_economic_indicators_accuracy(self):
        """Test accuracy of economic indicator calculations."""
        # Simple test data with clear trend
        simple_data = pd.DataFrame({
            'fecha': pd.date_range('2020-01-01', periods=10, freq='M'),
            'value': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]  # Clear upward trend
        })
        
        result = self.feature_engineer.create_economic_indicators(
            simple_data,
            target_columns=['value']
        )
        
        # Check trend calculation (should be positive for upward trend)
        trend_3 = result['value_trend_3'].dropna()
        self.assertTrue(all(trend_3 > 0))  # All trend values should be positive
        
        # Check rate of change (should be positive)
        roc_3 = result['value_roc_3'].dropna()
        self.assertTrue(all(roc_3 > 0))  # All ROC values should be positive
        
        # Check volatility (should be low for steady trend)
        volatility_3 = result['value_volatility_3'].dropna()
        self.assertTrue(all(volatility_3 < 5))  # Low volatility for steady data
    
    def test_create_feature_selection_methods(self):
        """Test feature selection methods."""
        # Create data with features
        data_with_features = self.feature_engineer.create_lag_features(self.sample_data)
        data_with_features = self.feature_engineer.create_rolling_features(data_with_features)
        
        # Test correlation-based selection
        selected_features = self.feature_engineer.create_feature_selection_methods(
            data_with_features,
            target_column='ipc_general_annual_rate',
            method='correlation',
            top_k=10
        )
        
        self.assertIsInstance(selected_features, list)
        self.assertLessEqual(len(selected_features), 10)
        
        # Test variance-based selection
        selected_features_var = self.feature_engineer.create_feature_selection_methods(
            data_with_features,
            target_column='ipc_general_annual_rate',
            method='variance',
            top_k=15
        )
        
        self.assertIsInstance(selected_features_var, list)
        self.assertLessEqual(len(selected_features_var), 15)
    
    def test_get_feature_summary(self):
        """Test feature summary generation."""
        # Create comprehensive feature set
        data_with_features = self.feature_engineer.create_lag_features(self.sample_data)
        data_with_features = self.feature_engineer.create_rolling_features(data_with_features)
        data_with_features = self.feature_engineer.create_seasonal_features(data_with_features)
        data_with_features = self.feature_engineer.create_economic_indicators(data_with_features)
        
        summary = self.feature_engineer.get_feature_summary(data_with_features)
        
        # Check summary structure
        required_keys = [
            'total_features', 'original_features', 'lag_features_count',
            'rolling_features_count', 'seasonal_features_count', 'economic_indicators_count'
        ]
        
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Check that counts are reasonable
        self.assertGreater(summary['total_features'], summary['original_features'])
        self.assertGreater(summary['lag_features_count'], 0)
        self.assertGreater(summary['rolling_features_count'], 0)
        self.assertGreater(summary['seasonal_features_count'], 0)
        self.assertGreater(summary['economic_indicators_count'], 0)
    
    def test_edge_case_insufficient_data_for_lags(self):
        """Test handling of insufficient data for lag features."""
        # Very small dataset
        small_data = pd.DataFrame({
            'fecha': pd.date_range('2020-01-01', periods=2, freq='M'),
            'value': [100, 101]
        })
        
        result = self.feature_engineer.create_lag_features(
            small_data,
            target_columns=['value'],
            lags=[1, 3, 6]
        )
        
        # Should still create lag columns, but with NaN values
        self.assertIn('value_lag_1', result.columns)
        self.assertIn('value_lag_3', result.columns)
        self.assertIn('value_lag_6', result.columns)
        
        # Most lag values should be NaN due to insufficient data
        self.assertTrue(pd.isna(result['value_lag_3'].iloc[1]))
        self.assertTrue(pd.isna(result['value_lag_6'].iloc[1]))
    
    def test_edge_case_missing_date_column(self):
        """Test handling of data without date column."""
        data_no_date = self.sample_data.drop('fecha', axis=1)
        
        # Should still work for non-seasonal features
        result = self.feature_engineer.create_lag_features(data_no_date)
        self.assertGreater(len([col for col in result.columns if '_lag_' in col]), 0)
        
        # Seasonal features should handle missing date column gracefully
        result_seasonal = self.feature_engineer.create_seasonal_features(data_no_date)
        # Should return original data if no date column found
        self.assertEqual(len(result_seasonal.columns), len(data_no_date.columns))
    
    def test_edge_case_all_nan_column(self):
        """Test handling of column with all NaN values."""
        data_with_nan = self.sample_data.copy()
        data_with_nan['all_nan_column'] = np.nan
        
        result = self.feature_engineer.create_lag_features(data_with_nan)
        
        # Should create lag features even for NaN column
        self.assertIn('all_nan_column_lag_1', result.columns)
        
        # All lag values should also be NaN
        self.assertTrue(result['all_nan_column_lag_1'].isna().all())
    
    def test_feature_consistency_across_methods(self):
        """Test that features are consistent when created in different orders."""
        # Create features in one order
        result1 = self.feature_engineer.create_lag_features(self.sample_data)
        result1 = self.feature_engineer.create_rolling_features(result1)
        result1 = self.feature_engineer.create_seasonal_features(result1)
        
        # Create features in different order
        result2 = self.feature_engineer.create_seasonal_features(self.sample_data)
        result2 = self.feature_engineer.create_lag_features(result2)
        result2 = self.feature_engineer.create_rolling_features(result2)
        
        # Should have same columns (order might differ)
        self.assertEqual(set(result1.columns), set(result2.columns))
        
        # Core data should be the same
        for col in self.sample_data.columns:
            pd.testing.assert_series_equal(result1[col], result2[col])
    
    def test_memory_efficiency(self):
        """Test that feature engineering doesn't create excessive memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create comprehensive feature set
        result = self.feature_engineer.create_lag_features(self.sample_data)
        result = self.feature_engineer.create_rolling_features(result)
        result = self.feature_engineer.create_seasonal_features(result)
        result = self.feature_engineer.create_economic_indicators(result)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (less than 100MB for test data)
        self.assertLess(memory_increase, 100)


if __name__ == '__main__':
    unittest.main()