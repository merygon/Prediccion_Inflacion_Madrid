"""
Unit tests for DataProcessor class.
Tests data cleaning functions, missing value handling, outlier detection,
date normalization, and inflation rate calculations.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src directory to path
sys.path.append('src')

from data_cleaner import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = DataProcessor()
        
        # Create sample data for testing
        self.sample_dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        np.random.seed(42)
        
        # Sample IPC data with realistic values
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
            'ipc_vivienda': [val + np.random.normal(0, 0.8) for val in self.sample_ipc_values]
        })
    
    def test_load_raw_data_success(self):
        """Test successful loading of CSV data."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Test loading
            loaded_data = self.processor.load_raw_data(temp_file)
            
            # Assertions
            self.assertIsInstance(loaded_data, pd.DataFrame)
            self.assertEqual(len(loaded_data), len(self.sample_data))
            self.assertEqual(list(loaded_data.columns), list(self.sample_data.columns))
            
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_load_raw_data_file_not_found(self):
        """Test loading non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_raw_data('non_existent_file.csv')
    
    def test_load_raw_data_empty_file(self):
        """Test loading empty CSV file raises EmptyDataError."""
        # Create empty CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('')  # Empty file
            temp_file = f.name
        
        try:
            with self.assertRaises(pd.errors.EmptyDataError):
                self.processor.load_raw_data(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_handle_missing_values_numeric(self):
        """Test handling missing values in numeric columns."""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[5:7, 'ipc_general'] = np.nan
        data_with_missing.loc[10:12, 'ipc_alimentacion'] = np.nan
        
        # Test missing value handling
        cleaned_data = self.processor.handle_missing_values(data_with_missing)
        
        # Assertions
        self.assertEqual(cleaned_data.isnull().sum().sum(), 0)  # No missing values
        self.assertEqual(len(cleaned_data), len(data_with_missing))  # Same length
        
        # Check interpolation worked (values should be reasonable)
        self.assertTrue(all(cleaned_data['ipc_general'] > 90))
        self.assertTrue(all(cleaned_data['ipc_general'] < 120))
    
    def test_handle_missing_values_categorical(self):
        """Test handling missing values in categorical columns."""
        # Create data with categorical column and missing values
        data_with_categorical = self.sample_data.copy()
        category_values = ['A', 'B', 'C'] * (len(data_with_categorical) // 3 + 1)
        data_with_categorical['category'] = category_values[:len(data_with_categorical)]
        data_with_categorical.loc[5:7, 'category'] = np.nan
        
        # Test missing value handling
        cleaned_data = self.processor.handle_missing_values(data_with_categorical)
        
        # Assertions
        self.assertEqual(cleaned_data['category'].isnull().sum(), 0)  # No missing values
        self.assertTrue(all(cleaned_data['category'].isin(['A', 'B', 'C'])))
    
    def test_detect_outliers_iqr_method(self):
        """Test outlier detection using IQR method."""
        # Create data with known outliers
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[10, 'ipc_general'] = 200  # Clear outlier
        data_with_outliers.loc[20, 'ipc_alimentacion'] = -50  # Clear outlier
        
        # Test outlier detection
        outlier_indices = self.processor.detect_outliers(data_with_outliers)
        
        # Assertions
        self.assertIsInstance(outlier_indices, list)
        self.assertIn(10, outlier_indices)  # Should detect the outlier
        self.assertIn(20, outlier_indices)  # Should detect the outlier
    
    def test_detect_outliers_no_outliers(self):
        """Test outlier detection with clean data."""
        outlier_indices = self.processor.detect_outliers(self.sample_data)
        
        # Should find few or no outliers in clean synthetic data
        self.assertIsInstance(outlier_indices, list)
        self.assertLessEqual(len(outlier_indices), len(self.sample_data) * 0.1)  # Less than 10%
    
    def test_normalize_dates_fecha_column(self):
        """Test date normalization with 'fecha' column."""
        # Create data with string dates
        data_with_string_dates = self.sample_data.copy()
        data_with_string_dates['fecha'] = data_with_string_dates['fecha'].dt.strftime('%Y-%m-%d')
        
        # Test date normalization
        normalized_data = self.processor.normalize_dates(data_with_string_dates)
        
        # Assertions
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(normalized_data['fecha']))
        self.assertEqual(len(normalized_data), len(data_with_string_dates))
    
    def test_normalize_dates_auto_detection(self):
        """Test automatic date column detection."""
        # Create data with date-like column
        data_with_date_col = self.sample_data.copy()
        data_with_date_col['date_column'] = data_with_date_col['fecha'].dt.strftime('%Y-%m-%d')
        data_with_date_col = data_with_date_col.drop('fecha', axis=1)
        
        # Test date normalization
        normalized_data = self.processor.normalize_dates(data_with_date_col)
        
        # Should detect and convert the date column
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(normalized_data['date_column']))
    
    def test_calculate_inflation_rates_ipc_columns(self):
        """Test inflation rate calculation for IPC columns."""
        # Test inflation rate calculation
        rates_data = self.processor.calculate_inflation_rates(self.sample_data)
        
        # Check that new rate columns were created
        expected_columns = [
            'ipc_general_monthly_rate',
            'ipc_general_annual_rate',
            'ipc_alimentacion_monthly_rate',
            'ipc_alimentacion_annual_rate',
            'ipc_vivienda_monthly_rate',
            'ipc_vivienda_annual_rate'
        ]
        
        for col in expected_columns:
            self.assertIn(col, rates_data.columns)
        
        # Check that rates are reasonable (not NaN for most values)
        monthly_rate = rates_data['ipc_general_monthly_rate'].dropna()
        annual_rate = rates_data['ipc_general_annual_rate'].dropna()
        
        self.assertGreater(len(monthly_rate), len(self.sample_data) * 0.8)  # Most values should be valid
        self.assertGreater(len(annual_rate), len(self.sample_data) * 0.7)   # Annual needs 12 months
        
        # Check rate ranges are reasonable (inflation typically -5% to 15% annually)
        self.assertTrue(all(annual_rate.abs() < 50))  # Reasonable range
    
    def test_calculate_inflation_rates_accuracy(self):
        """Test accuracy of inflation rate calculations."""
        # Create simple test data with known inflation
        simple_data = pd.DataFrame({
            'ipc_general': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
        })
        
        rates_data = self.processor.calculate_inflation_rates(simple_data)
        
        # Monthly rate should be approximately 1% (except first value which is NaN)
        monthly_rates = rates_data['ipc_general_monthly_rate'].dropna()
        expected_monthly_rate = 1.0  # 1% monthly
        
        # Allow some tolerance for floating point precision
        self.assertTrue(all(abs(monthly_rates - expected_monthly_rate) < 0.1))
        
        # Annual rate at index 12 should be approximately 12% ((112-100)/100 * 100)
        annual_rate_12 = rates_data['ipc_general_annual_rate'].iloc[12]
        expected_annual_rate = 12.0
        self.assertAlmostEqual(annual_rate_12, expected_annual_rate, places=1)
    
    def test_generate_statistics_basic_info(self):
        """Test generation of basic dataset statistics."""
        stats = self.processor.generate_statistics(self.sample_data)
        
        # Check required sections
        required_sections = ['dataset_info', 'missing_values', 'numeric_statistics', 'data_quality']
        for section in required_sections:
            self.assertIn(section, stats)
        
        # Check dataset info
        dataset_info = stats['dataset_info']
        self.assertEqual(dataset_info['total_rows'], len(self.sample_data))
        self.assertEqual(dataset_info['total_columns'], len(self.sample_data.columns))
        self.assertGreater(dataset_info['memory_usage_mb'], 0)
    
    def test_generate_statistics_missing_values(self):
        """Test missing values analysis in statistics."""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[5:7, 'ipc_general'] = np.nan
        
        stats = self.processor.generate_statistics(data_with_missing)
        
        # Check missing values section
        missing_info = stats['missing_values']
        self.assertGreater(missing_info['total_missing'], 0)
        self.assertGreater(missing_info['missing_percentage'], 0)
        self.assertIn('ipc_general', missing_info['columns_with_missing'])
    
    def test_generate_statistics_numeric_analysis(self):
        """Test numeric statistics generation."""
        stats = self.processor.generate_statistics(self.sample_data)
        
        # Check numeric statistics
        numeric_stats = stats['numeric_statistics']
        self.assertIn('summary', numeric_stats)
        self.assertIn('skewness', numeric_stats)
        self.assertIn('kurtosis', numeric_stats)
        
        # Check that all numeric columns are analyzed
        numeric_columns = self.sample_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            self.assertIn(col, numeric_stats['summary'])
            self.assertIn(col, numeric_stats['skewness'])
            self.assertIn(col, numeric_stats['kurtosis'])
    
    def test_generate_statistics_data_quality(self):
        """Test data quality metrics generation."""
        stats = self.processor.generate_statistics(self.sample_data)
        
        # Check data quality section
        quality_info = stats['data_quality']
        self.assertIn('duplicate_rows', quality_info)
        self.assertIn('duplicate_percentage', quality_info)
        self.assertIn('completeness_score', quality_info)
        
        # For clean sample data, should have no duplicates and high completeness
        self.assertEqual(quality_info['duplicate_rows'], 0)
        self.assertEqual(quality_info['duplicate_percentage'], 0.0)
        self.assertEqual(quality_info['completeness_score'], 100.0)
    
    def test_generate_statistics_inflation_specific(self):
        """Test inflation-specific statistics generation."""
        # Add inflation rate columns
        rates_data = self.processor.calculate_inflation_rates(self.sample_data)
        stats = self.processor.generate_statistics(rates_data)
        
        # Should have inflation statistics section
        if 'inflation_statistics' in stats:
            inflation_stats = stats['inflation_statistics']
            
            # Check for rate columns
            rate_columns = [col for col in rates_data.columns if 'rate' in col.lower()]
            for col in rate_columns:
                if col in inflation_stats:
                    col_stats = inflation_stats[col]
                    required_metrics = ['mean_rate', 'median_rate', 'std_rate', 'min_rate', 'max_rate']
                    for metric in required_metrics:
                        self.assertIn(metric, col_stats)
    
    def test_edge_case_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        # Should handle empty DataFrame gracefully
        cleaned_empty = self.processor.handle_missing_values(empty_df)
        self.assertTrue(cleaned_empty.empty)
        
        outliers_empty = self.processor.detect_outliers(empty_df)
        self.assertEqual(outliers_empty, [])
    
    def test_edge_case_single_row(self):
        """Test handling of single-row DataFrame."""
        single_row = self.sample_data.iloc[:1].copy()
        
        # Should handle single row gracefully
        cleaned_single = self.processor.handle_missing_values(single_row)
        self.assertEqual(len(cleaned_single), 1)
        
        outliers_single = self.processor.detect_outliers(single_row)
        self.assertIsInstance(outliers_single, list)
    
    def test_edge_case_all_missing_values(self):
        """Test handling of column with all missing values."""
        data_all_missing = self.sample_data.copy()
        data_all_missing['all_missing'] = np.nan
        
        cleaned_data = self.processor.handle_missing_values(data_all_missing)
        
        # Should still have the column, but values might still be NaN
        self.assertIn('all_missing', cleaned_data.columns)
    
    def test_data_types_preservation(self):
        """Test that data types are preserved during processing."""
        # Test with mixed data types
        mixed_data = self.sample_data.copy()
        mixed_data['integer_col'] = range(len(mixed_data))
        mixed_data['string_col'] = ['test'] * len(mixed_data)
        
        # Process data
        cleaned_data = self.processor.handle_missing_values(mixed_data)
        normalized_data = self.processor.normalize_dates(cleaned_data)
        
        # Check that appropriate types are maintained
        self.assertTrue(pd.api.types.is_numeric_dtype(normalized_data['ipc_general']))
        self.assertTrue(pd.api.types.is_integer_dtype(normalized_data['integer_col']))
        self.assertTrue(pd.api.types.is_object_dtype(normalized_data['string_col']))


if __name__ == '__main__':
    unittest.main()