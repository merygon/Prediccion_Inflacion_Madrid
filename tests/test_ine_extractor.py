"""
Unit tests for INEExtractor class.
Tests data extraction, connection handling, retry logic, and data validation.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import requests
import sys

# Add src directory to path
sys.path.append('src')

from ine_extractor import INEExtractor


class TestINEExtractor(unittest.TestCase):
    """Test cases for INEExtractor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary config file
        self.temp_config = {
            'data': {
                'ine_base_url': 'https://test.ine.es/api/',
                'urls': {
                    'ipc_general': 'https://test.ine.es/api/ipc_general',
                    'ipc_groups': 'https://test.ine.es/api/ipc_groups',
                    'ipca': 'https://test.ine.es/api/ipca'
                },
                'retry': {
                    'max_attempts': 3,
                    'backoff_factor': 1,
                    'timeout': 10
                }
            },
            'paths': {
                'data': {
                    'raw': 'data/raw/'
                }
            },
            'output': {
                'csv_encoding': 'utf-8',
                'decimal_places': 4,
                'date_format': '%Y-%m-%d'
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(self.temp_config, f)
            self.config_file = f.name
        
        # Create sample response data
        self.sample_response_data = [
            {'Fecha': '2023-01-01', 'Valor': 105.2},
            {'Fecha': '2023-02-01', 'Valor': 105.8},
            {'Fecha': '2023-03-01', 'Valor': 106.1}
        ]
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'config_file') and os.path.exists(self.config_file):
            os.unlink(self.config_file)
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    def test_load_config_success(self, mock_open, mock_yaml_load):
        """Test successful configuration loading."""
        mock_yaml_load.return_value = self.temp_config
        mock_open.return_value.__enter__.return_value = Mock()
        
        extractor = INEExtractor(self.config_file)
        
        self.assertEqual(extractor.base_url, self.temp_config['data']['ine_base_url'])
        self.assertEqual(extractor.max_attempts, self.temp_config['data']['retry']['max_attempts'])
    
    def test_load_config_file_not_found(self):
        """Test configuration loading with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            INEExtractor('non_existent_config.yaml')
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    def test_create_session(self, mock_open, mock_yaml_load):
        """Test HTTP session creation."""
        mock_yaml_load.return_value = self.temp_config
        
        extractor = INEExtractor(self.config_file)
        
        # Check session headers
        self.assertIn('User-Agent', extractor.session.headers)
        self.assertIn('Accept', extractor.session.headers)
        self.assertEqual(extractor.session.headers['Accept'], 'application/json')
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    @patch('requests.Session.get')
    def test_make_request_with_retry_success(self, mock_get, mock_open, mock_yaml_load):
        """Test successful HTTP request."""
        mock_yaml_load.return_value = self.temp_config
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        extractor = INEExtractor(self.config_file)
        response = extractor._make_request_with_retry('https://test.url')
        
        self.assertEqual(response, mock_response)
        mock_get.assert_called_once()
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    @patch('requests.Session.get')
    @patch('time.sleep')
    def test_make_request_with_retry_failure(self, mock_sleep, mock_get, mock_open, mock_yaml_load):
        """Test HTTP request with retry on failure."""
        mock_yaml_load.return_value = self.temp_config
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
        
        extractor = INEExtractor(self.config_file)
        
        with self.assertRaises(requests.exceptions.RequestException):
            extractor._make_request_with_retry('https://test.url')
        
        # Should retry max_attempts times
        self.assertEqual(mock_get.call_count, self.temp_config['data']['retry']['max_attempts'])
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    def test_parse_ine_response_list_format(self, mock_open, mock_yaml_load):
        """Test parsing INE response in list format."""
        mock_yaml_load.return_value = self.temp_config
        
        extractor = INEExtractor(self.config_file)
        
        # Mock response with list data
        mock_response = Mock()
        mock_response.json.return_value = self.sample_response_data
        
        df = extractor._parse_ine_response(mock_response)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.sample_response_data))
        self.assertIn('Fecha', df.columns)
        self.assertIn('Valor', df.columns)
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    def test_parse_ine_response_dict_format(self, mock_open, mock_yaml_load):
        """Test parsing INE response in dictionary format."""
        mock_yaml_load.return_value = self.temp_config
        
        extractor = INEExtractor(self.config_file)
        
        # Mock response with dict data
        mock_response = Mock()
        mock_response.json.return_value = {'Data': self.sample_response_data}
        
        df = extractor._parse_ine_response(mock_response)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.sample_response_data))
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    def test_parse_ine_response_invalid_format(self, mock_open, mock_yaml_load):
        """Test parsing invalid INE response format."""
        mock_yaml_load.return_value = self.temp_config
        
        extractor = INEExtractor(self.config_file)
        
        # Mock response with invalid data
        mock_response = Mock()
        mock_response.json.return_value = "invalid_data"
        mock_response.text = "invalid response text"
        
        with self.assertRaises(ValueError):
            extractor._parse_ine_response(mock_response)
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    @patch.object(INEExtractor, '_make_request_with_retry')
    def test_connection_tests(self, mock_request, mock_open, mock_yaml_load):
        """Test connection testing methods."""
        mock_yaml_load.return_value = self.temp_config
        mock_request.return_value = Mock()
        
        extractor = INEExtractor(self.config_file)
        
        # Test individual connections
        self.assertTrue(extractor.get_ipc_general_connection())
        self.assertTrue(extractor.get_ipc_groups_connection())
        self.assertTrue(extractor.get_ipca_connection())
        
        # Test all connections
        results = extractor.test_all_connections()
        self.assertIn('ipc_general', results)
        self.assertIn('ipc_groups', results)
        self.assertIn('ipca', results)
        self.assertTrue(all(results.values()))
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    @patch.object(INEExtractor, '_make_request_with_retry')
    @patch.object(INEExtractor, '_parse_ine_response')
    @patch.object(INEExtractor, '_standardize_ipc_general_columns')
    def test_download_ipc_general_success(self, mock_standardize, mock_parse, mock_request, mock_open, mock_yaml_load):
        """Test successful IPC general data download."""
        mock_yaml_load.return_value = self.temp_config
        mock_request.return_value = Mock()
        
        # Mock parsed DataFrame
        mock_df = pd.DataFrame(self.sample_response_data)
        mock_parse.return_value = mock_df
        mock_standardize.return_value = mock_df
        
        extractor = INEExtractor(self.config_file)
        result = extractor.download_ipc_general('2023-01-01', '2023-12-31')
        
        self.assertIsInstance(result, pd.DataFrame)
        mock_request.assert_called_once()
        mock_parse.assert_called_once()
        mock_standardize.assert_called_once()
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    def test_download_invalid_date_format(self, mock_open, mock_yaml_load):
        """Test download with invalid date format."""
        mock_yaml_load.return_value = self.temp_config
        
        extractor = INEExtractor(self.config_file)
        
        with self.assertRaises(ValueError):
            extractor.download_ipc_general('invalid-date', '2023-12-31')
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    def test_standardize_ipc_general_columns(self, mock_open, mock_yaml_load):
        """Test IPC general column standardization."""
        mock_yaml_load.return_value = self.temp_config
        
        extractor = INEExtractor(self.config_file)
        
        # Test with various column names
        test_df = pd.DataFrame({
            'Fecha': ['2023-01-01', '2023-02-01'],
            'Valor': [105.2, 105.8]
        })
        
        standardized = extractor._standardize_ipc_general_columns(test_df)
        
        self.assertIn('fecha', standardized.columns)
        self.assertIn('ipc_general', standardized.columns)
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    def test_standardize_ipc_groups_columns(self, mock_open, mock_yaml_load):
        """Test IPC groups column standardization."""
        mock_yaml_load.return_value = self.temp_config
        
        extractor = INEExtractor(self.config_file)
        
        # Test with groups data
        test_df = pd.DataFrame({
            'Fecha': ['2023-01-01', '2023-02-01'],
            'Alimentos y bebidas no alcohólicas': [105.2, 105.8],
            'Vivienda': [103.1, 103.5]
        })
        
        standardized = extractor._standardize_ipc_groups_columns(test_df)
        
        self.assertIn('fecha', standardized.columns)
        self.assertIn('ipc_alimentacion', standardized.columns)
        self.assertIn('ipc_vivienda', standardized.columns)
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    def test_validate_data_for_export_success(self, mock_open, mock_yaml_load):
        """Test successful data validation for export."""
        mock_yaml_load.return_value = self.temp_config
        
        extractor = INEExtractor(self.config_file)
        
        # Valid data
        valid_data = pd.DataFrame({
            'fecha': ['2023-01-01', '2023-02-01'],
            'ipc_general': [105.2, 105.8]
        })
        
        self.assertTrue(extractor._validate_data_for_export(valid_data, 'general'))
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    def test_validate_data_for_export_empty(self, mock_open, mock_yaml_load):
        """Test data validation with empty DataFrame."""
        mock_yaml_load.return_value = self.temp_config
        
        extractor = INEExtractor(self.config_file)
        
        # Empty data
        empty_data = pd.DataFrame()
        
        self.assertFalse(extractor._validate_data_for_export(empty_data, 'general'))
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    def test_validate_data_for_export_missing_columns(self, mock_open, mock_yaml_load):
        """Test data validation with missing required columns."""
        mock_yaml_load.return_value = self.temp_config
        
        extractor = INEExtractor(self.config_file)
        
        # Data missing required columns
        invalid_data = pd.DataFrame({
            'wrong_column': [105.2, 105.8]
        })
        
        self.assertFalse(extractor._validate_data_for_export(invalid_data, 'general'))
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    @patch('os.makedirs')
    def test_save_to_csv_success(self, mock_makedirs, mock_open, mock_yaml_load):
        """Test successful CSV saving."""
        mock_yaml_load.return_value = self.temp_config
        
        extractor = INEExtractor(self.config_file)
        
        # Valid data
        valid_data = pd.DataFrame({
            'fecha': pd.to_datetime(['2023-01-01', '2023-02-01']),
            'ipc_general': [105.2, 105.8]
        })
        
        # Mock the to_csv method
        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            extractor.save_to_csv(valid_data, 'test_file', 'general')
            mock_to_csv.assert_called_once()
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    def test_prepare_data_for_export(self, mock_open, mock_yaml_load):
        """Test data preparation for export."""
        mock_yaml_load.return_value = self.temp_config
        
        extractor = INEExtractor(self.config_file)
        
        # Test data
        test_data = pd.DataFrame({
            'fecha': pd.to_datetime(['2023-01-01', '2023-02-01']),
            'ipc_general': [105.2, 105.8]
        })
        
        prepared_data = extractor._prepare_data_for_export(test_data, 'general')
        
        # Check that metadata columns were added
        self.assertIn('data_type', prepared_data.columns)
        self.assertIn('extraction_date', prepared_data.columns)
        
        # Check that data is sorted by date
        self.assertTrue(prepared_data['fecha'].is_monotonic_increasing)
    
    @patch('src.ine_extractor.yaml.safe_load')
    @patch('builtins.open')
    def test_get_data_summary(self, mock_open, mock_yaml_load):
        """Test data summary generation."""
        mock_yaml_load.return_value = self.temp_config
        
        extractor = INEExtractor(self.config_file)
        
        # Test data
        test_data = pd.DataFrame({
            'fecha': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
            'ipc_general': [105.2, 105.8, 106.1]
        })
        
        summary = extractor.get_data_summary(test_data, 'general')
        
        # Check summary structure
        required_keys = ['data_type', 'total_records', 'date_range', 'columns', 'missing_values', 'numeric_summary']
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Check specific values
        self.assertEqual(summary['data_type'], 'general')
        self.assertEqual(summary['total_records'], 3)
        self.assertEqual(summary['date_range']['total_months'], 3)
    
    def test_edge_case_malformed_json(self):
        """Test handling of malformed JSON response."""
        with patch('src.ine_extractor.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.temp_config
            
            extractor = INEExtractor(self.config_file)
            
            # Mock response with malformed JSON
            mock_response = Mock()
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.text = "malformed response"
            
            with self.assertRaises(json.JSONDecodeError):
                extractor._parse_ine_response(mock_response)
    
    def test_edge_case_network_timeout(self):
        """Test handling of network timeout."""
        with patch('src.ine_extractor.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.temp_config
            
            with patch('requests.Session.get') as mock_get:
                mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
                
                extractor = INEExtractor(self.config_file)
                
                with self.assertRaises(requests.exceptions.Timeout):
                    extractor._make_request_with_retry('https://test.url')


if __name__ == '__main__':
    unittest.main()