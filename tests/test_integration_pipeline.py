"""
Integration tests for the complete inflation prediction pipeline.
Tests end-to-end pipeline execution, output validation, and error handling.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import yaml
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src directory to path
sys.path.append('src')

from main import InflationPredictionPipeline


class TestInflationPredictionPipeline(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / 'data'
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.models_dir = Path(self.temp_dir) / 'models'
        self.reports_dir = Path(self.temp_dir) / 'reports'
        self.logs_dir = Path(self.temp_dir) / 'logs'
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.models_dir, self.reports_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create test configuration
        self.test_config = {
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2023-12-31',
                'ine_base_url': 'https://test.ine.es/api/',
                'urls': {
                    'ipc_general': 'https://test.ine.es/api/ipc_general',
                    'ipc_groups': 'https://test.ine.es/api/ipc_groups',
                    'ipca': 'https://test.ine.es/api/ipca'
                },
                'retry': {
                    'max_attempts': 2,
                    'backoff_factor': 1,
                    'timeout': 5
                }
            },
            'models': {
                'arima': {
                    'max_p': 3,
                    'max_d': 2,
                    'max_q': 3
                },
                'random_forest': {
                    'n_estimators': 10,
                    'max_depth': 5
                },
                'lstm': {
                    'epochs': 5,
                    'batch_size': 16,
                    'hidden_units': 20
                }
            },
            'paths': {
                'data': {
                    'raw': str(self.raw_dir) + '/',
                    'processed': str(self.processed_dir) + '/'
                },
                'models': str(self.models_dir) + '/',
                'reports': str(self.reports_dir) + '/',
                'logs': str(self.logs_dir) + '/'
            },
            'output': {
                'csv_encoding': 'utf-8',
                'decimal_places': 4,
                'date_format': '%Y-%m-%d'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'test_pipeline.log'
            },
            'prediction': {
                'horizon_months': 6
            },
            'feature_engineering': {
                'lag_periods': [1, 3, 6],
                'rolling_windows': [3, 6],
                'seasonal_features': True,
                'economic_indicators': True
            }
        }
        
        # Create config file
        self.config_file = Path(self.temp_dir) / 'test_config.yaml'
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.test_config, f)
        
        # Create sample data files
        self._create_sample_data_files()
    
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            try:
                # Close any open log handlers
                import logging
                for handler in logging.root.handlers[:]:
                    handler.close()
                    logging.root.removeHandler(handler)
                
                shutil.rmtree(self.temp_dir)
            except (PermissionError, OSError):
                # On Windows, sometimes files are locked, try again
                import time
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.temp_dir)
                except (PermissionError, OSError):
                    pass  # Ignore cleanup errors in tests
    
    def _create_sample_data_files(self):
        """Create sample data files for testing."""
        # Create sample dates and data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        np.random.seed(42)
        
        # Sample IPC data
        base_ipc = 100
        ipc_values = []
        for i in range(len(dates)):
            trend = i * 0.2
            seasonal = 2 * np.sin(2 * np.pi * i / 12)
            noise = np.random.normal(0, 0.5)
            ipc_values.append(base_ipc + trend + seasonal + noise)
        
        # Create IPC general data
        ipc_general_data = pd.DataFrame({
            'fecha': dates.strftime('%Y-%m-%d'),
            'ipc_general': ipc_values,
            'data_type': 'general',
            'extraction_date': '2024-01-01 12:00:00'
        })
        
        # Create IPC groups data
        ipc_groups_data = pd.DataFrame({
            'fecha': dates.strftime('%Y-%m-%d'),
            'ipc_alimentacion': [val + np.random.normal(0, 1) for val in ipc_values],
            'ipc_vivienda': [val + np.random.normal(0, 0.8) for val in ipc_values],
            'ipc_transporte': [val + np.random.normal(0, 1.2) for val in ipc_values],
            'data_type': 'groups',
            'extraction_date': '2024-01-01 12:00:00'
        })
        
        # Create IPCA data
        ipca_data = pd.DataFrame({
            'fecha': dates.strftime('%Y-%m-%d'),
            'ipca': [val + np.random.normal(0, 0.3) for val in ipc_values],
            'data_type': 'ipca',
            'extraction_date': '2024-01-01 12:00:00'
        })
        
        # Save sample data files
        ipc_general_data.to_csv(self.raw_dir / 'ipc_general_2020-01-01_2023-12-31.csv', index=False)
        ipc_groups_data.to_csv(self.raw_dir / 'ipc_groups_2020-01-01_2023-12-31.csv', index=False)
        ipca_data.to_csv(self.raw_dir / 'ipca_2020-01-01_2023-12-31.csv', index=False)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with configuration."""
        pipeline = InflationPredictionPipeline(str(self.config_file))
        
        # Check that pipeline is initialized correctly
        self.assertIsNotNone(pipeline.config)
        self.assertIsNotNone(pipeline.logger)
        self.assertEqual(pipeline.config['data']['start_date'], '2020-01-01')
        self.assertEqual(pipeline.config['data']['end_date'], '2023-12-31')
    
    def test_configuration_validation_success(self):
        """Test successful configuration validation."""
        pipeline = InflationPredictionPipeline(str(self.config_file))
        
        # Should validate successfully
        self.assertTrue(pipeline._validate_configuration())
    
    def test_configuration_validation_missing_section(self):
        """Test configuration validation with missing required section."""
        # Create invalid config
        invalid_config = self.test_config.copy()
        del invalid_config['data']
        
        invalid_config_file = Path(self.temp_dir) / 'invalid_config.yaml'
        with open(invalid_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(invalid_config, f)
        
        pipeline = InflationPredictionPipeline(str(invalid_config_file))
        
        # Should fail validation
        self.assertFalse(pipeline._validate_configuration())
    
    def test_module_initialization(self):
        """Test initialization of all pipeline modules."""
        pipeline = InflationPredictionPipeline(str(self.config_file))
        pipeline._validate_configuration()
        pipeline._initialize_modules()
        
        # Check that all modules are initialized
        expected_modules = ['extractor', 'processor', 'feature_engineer', 'model_trainer', 'predictor', 'report_generator']
        for module_name in expected_modules:
            self.assertIn(module_name, pipeline.modules)
            self.assertIsNotNone(pipeline.modules[module_name])
    
    @patch('src.ine_extractor.INEExtractor.test_all_connections')
    @patch('src.ine_extractor.INEExtractor.export_all_data')
    def test_step_1_data_extraction_success(self, mock_export, mock_connections):
        """Test successful data extraction step."""
        # Mock successful connections and export
        mock_connections.return_value = {'ipc_general': True, 'ipc_groups': True, 'ipca': True}
        mock_export.return_value = {
            'ipc_general': str(self.raw_dir / 'ipc_general_2020-01-01_2023-12-31.csv'),
            'ipc_groups': str(self.raw_dir / 'ipc_groups_2020-01-01_2023-12-31.csv'),
            'ipca': str(self.raw_dir / 'ipca_2020-01-01_2023-12-31.csv')
        }
        
        pipeline = InflationPredictionPipeline(str(self.config_file))
        pipeline._validate_configuration()
        pipeline._initialize_modules()
        
        result = pipeline.step_1_data_extraction()
        
        # Check result
        self.assertEqual(result['status'], 'success')
        self.assertIn('exported_files', result)
        self.assertIn('connection_results', result)
        self.assertEqual(len(result['exported_files']), 3)
    
    def test_step_2_data_processing_success(self):
        """Test successful data processing step."""
        pipeline = InflationPredictionPipeline(str(self.config_file))
        pipeline._validate_configuration()
        pipeline._initialize_modules()
        
        # Set up pipeline state with extraction results
        pipeline.pipeline_state['results']['data_extraction'] = {
            'exported_files': {
                'ipc_general': str(self.raw_dir / 'ipc_general_2020-01-01_2023-12-31.csv'),
                'ipc_groups': str(self.raw_dir / 'ipc_groups_2020-01-01_2023-12-31.csv'),
                'ipca': str(self.raw_dir / 'ipca_2020-01-01_2023-12-31.csv')
            }
        }
        
        result = pipeline.step_2_data_processing()
        
        # Check result
        self.assertEqual(result['status'], 'success')
        self.assertIn('processed_data', result)
        self.assertEqual(result['total_files_processed'], 3)
        
        # Check that processed files were created
        for data_type in ['ipc_general', 'ipc_groups', 'ipca']:
            processed_file = self.processed_dir / f'processed_{data_type}.csv'
            self.assertTrue(processed_file.exists())
    
    def test_step_3_feature_engineering_success(self):
        """Test successful feature engineering step."""
        pipeline = InflationPredictionPipeline(str(self.config_file))
        pipeline._validate_configuration()
        pipeline._initialize_modules()
        
        # Run data processing first
        pipeline.pipeline_state['results']['data_extraction'] = {
            'exported_files': {
                'ipc_general': str(self.raw_dir / 'ipc_general_2020-01-01_2023-12-31.csv'),
                'ipc_groups': str(self.raw_dir / 'ipc_groups_2020-01-01_2023-12-31.csv'),
                'ipca': str(self.raw_dir / 'ipca_2020-01-01_2023-12-31.csv')
            }
        }
        
        processing_result = pipeline.step_2_data_processing()
        pipeline.pipeline_state['results']['data_processing'] = processing_result
        
        # Run feature engineering
        result = pipeline.step_3_feature_engineering()
        
        # Check result
        self.assertEqual(result['status'], 'success')
        self.assertIn('engineered_data', result)
        self.assertIn('feature_summary', result)
        
        # Check that features file was created
        features_file = self.processed_dir / 'engineered_features.csv'
        self.assertTrue(features_file.exists())
        
        # Check feature summary
        summary = result['feature_summary']
        self.assertGreater(summary['total_features'], summary['original_features'])
        self.assertGreater(summary['lag_features_count'], 0)
        self.assertGreater(summary['rolling_features_count'], 0)
    
    @patch('src.model_trainer.ModelTrainer.train_arima')
    @patch('src.model_trainer.ModelTrainer.train_random_forest')
    @patch('src.model_trainer.ModelTrainer.train_lstm')
    @patch('src.model_trainer.ModelTrainer.evaluate_models')
    @patch('src.model_trainer.ModelTrainer.select_best_model')
    @patch('src.model_trainer.ModelTrainer.save_models')
    def test_step_4_model_training_success(self, mock_save, mock_select, mock_evaluate, 
                                         mock_lstm, mock_rf, mock_arima):
        """Test successful model training step."""
        # Mock model training results
        mock_arima.return_value = {'status': 'success', 'model': 'arima_model', 'validation': {'mae': 0.5}}
        mock_rf.return_value = {'status': 'success', 'model': 'rf_model', 'validation': {'mae': 0.4}}
        mock_lstm.return_value = {'status': 'success', 'model': 'lstm_model', 'validation': {'mae': 0.3}}
        
        mock_evaluate.return_value = {
            'arima': {'metrics': {'MAE': 0.5, 'RMSE': 0.7, 'MAPE': 5.0}},
            'random_forest': {'metrics': {'MAE': 0.4, 'RMSE': 0.6, 'MAPE': 4.0}},
            'lstm': {'metrics': {'MAE': 0.3, 'RMSE': 0.5, 'MAPE': 3.0}}
        }
        
        mock_select.return_value = {'model_name': 'lstm', 'model_type': 'LSTM', 'best_score': 0.3}
        mock_save.return_value = {'lstm': str(self.models_dir / 'lstm_model.pkl')}
        
        pipeline = InflationPredictionPipeline(str(self.config_file))
        pipeline._validate_configuration()
        pipeline._initialize_modules()
        
        # Set up previous steps
        self._setup_pipeline_for_model_training(pipeline)
        
        result = pipeline.step_4_model_training()
        
        # Check result
        self.assertEqual(result['status'], 'success')
        self.assertIn('trained_models', result)
        self.assertIn('evaluation_results', result)
        self.assertIn('best_model', result)
        self.assertEqual(result['best_model']['model_name'], 'lstm')
    
    @patch('src.predictor.Predictor.load_best_model')
    @patch('src.predictor.Predictor.generate_predictions')
    @patch('src.predictor.Predictor.validate_predictions')
    @patch('src.predictor.Predictor.export_predictions_csv')
    @patch('src.predictor.Predictor.export_predictions_json')
    @patch('src.predictor.Predictor.get_prediction_summary')
    def test_step_5_prediction_generation_success(self, mock_summary, mock_json, mock_csv,
                                                 mock_validate, mock_generate, mock_load):
        """Test successful prediction generation step."""
        # Mock prediction results
        sample_predictions = pd.DataFrame({
            'fecha': pd.date_range('2024-01-01', periods=6, freq='M'),
            'predicted_inflation': [2.1, 2.2, 2.0, 1.9, 2.1, 2.3],
            'confidence_lower': [1.8, 1.9, 1.7, 1.6, 1.8, 2.0],
            'confidence_upper': [2.4, 2.5, 2.3, 2.2, 2.4, 2.6]
        })
        
        mock_load.return_value = {'model_type': 'LSTM', 'model': 'mock_model'}
        mock_generate.return_value = sample_predictions
        mock_validate.return_value = {'valid': True, 'issues': []}
        mock_csv.return_value = str(self.reports_dir / 'predictions.csv')
        mock_json.return_value = str(self.reports_dir / 'predictions.json')
        mock_summary.return_value = {'mean_prediction': 2.1, 'prediction_range': [1.9, 2.3]}
        
        pipeline = InflationPredictionPipeline(str(self.config_file))
        pipeline._validate_configuration()
        pipeline._initialize_modules()
        
        # Set up previous steps
        self._setup_pipeline_for_prediction(pipeline)
        
        result = pipeline.step_5_prediction_generation()
        
        # Check result
        self.assertEqual(result['status'], 'success')
        self.assertIn('predictions', result)
        self.assertIn('model_used', result)
        self.assertIn('validation_results', result)
        self.assertIn('exported_files', result)
    
    @patch('src.report_generator.ReportGenerator.create_visualizations')
    @patch('src.report_generator.ReportGenerator.generate_economic_analysis')
    @patch('src.report_generator.ReportGenerator.create_technical_report')
    @patch('src.report_generator.ReportGenerator.export_code_screenshots')
    def test_step_6_report_generation_success(self, mock_screenshots, mock_report,
                                            mock_analysis, mock_viz):
        """Test successful report generation step."""
        # Mock report generation results
        mock_viz.return_value = ['chart1.png', 'chart2.png', 'chart3.png']
        mock_analysis.return_value = {'summary': 'Economic analysis summary', 'trends': ['trend1', 'trend2']}
        mock_report.return_value = str(self.reports_dir / 'technical_report.pdf')
        mock_screenshots.return_value = {'code_docs': str(self.reports_dir / 'code_documentation.zip')}
        
        pipeline = InflationPredictionPipeline(str(self.config_file))
        pipeline._validate_configuration()
        pipeline._initialize_modules()
        
        # Set up previous steps
        self._setup_pipeline_for_report_generation(pipeline)
        
        result = pipeline.step_6_report_generation()
        
        # Check result
        self.assertEqual(result['status'], 'success')
        self.assertIn('visualizations', result)
        self.assertIn('economic_analysis', result)
        self.assertIn('pdf_report', result)
        self.assertIn('code_documentation', result)
    
    def test_pipeline_error_handling_critical_step(self):
        """Test pipeline error handling for critical steps."""
        pipeline = InflationPredictionPipeline(str(self.config_file))
        pipeline._validate_configuration()
        pipeline._initialize_modules()
        
        # Simulate error in critical step (data extraction)
        with patch.object(pipeline, 'step_1_data_extraction', side_effect=Exception("Connection failed")):
            # Should handle error and not continue
            should_continue = pipeline._handle_step_error('data_extraction', Exception("Connection failed"))
            self.assertFalse(should_continue)
            
            # Check pipeline state
            self.assertIn('data_extraction', pipeline.pipeline_state['failed_steps'])
            self.assertEqual(pipeline.pipeline_state['status'], 'failed')
    
    def test_pipeline_error_handling_non_critical_step(self):
        """Test pipeline error handling for non-critical steps."""
        pipeline = InflationPredictionPipeline(str(self.config_file))
        pipeline._validate_configuration()
        pipeline._initialize_modules()
        
        # Simulate error in non-critical step (report generation)
        should_continue = pipeline._handle_step_error('report_generation', Exception("Report failed"))
        self.assertTrue(should_continue)
        
        # Check pipeline state
        self.assertIn('report_generation', pipeline.pipeline_state['failed_steps'])
    
    def test_pipeline_performance_monitoring(self):
        """Test pipeline performance monitoring functionality."""
        pipeline = InflationPredictionPipeline(str(self.config_file))
        pipeline._validate_configuration()
        pipeline._initialize_modules()
        
        # Test performance monitoring
        import time
        start_time = time.time()
        time.sleep(0.1)  # Simulate some work
        
        pipeline._monitor_performance('test_step', start_time)
        
        # Check that metrics were recorded
        self.assertIn('test_step', pipeline.performance_metrics['execution_times'])
        self.assertIn('test_step', pipeline.performance_metrics['memory_usage'])
        self.assertIn('test_step', pipeline.performance_metrics['cpu_usage'])
        self.assertGreater(pipeline.performance_metrics['execution_times']['test_step'], 0.1)
    
    def test_pipeline_state_tracking(self):
        """Test pipeline state tracking functionality."""
        pipeline = InflationPredictionPipeline(str(self.config_file))
        
        # Test state updates
        pipeline._update_pipeline_state('test_step', 'completed', {'result': 'success'})
        
        self.assertEqual(pipeline.pipeline_state['current_step'], 'test_step')
        self.assertIn('test_step', pipeline.pipeline_state['completed_steps'])
        self.assertEqual(pipeline.pipeline_state['results']['test_step']['result'], 'success')
    
    def test_output_file_validation(self):
        """Test validation of output files and formats."""
        pipeline = InflationPredictionPipeline(str(self.config_file))
        pipeline._validate_configuration()
        pipeline._initialize_modules()
        
        # Run data processing to create output files
        pipeline.pipeline_state['results']['data_extraction'] = {
            'exported_files': {
                'ipc_general': str(self.raw_dir / 'ipc_general_2020-01-01_2023-12-31.csv'),
                'ipc_groups': str(self.raw_dir / 'ipc_groups_2020-01-01_2023-12-31.csv'),
                'ipca': str(self.raw_dir / 'ipca_2020-01-01_2023-12-31.csv')
            }
        }
        
        result = pipeline.step_2_data_processing()
        
        # Validate output files
        for data_type in ['ipc_general', 'ipc_groups', 'ipca']:
            processed_file = self.processed_dir / f'processed_{data_type}.csv'
            self.assertTrue(processed_file.exists())
            
            # Validate file content
            df = pd.read_csv(processed_file)
            self.assertGreater(len(df), 0)
            self.assertIn('fecha', df.columns)
            
            # Check data types
            self.assertTrue(pd.api.types.is_object_dtype(df['fecha']))
    
    def test_memory_optimization(self):
        """Test memory optimization functionality."""
        pipeline = InflationPredictionPipeline(str(self.config_file))
        
        # Test memory optimization
        optimization_result = pipeline._optimize_memory_usage('test_step')
        
        # Check optimization result structure
        required_keys = ['memory_before_mb', 'memory_after_mb', 'memory_freed_mb', 'objects_collected']
        for key in required_keys:
            self.assertIn(key, optimization_result)
        
        # Check that metrics are recorded
        self.assertIn('test_step', pipeline.performance_metrics['memory_optimization'])
    
    def test_pipeline_status_reporting(self):
        """Test pipeline status reporting functionality."""
        pipeline = InflationPredictionPipeline(str(self.config_file))
        
        # Set up some pipeline state
        pipeline.pipeline_state['status'] = 'running'
        pipeline.pipeline_state['completed_steps'] = ['step1', 'step2']
        pipeline.pipeline_state['failed_steps'] = []
        
        status = pipeline.get_pipeline_status()
        
        # Check status structure
        required_keys = ['status', 'completed_steps', 'failed_steps', 'total_steps']
        for key in required_keys:
            self.assertIn(key, status)
        
        self.assertEqual(status['status'], 'running')
        self.assertEqual(status['completed_steps'], 2)
        self.assertEqual(status['total_steps'], 6)
    
    def test_create_status_report(self):
        """Test status report creation."""
        pipeline = InflationPredictionPipeline(str(self.config_file))
        
        # Set up pipeline state
        pipeline.pipeline_state['completed_steps'] = ['step1', 'step2']
        pipeline.performance_metrics['execution_times'] = {'step1': 10.5, 'step2': 15.2}
        pipeline.performance_metrics['memory_usage'] = {'step1': 100.0, 'step2': 120.0}
        
        report_path = pipeline.create_status_report()
        
        # Check that report file was created
        self.assertTrue(os.path.exists(report_path))
        
        # Check report content
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('PIPELINE STATUS', content)
            self.assertIn('step1', content)
            self.assertIn('step2', content)
    
    def _setup_pipeline_for_model_training(self, pipeline):
        """Helper method to set up pipeline state for model training tests."""
        # Mock data extraction results
        pipeline.pipeline_state['results']['data_extraction'] = {
            'exported_files': {
                'ipc_general': str(self.raw_dir / 'ipc_general_2020-01-01_2023-12-31.csv')
            }
        }
        
        # Mock data processing results
        sample_data = pd.DataFrame({
            'fecha': pd.date_range('2020-01-01', '2023-12-31', freq='M'),
            'ipc_general': np.random.normal(100, 5, 48),
            'ipc_general_annual_rate': np.random.normal(2.5, 1.5, 48)
        })
        
        pipeline.pipeline_state['results']['data_processing'] = {
            'processed_data': {
                'ipc_general': {'data': sample_data}
            }
        }
        
        # Mock feature engineering results
        pipeline.pipeline_state['results']['feature_engineering'] = {
            'engineered_data': sample_data
        }
    
    def _setup_pipeline_for_prediction(self, pipeline):
        """Helper method to set up pipeline state for prediction tests."""
        self._setup_pipeline_for_model_training(pipeline)
        
        # Mock model training results
        pipeline.pipeline_state['results']['model_training'] = {
            'best_model': {'model_name': 'lstm', 'model_type': 'LSTM'},
            'saved_models': {'lstm': str(self.models_dir / 'lstm_model.pkl')}
        }
    
    def _setup_pipeline_for_report_generation(self, pipeline):
        """Helper method to set up pipeline state for report generation tests."""
        self._setup_pipeline_for_prediction(pipeline)
        
        # Mock prediction results
        sample_predictions = pd.DataFrame({
            'fecha': pd.date_range('2024-01-01', periods=6, freq='M'),
            'predicted_inflation': [2.1, 2.2, 2.0, 1.9, 2.1, 2.3]
        })
        
        pipeline.pipeline_state['results']['prediction_generation'] = {
            'predictions': sample_predictions
        }


if __name__ == '__main__':
    unittest.main()