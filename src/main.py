"""
Main pipeline orchestrator for Spanish inflation prediction system.
Coordinates sequential execution of all modules with error handling and logging.
"""

import os
import sys
import logging
import time
import traceback
import psutil
import gc
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import yaml
import json
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from ine_extractor import INEExtractor
from data_cleaner import DataProcessor
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from predictor import Predictor
from report_generator import ReportGenerator


class InflationPredictionPipeline:
    """
    Main pipeline orchestrator for inflation prediction system.
    
    Coordinates the sequential execution of all modules:
    1. Data extraction from INE
    2. Data cleaning and processing
    3. Feature engineering
    4. Model training and evaluation
    5. Prediction generation
    6. Report generation
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Pipeline state tracking
        self.pipeline_state = {
            'start_time': None,
            'end_time': None,
            'duration': None,
            'status': 'not_started',
            'current_step': None,
            'completed_steps': [],
            'failed_steps': [],
            'results': {}
        }
        
        # Performance monitoring
        self.performance_metrics = {
            'memory_usage': {},
            'execution_times': {},
            'cpu_usage': {},
            'memory_optimization': {},
            'system_resources': {}
        }
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        self.resource_history = []
        
        # Initialize modules
        self.modules = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        # Base logs directory from config (may be relative)
        logs_dir = Path(self.config.get('paths', {}).get('logs', 'logs/'))
        # Ensure the logs directory exists; final log file parent will be created below
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        log_level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO'))
        log_format = self.config.get('logging', {}).get('format', 
                                   '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Resolve log file path robustly to avoid doubling directories (e.g. 'logs/logs/...')
        log_file_setting = self.config.get('logging', {}).get('file', 'inflation_prediction.log')
        log_file_path = Path(log_file_setting)

        if log_file_path.is_absolute():
            # Absolute path provided in config
            log_file = log_file_path
        else:
            # If the configured path already starts with the logs dir name (e.g. 'logs/...'),
            # use it as-is (relative to project root). Otherwise join logs_dir with the basename
            # to avoid creating paths like 'logs/logs/inflation_prediction.log'.
            if log_file_path.parts and log_file_path.parts[0] == logs_dir.name:
                log_file = log_file_path
            elif log_file_path.parent != Path('.'):
                # e.g. 'some/subdir/name.log' -> place only the file name inside logs_dir
                log_file = logs_dir / log_file_path.name
            else:
                log_file = logs_dir / log_file_path

        # Ensure parent directory for the resolved log file exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        logger = logging.getLogger('InflationPipeline')
        logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _validate_configuration(self) -> bool:
        """Validate configuration parameters."""
        self.logger.info("Validating configuration...")
        
        required_sections = ['data', 'models', 'paths']
        for section in required_sections:
            if section not in self.config:
                self.logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate data configuration
        data_config = self.config.get('data', {})
        required_data_keys = ['start_date', 'end_date', 'urls']
        for key in required_data_keys:
            if key not in data_config:
                self.logger.error(f"Missing required data configuration: {key}")
                return False
        
        # Validate paths
        paths_config = self.config.get('paths', {})
        required_paths = ['data', 'models', 'reports']
        for path_key in required_paths:
            if path_key not in paths_config:
                self.logger.error(f"Missing required path configuration: {path_key}")
                return False
        
        # Create directories
        for path_key, path_config in paths_config.items():
            if isinstance(path_config, dict):
                for subpath in path_config.values():
                    Path(subpath).mkdir(parents=True, exist_ok=True)
            else:
                Path(path_config).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Configuration validation completed successfully")
        return True
    
    def _initialize_modules(self) -> None:
        """Initialize all pipeline modules."""
        self.logger.info("Initializing pipeline modules...")
        
        try:
            self.modules = {
                'extractor': INEExtractor(self.config_path),
                'processor': DataProcessor(),
                'feature_engineer': FeatureEngineer(self.config_path),
                'model_trainer': ModelTrainer(self.config_path),
                'predictor': Predictor(self.config_path),
                'report_generator': ReportGenerator(self.config_path)
            }
            self.logger.info("All modules initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing modules: {e}")
            raise
    
    def _start_resource_monitoring(self) -> None:
        """Start background resource monitoring."""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_resources_background)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        self.logger.debug("Background resource monitoring started")
    
    def _stop_resource_monitoring(self) -> None:
        """Stop background resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        self.logger.debug("Background resource monitoring stopped")
    
    def _monitor_resources_background(self) -> None:
        """Background thread for continuous resource monitoring."""
        while self.monitoring_active:
            try:
                process = psutil.Process()
                
                # System resources
                system_memory = psutil.virtual_memory()
                system_cpu = psutil.cpu_percent(interval=1)
                
                # Process resources
                memory_info = process.memory_info()
                process_memory_mb = memory_info.rss / 1024 / 1024
                process_cpu = process.cpu_percent()
                
                # Store resource snapshot
                resource_snapshot = {
                    'timestamp': time.time(),
                    'system_memory_percent': system_memory.percent,
                    'system_memory_available_gb': system_memory.available / 1024 / 1024 / 1024,
                    'system_cpu_percent': system_cpu,
                    'process_memory_mb': process_memory_mb,
                    'process_cpu_percent': process_cpu
                }
                
                self.resource_history.append(resource_snapshot)
                
                # Keep only last 100 snapshots to prevent memory buildup
                if len(self.resource_history) > 100:
                    self.resource_history = self.resource_history[-100:]
                
                # Check for resource warnings
                self._check_resource_warnings(resource_snapshot)
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.debug(f"Resource monitoring error: {e}")
                time.sleep(5)
    
    def _check_resource_warnings(self, snapshot: Dict[str, Any]) -> None:
        """Check for resource usage warnings."""
        # Memory warnings
        if snapshot['system_memory_percent'] > 90:
            self.logger.warning(f"High system memory usage: {snapshot['system_memory_percent']:.1f}%")
        
        if snapshot['process_memory_mb'] > 2048:  # 2GB
            self.logger.warning(f"High process memory usage: {snapshot['process_memory_mb']:.1f}MB")
        
        # CPU warnings
        if snapshot['system_cpu_percent'] > 90:
            self.logger.warning(f"High system CPU usage: {snapshot['system_cpu_percent']:.1f}%")
    
    def _optimize_memory_usage(self, step_name: str) -> Dict[str, Any]:
        """Optimize memory usage after each step."""
        # Get memory before optimization
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory after optimization
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_freed = memory_before - memory_after
        
        optimization_result = {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_freed_mb': memory_freed,
            'objects_collected': collected
        }
        
        self.performance_metrics['memory_optimization'][step_name] = optimization_result
        
        if memory_freed > 10:  # Only log if significant memory was freed
            self.logger.info(f"Memory optimization for '{step_name}': "
                           f"freed {memory_freed:.1f}MB, collected {collected} objects")
        
        return optimization_result
    
    def _monitor_performance(self, step_name: str, start_time: float) -> None:
        """Monitor performance metrics for a pipeline step."""
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # CPU usage
        cpu_percent = process.cpu_percent()
        
        # System resources
        system_memory = psutil.virtual_memory()
        system_cpu = psutil.cpu_percent()
        
        # Store metrics
        self.performance_metrics['execution_times'][step_name] = execution_time
        self.performance_metrics['memory_usage'][step_name] = memory_mb
        self.performance_metrics['cpu_usage'][step_name] = cpu_percent
        self.performance_metrics['system_resources'][step_name] = {
            'system_memory_percent': system_memory.percent,
            'system_cpu_percent': system_cpu,
            'available_memory_gb': system_memory.available / 1024 / 1024 / 1024
        }
        
        # Optimize memory after step completion
        self._optimize_memory_usage(step_name)
        
        self.logger.info(f"Step '{step_name}' completed in {execution_time:.2f}s, "
                        f"Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%, "
                        f"System Memory: {system_memory.percent:.1f}%")
    
    def _update_pipeline_state(self, step_name: str, status: str, result: Any = None) -> None:
        """Update pipeline state tracking."""
        self.pipeline_state['current_step'] = step_name
        
        if status == 'completed':
            self.pipeline_state['completed_steps'].append(step_name)
            if result is not None:
                self.pipeline_state['results'][step_name] = result
        elif status == 'failed':
            self.pipeline_state['failed_steps'].append(step_name)
            self.pipeline_state['status'] = 'failed'
    
    def _handle_step_error(self, step_name: str, error: Exception) -> bool:
        """
        Handle errors in pipeline steps.
        
        Args:
            step_name (str): Name of the failed step
            error (Exception): The exception that occurred
            
        Returns:
            bool: True if pipeline should continue, False if it should stop
        """
        self.logger.error(f"Error in step '{step_name}': {str(error)}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        self._update_pipeline_state(step_name, 'failed')
        
        # Define critical steps that should stop the pipeline
        critical_steps = ['data_extraction', 'data_processing']
        
        if step_name in critical_steps:
            self.logger.error(f"Critical step '{step_name}' failed. Stopping pipeline.")
            return False
        else:
            self.logger.warning(f"Non-critical step '{step_name}' failed. Continuing pipeline.")
            return True
    
    def step_1_data_extraction(self) -> Dict[str, Any]:
        """Step 1: Extract data from INE API."""
        step_name = "data_extraction"
        self.logger.info(f"Starting {step_name}...")
        start_time = time.time()
        
        try:
            extractor = self.modules['extractor']
            
            # Get date range from config
            start_date = self.config['data']['start_date']
            end_date = self.config['data']['end_date']
            
            # Test connections first
            self.logger.info("Testing INE API connections...")
            connection_results = extractor.test_all_connections()
            
            if not any(connection_results.values()):
                raise Exception("All INE API connections failed")
            
            # Download all data types
            self.logger.info(f"Downloading INE data from {start_date} to {end_date}...")
            exported_files = extractor.export_all_data(start_date, end_date)
            
            result = {
                'status': 'success',
                'exported_files': exported_files,
                'connection_results': connection_results,
                'date_range': {'start': start_date, 'end': end_date}
            }
            
            self._monitor_performance(step_name, start_time)
            self._update_pipeline_state(step_name, 'completed', result)
            
            self.logger.info(f"Data extraction completed successfully")
            return result
            
        except Exception as e:
            self._handle_step_error(step_name, e)
            raise
    
    def step_2_data_processing(self) -> Dict[str, Any]:
        """Step 2: Clean and process the downloaded data."""
        step_name = "data_processing"
        self.logger.info(f"Starting {step_name}...")
        start_time = time.time()
        
        try:
            processor = self.modules['processor']
            
            # Get raw data files from previous step
            extraction_result = self.pipeline_state['results'].get('data_extraction', {})
            exported_files = extraction_result.get('exported_files', {})
            
            if not exported_files:
                raise Exception("No data files available from extraction step")
            
            processed_data = {}
            
            # Process each data type
            for data_type, file_path in exported_files.items():
                self.logger.info(f"Processing {data_type} data from {file_path}")
                
                # Load raw data
                raw_data = processor.load_raw_data(file_path)
                
                # Clean and process
                cleaned_data = processor.handle_missing_values(raw_data)
                cleaned_data = processor.normalize_dates(cleaned_data)
                cleaned_data = processor.calculate_inflation_rates(cleaned_data)
                
                # Detect outliers (but don't remove them automatically)
                outliers = processor.detect_outliers(cleaned_data)
                
                # Generate statistics
                statistics = processor.generate_statistics(cleaned_data)
                
                # Save processed data
                processed_dir = Path(self.config['paths']['data']['processed'])
                processed_file = processed_dir / f"processed_{data_type}.csv"
                cleaned_data.to_csv(processed_file, index=False, encoding='utf-8')
                
                processed_data[data_type] = {
                    'data': cleaned_data,
                    'file_path': str(processed_file),
                    'outliers': outliers,
                    'statistics': statistics
                }
            
            result = {
                'status': 'success',
                'processed_data': processed_data,
                'total_files_processed': len(processed_data)
            }
            
            self._monitor_performance(step_name, start_time)
            self._update_pipeline_state(step_name, 'completed', result)
            
            self.logger.info(f"Data processing completed successfully")
            return result
            
        except Exception as e:
            self._handle_step_error(step_name, e)
            raise
    
    def step_3_feature_engineering(self) -> Dict[str, Any]:
        """Step 3: Create features for machine learning models."""
        step_name = "feature_engineering"
        self.logger.info(f"Starting {step_name}...")
        start_time = time.time()
        
        try:
            feature_engineer = self.modules['feature_engineer']
            
            # Get processed data from previous step
            processing_result = self.pipeline_state['results'].get('data_processing', {})
            processed_data = processing_result.get('processed_data', {})
            
            if not processed_data:
                raise Exception("No processed data available from processing step")
            
            # Use IPC general data for feature engineering
            ipc_general_data = processed_data.get('ipc_general', {}).get('data')
            if ipc_general_data is None:
                raise Exception("IPC general data not found")
            
            self.logger.info("Creating lag features...")
            data_with_lags = feature_engineer.create_lag_features(ipc_general_data)
            
            self.logger.info("Creating rolling features...")
            data_with_rolling = feature_engineer.create_rolling_features(data_with_lags)
            
            self.logger.info("Creating seasonal features...")
            data_with_seasonal = feature_engineer.create_seasonal_features(data_with_rolling)
            
            self.logger.info("Creating economic indicators...")
            final_features = feature_engineer.create_economic_indicators(data_with_seasonal)
            
            # Generate feature summary
            feature_summary = feature_engineer.get_feature_summary(final_features)
            
            # Save engineered features
            features_dir = Path(self.config['paths']['data']['processed'])
            features_file = features_dir / "engineered_features.csv"
            final_features.to_csv(features_file, index=False, encoding='utf-8')
            
            result = {
                'status': 'success',
                'engineered_data': final_features,
                'features_file': str(features_file),
                'feature_summary': feature_summary
            }
            
            self._monitor_performance(step_name, start_time)
            self._update_pipeline_state(step_name, 'completed', result)
            
            self.logger.info(f"Feature engineering completed successfully")
            return result
            
        except Exception as e:
            self._handle_step_error(step_name, e)
            raise
    
    def step_4_model_training(self) -> Dict[str, Any]:
        """Step 4: Train and evaluate machine learning models."""
        step_name = "model_training"
        self.logger.info(f"Starting {step_name}...")
        start_time = time.time()
        
        try:
            model_trainer = self.modules['model_trainer']
            
            # Get engineered features from previous step
            feature_result = self.pipeline_state['results'].get('feature_engineering', {})
            engineered_data = feature_result.get('engineered_data')
            
            if engineered_data is None:
                raise Exception("No engineered features available from feature engineering step")
            
            trained_models = {}
            
            # Train ARIMA model
            self.logger.info("Training ARIMA model...")
            try:
                arima_result = model_trainer.train_arima(engineered_data)
                trained_models['arima'] = arima_result
            except Exception as e:
                self.logger.warning(f"ARIMA training failed: {e}")
                trained_models['arima'] = {'status': 'failed', 'error': str(e)}
            
            # Prepare data for ML models
            # Find target column (inflation rate)
            target_columns = [col for col in engineered_data.columns 
                            if 'annual_rate' in col.lower() or 'inflacion' in col.lower()]
            
            if target_columns:
                target_column = target_columns[0]
                
                # Prepare features (exclude target and date columns)
                feature_columns = [col for col in engineered_data.columns 
                                 if col != target_column and 'fecha' not in col.lower()]
                
                X = engineered_data[feature_columns]
                y = engineered_data[target_column]
                
                # Train Random Forest model
                self.logger.info("Training Random Forest model...")
                try:
                    rf_result = model_trainer.train_random_forest(X, y)
                    trained_models['random_forest'] = rf_result
                except Exception as e:
                    self.logger.warning(f"Random Forest training failed: {e}")
                    trained_models['random_forest'] = {'status': 'failed', 'error': str(e)}
                
                # Train LSTM model
                self.logger.info("Training LSTM model...")
                try:
                    lstm_result = model_trainer.train_lstm(engineered_data, target_column)
                    trained_models['lstm'] = lstm_result
                except Exception as e:
                    self.logger.warning(f"LSTM training failed: {e}")
                    trained_models['lstm'] = {'status': 'failed', 'error': str(e)}
            
            # Evaluate all models
            self.logger.info("Evaluating models...")
            evaluation_results = model_trainer.evaluate_models(trained_models)
            
            # Select best model
            try:
                best_model_info = model_trainer.select_best_model(evaluation_results)
            except Exception as e:
                self.logger.warning(f"Best model selection failed: {e}")
                best_model_info = None
            
            # Save models
            self.logger.info("Saving trained models...")
            saved_models = model_trainer.save_models(self.config['paths']['models'])
            
            result = {
                'status': 'success',
                'trained_models': trained_models,
                'evaluation_results': evaluation_results,
                'best_model': best_model_info,
                'saved_models': saved_models
            }
            
            self._monitor_performance(step_name, start_time)
            self._update_pipeline_state(step_name, 'completed', result)
            
            self.logger.info(f"Model training completed successfully")
            return result
            
        except Exception as e:
            self._handle_step_error(step_name, e)
            raise
    
    def step_5_prediction_generation(self) -> Dict[str, Any]:
        """Step 5: Generate predictions using the best model."""
        step_name = "prediction_generation"
        self.logger.info(f"Starting {step_name}...")
        start_time = time.time()
        
        try:
            predictor = self.modules['predictor']
            
            # Get model training results
            training_result = self.pipeline_state['results'].get('model_training', {})
            best_model_info = training_result.get('best_model')
            saved_models = training_result.get('saved_models', {})
            
            if not best_model_info:
                self.logger.warning("No best model identified, using first available model")
                if saved_models:
                    model_name = list(saved_models.keys())[0]
                    model_path = saved_models[model_name]
                else:
                    raise Exception("No trained models available")
            else:
                model_name = best_model_info['model_name']
                model_path = saved_models.get(model_name)
                if not model_path:
                    raise Exception(f"Model file not found for {model_name}")
            
            # Load the best model
            self.logger.info(f"Loading model: {model_name}")
            model_info = predictor.load_best_model(model_path)
            
            # Get input data for prediction
            feature_result = self.pipeline_state['results'].get('feature_engineering', {})
            input_data = feature_result.get('engineered_data')
            
            # Generate predictions
            horizon = self.config.get('prediction', {}).get('horizon_months', 12)
            self.logger.info(f"Generating {horizon}-month predictions...")
            
            if model_info['model_type'] == 'ARIMA':
                predictions = predictor.generate_predictions(horizon)
            else:
                predictions = predictor.generate_predictions(horizon, input_data)
            
            # Validate predictions
            self.logger.info("Validating predictions...")
            validation_results = predictor.validate_predictions(predictions)
            
            # Export predictions
            reports_dir = Path(self.config['paths']['reports'])
            
            # Export to CSV
            csv_path = predictor.export_predictions_csv(
                predictions, 
                str(reports_dir / "predictions.csv")
            )
            
            # Export to JSON
            json_path = predictor.export_predictions_json(
                predictions, 
                str(reports_dir / "predictions.json")
            )
            
            # Generate prediction summary
            prediction_summary = predictor.get_prediction_summary(predictions)
            
            result = {
                'status': 'success',
                'predictions': predictions,
                'model_used': model_name,
                'validation_results': validation_results,
                'prediction_summary': prediction_summary,
                'exported_files': {
                    'csv': csv_path,
                    'json': json_path
                }
            }
            
            self._monitor_performance(step_name, start_time)
            self._update_pipeline_state(step_name, 'completed', result)
            
            self.logger.info(f"Prediction generation completed successfully")
            return result
            
        except Exception as e:
            self._handle_step_error(step_name, e)
            raise
    
    def step_6_report_generation(self) -> Dict[str, Any]:
        """Step 6: Generate comprehensive reports and visualizations."""
        step_name = "report_generation"
        self.logger.info(f"Starting {step_name}...")
        start_time = time.time()
        
        try:
            report_generator = self.modules['report_generator']
            
            # Get data from previous steps
            processing_result = self.pipeline_state['results'].get('data_processing', {})
            prediction_result = self.pipeline_state['results'].get('prediction_generation', {})
            training_result = self.pipeline_state['results'].get('model_training', {})
            
            # Get historical data
            processed_data = processing_result.get('processed_data', {})
            historical_data = processed_data.get('ipc_general', {}).get('data')
            
            # Get predictions
            predictions = prediction_result.get('predictions')
            
            # Get model results
            evaluation_results = training_result.get('evaluation_results', {})
            
            if historical_data is None or predictions is None:
                raise Exception("Missing required data for report generation")
            
            # Create visualizations
            self.logger.info("Creating visualizations...")
            visualizations = report_generator.create_visualizations(
                historical_data, predictions, evaluation_results
            )
            
            # Generate economic analysis
            self.logger.info("Generating economic analysis...")
            economic_analysis = report_generator.generate_economic_analysis(
                historical_data, predictions, evaluation_results
            )
            
            # Create technical report
            self.logger.info("Creating technical PDF report...")
            pdf_report_path = report_generator.create_technical_report(
                economic_analysis, visualizations, evaluation_results
            )
            
            # Export code documentation
            self.logger.info("Creating code documentation...")
            code_docs = report_generator.export_code_screenshots()
            
            result = {
                'status': 'success',
                'visualizations': visualizations,
                'economic_analysis': economic_analysis,
                'pdf_report': pdf_report_path,
                'code_documentation': code_docs
            }
            
            self._monitor_performance(step_name, start_time)
            self._update_pipeline_state(step_name, 'completed', result)
            
            self.logger.info(f"Report generation completed successfully")
            return result
            
        except Exception as e:
            self._handle_step_error(step_name, e)
            raise
    
    def _estimate_pipeline_duration(self) -> float:
        """Estimate total pipeline duration based on system resources."""
        # Base estimates in seconds for each step
        base_estimates = {
            'data_extraction': 60,
            'data_processing': 30,
            'feature_engineering': 45,
            'model_training': 300,  # 5 minutes
            'prediction_generation': 15,
            'report_generation': 60
        }
        
        # Adjust based on system resources
        system_memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        # Memory adjustment factor (less memory = slower)
        memory_factor = max(0.5, min(2.0, system_memory.available / (4 * 1024**3)))  # 4GB baseline
        
        # CPU adjustment factor (fewer cores = slower)
        cpu_factor = max(0.5, min(2.0, cpu_count / 4))  # 4 cores baseline
        
        # Calculate adjusted estimates
        total_estimate = 0
        for step, base_time in base_estimates.items():
            adjusted_time = base_time / (memory_factor * cpu_factor)
            total_estimate += adjusted_time
        
        return total_estimate
    
    def _create_progress_reporter(self, total_steps: int) -> Callable[[int], None]:
        """Create a progress reporting function."""
        def report_progress(completed_steps: int):
            progress_percent = (completed_steps / total_steps) * 100
            elapsed_time = (datetime.now() - self.pipeline_state['start_time']).total_seconds()
            
            if completed_steps > 0:
                estimated_total_time = elapsed_time * (total_steps / completed_steps)
                remaining_time = estimated_total_time - elapsed_time
                
                self.logger.info(f"Progress: {progress_percent:.1f}% "
                               f"({completed_steps}/{total_steps} steps), "
                               f"Elapsed: {elapsed_time:.0f}s, "
                               f"Estimated remaining: {remaining_time:.0f}s")
        
        return report_progress
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete inflation prediction pipeline.
        
        Returns:
            Dict[str, Any]: Complete pipeline results and status
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING SPANISH INFLATION PREDICTION PIPELINE")
        self.logger.info("=" * 80)
        
        # Initialize pipeline state
        self.pipeline_state['start_time'] = datetime.now()
        self.pipeline_state['status'] = 'running'
        
        # Start resource monitoring
        self._start_resource_monitoring()
        
        try:
            # System information
            system_info = {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'memory_available_gb': psutil.virtual_memory().available / 1024**3,
                'python_version': sys.version,
                'platform': sys.platform
            }
            
            self.logger.info(f"System Info: {system_info['cpu_count']} CPUs, "
                           f"{system_info['memory_total_gb']:.1f}GB total memory, "
                           f"{system_info['memory_available_gb']:.1f}GB available")
            
            # Estimate pipeline duration
            estimated_duration = self._estimate_pipeline_duration()
            self.logger.info(f"Estimated pipeline duration: {estimated_duration/60:.1f} minutes")
            
            # Validate configuration
            if not self._validate_configuration():
                raise Exception("Configuration validation failed")
            
            # Initialize modules
            self._initialize_modules()
            
            # Execute pipeline steps
            pipeline_steps = [
                ('data_extraction', 'Step 1: Data Extraction', self.step_1_data_extraction),
                ('data_processing', 'Step 2: Data Processing', self.step_2_data_processing),
                ('feature_engineering', 'Step 3: Feature Engineering', self.step_3_feature_engineering),
                ('model_training', 'Step 4: Model Training', self.step_4_model_training),
                ('prediction_generation', 'Step 5: Prediction Generation', self.step_5_prediction_generation),
                ('report_generation', 'Step 6: Report Generation', self.step_6_report_generation)
            ]
            
            # Create progress reporter
            progress_reporter = self._create_progress_reporter(len(pipeline_steps))
            
            for i, (step_name, step_description, step_function) in enumerate(pipeline_steps):
                self.logger.info(f"\n{'-' * 60}")
                self.logger.info(f"EXECUTING: {step_description}")
                self.logger.info(f"{'-' * 60}")
                
                # Report progress
                progress_reporter(i)
                
                try:
                    step_result = step_function()
                    self.logger.info(f"✓ {step_description} completed successfully")
                except Exception as e:
                    self.logger.error(f"✗ {step_description} failed: {str(e)}")
                    if not self._handle_step_error(step_name, e):
                        break
            
            # Final progress report
            progress_reporter(len(self.pipeline_state['completed_steps']))
            
            # Finalize pipeline
            self.pipeline_state['end_time'] = datetime.now()
            self.pipeline_state['duration'] = (
                self.pipeline_state['end_time'] - self.pipeline_state['start_time']
            ).total_seconds()
            
            # Determine final status
            if self.pipeline_state['failed_steps']:
                if len(self.pipeline_state['completed_steps']) == 0:
                    self.pipeline_state['status'] = 'failed'
                else:
                    self.pipeline_state['status'] = 'partial_success'
            else:
                self.pipeline_state['status'] = 'success'
            
            # Add system info to results
            self.pipeline_state['system_info'] = system_info
            
            # Generate pipeline summary
            self._generate_pipeline_summary()
            
            return self.pipeline_state
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            self.pipeline_state['end_time'] = datetime.now()
            self.pipeline_state['status'] = 'failed'
            self.pipeline_state['error'] = str(e)
            
            return self.pipeline_state
        
        finally:
            # Stop resource monitoring
            self._stop_resource_monitoring()
    
    def _generate_pipeline_summary(self) -> None:
        """Generate and log pipeline execution summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        
        # Basic information
        self.logger.info(f"Status: {self.pipeline_state['status'].upper()}")
        self.logger.info(f"Duration: {self.pipeline_state['duration']:.2f} seconds")
        self.logger.info(f"Completed Steps: {len(self.pipeline_state['completed_steps'])}")
        self.logger.info(f"Failed Steps: {len(self.pipeline_state['failed_steps'])}")
        
        # Step details
        if self.pipeline_state['completed_steps']:
            self.logger.info(f"\nCompleted Steps:")
            for step in self.pipeline_state['completed_steps']:
                execution_time = self.performance_metrics['execution_times'].get(step, 0)
                memory_usage = self.performance_metrics['memory_usage'].get(step, 0)
                self.logger.info(f"  ✓ {step} ({execution_time:.2f}s, {memory_usage:.1f}MB)")
        
        if self.pipeline_state['failed_steps']:
            self.logger.info(f"\nFailed Steps:")
            for step in self.pipeline_state['failed_steps']:
                self.logger.info(f"  ✗ {step}")
        
        # Performance summary
        total_execution_time = sum(self.performance_metrics['execution_times'].values())
        max_memory_usage = max(self.performance_metrics['memory_usage'].values()) if self.performance_metrics['memory_usage'] else 0
        total_memory_freed = sum(opt.get('memory_freed_mb', 0) for opt in self.performance_metrics['memory_optimization'].values())
        
        self.logger.info(f"\nPerformance Summary:")
        self.logger.info(f"  Total Execution Time: {total_execution_time:.2f} seconds")
        self.logger.info(f"  Peak Memory Usage: {max_memory_usage:.1f} MB")
        self.logger.info(f"  Total Memory Freed: {total_memory_freed:.1f} MB")
        
        # Resource monitoring summary
        if self.resource_history:
            avg_system_memory = sum(r['system_memory_percent'] for r in self.resource_history) / len(self.resource_history)
            avg_system_cpu = sum(r['system_cpu_percent'] for r in self.resource_history) / len(self.resource_history)
            max_process_memory = max(r['process_memory_mb'] for r in self.resource_history)
            
            self.logger.info(f"  Average System Memory Usage: {avg_system_memory:.1f}%")
            self.logger.info(f"  Average System CPU Usage: {avg_system_cpu:.1f}%")
            self.logger.info(f"  Peak Process Memory: {max_process_memory:.1f} MB")
        
        # Output files summary
        self.logger.info(f"\nGenerated Outputs:")
        
        # Check for key outputs
        prediction_result = self.pipeline_state['results'].get('prediction_generation', {})
        if prediction_result.get('exported_files'):
            for file_type, file_path in prediction_result['exported_files'].items():
                self.logger.info(f"  • Predictions ({file_type.upper()}): {file_path}")
        
        report_result = self.pipeline_state['results'].get('report_generation', {})
        if report_result.get('pdf_report'):
            self.logger.info(f"  • Technical Report: {report_result['pdf_report']}")
        
        if report_result.get('visualizations'):
            self.logger.info(f"  • Visualizations: {len(report_result['visualizations'])} charts created")
        
        # Save pipeline state
        self._save_pipeline_state()
        
        self.logger.info("=" * 80)
    
    def _save_pipeline_state(self) -> None:
        """Save pipeline state to JSON file."""
        try:
            reports_dir = Path(self.config['paths']['reports'])
            state_file = reports_dir / "pipeline_execution_state.json"
            
            # Prepare state for JSON serialization
            serializable_state = self.pipeline_state.copy()
            
            # Convert datetime objects to strings
            if serializable_state['start_time']:
                serializable_state['start_time'] = serializable_state['start_time'].isoformat()
            if serializable_state['end_time']:
                serializable_state['end_time'] = serializable_state['end_time'].isoformat()
            
            # Remove non-serializable objects from results
            if 'results' in serializable_state:
                for step_name, result in serializable_state['results'].items():
                    if isinstance(result, dict) and 'engineered_data' in result:
                        # Remove DataFrame objects
                        result = result.copy()
                        result['engineered_data'] = f"DataFrame with {len(result['engineered_data'])} rows"
                        serializable_state['results'][step_name] = result
            
            # Add performance metrics
            serializable_state['performance_metrics'] = self.performance_metrics
            
            # Add resource monitoring summary
            if self.resource_history:
                resource_summary = {
                    'monitoring_duration': len(self.resource_history) * 5,  # 5 seconds per snapshot
                    'avg_system_memory_percent': sum(r['system_memory_percent'] for r in self.resource_history) / len(self.resource_history),
                    'avg_system_cpu_percent': sum(r['system_cpu_percent'] for r in self.resource_history) / len(self.resource_history),
                    'max_process_memory_mb': max(r['process_memory_mb'] for r in self.resource_history),
                    'min_available_memory_gb': min(r['system_memory_available_gb'] for r in self.resource_history)
                }
                serializable_state['resource_monitoring_summary'] = resource_summary
            
            # Save to file
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_state, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Pipeline state saved to: {state_file}")
            
        except Exception as e:
            self.logger.warning(f"Could not save pipeline state: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status for monitoring.
        
        Returns:
            Dict[str, Any]: Current pipeline status and metrics
        """
        current_time = datetime.now()
        
        status = {
            'status': self.pipeline_state['status'],
            'current_step': self.pipeline_state.get('current_step'),
            'completed_steps': len(self.pipeline_state['completed_steps']),
            'failed_steps': len(self.pipeline_state['failed_steps']),
            'total_steps': 6,
            'start_time': self.pipeline_state.get('start_time'),
            'current_time': current_time
        }
        
        # Add duration if pipeline has started
        if self.pipeline_state.get('start_time'):
            if self.pipeline_state.get('end_time'):
                status['duration'] = self.pipeline_state['duration']
            else:
                status['duration'] = (current_time - self.pipeline_state['start_time']).total_seconds()
        
        # Add current resource usage
        try:
            process = psutil.Process()
            system_memory = psutil.virtual_memory()
            
            status['current_resources'] = {
                'process_memory_mb': process.memory_info().rss / 1024 / 1024,
                'process_cpu_percent': process.cpu_percent(),
                'system_memory_percent': system_memory.percent,
                'system_memory_available_gb': system_memory.available / 1024 / 1024 / 1024
            }
        except Exception:
            status['current_resources'] = {}
        
        # Add performance metrics summary
        if self.performance_metrics['execution_times']:
            status['performance_summary'] = {
                'total_execution_time': sum(self.performance_metrics['execution_times'].values()),
                'average_step_time': sum(self.performance_metrics['execution_times'].values()) / len(self.performance_metrics['execution_times']),
                'peak_memory_usage': max(self.performance_metrics['memory_usage'].values()) if self.performance_metrics['memory_usage'] else 0
            }
        
        return status
    
    def create_status_report(self, output_path: Optional[str] = None) -> str:
        """
        Create a detailed status report file.
        
        Args:
            output_path (str, optional): Path for the status report file
            
        Returns:
            str: Path to the created status report
        """
        if output_path is None:
            reports_dir = Path(self.config['paths']['reports'])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = reports_dir / f"pipeline_status_report_{timestamp}.txt"
        
        status = self.get_pipeline_status()
        
        report_lines = [
            "=" * 80,
            "SPANISH INFLATION PREDICTION PIPELINE - STATUS REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "PIPELINE STATUS:",
            f"  Status: {status['status'].upper()}",
            f"  Current Step: {status.get('current_step', 'N/A')}",
            f"  Progress: {status['completed_steps']}/{status['total_steps']} steps completed",
            f"  Duration: {status.get('duration', 0):.2f} seconds",
            "",
            "STEP DETAILS:",
        ]
        
        # Add completed steps
        if self.pipeline_state['completed_steps']:
            report_lines.append("  Completed Steps:")
            for step in self.pipeline_state['completed_steps']:
                execution_time = self.performance_metrics['execution_times'].get(step, 0)
                memory_usage = self.performance_metrics['memory_usage'].get(step, 0)
                report_lines.append(f"    ✓ {step} ({execution_time:.2f}s, {memory_usage:.1f}MB)")
        
        # Add failed steps
        if self.pipeline_state['failed_steps']:
            report_lines.append("  Failed Steps:")
            for step in self.pipeline_state['failed_steps']:
                report_lines.append(f"    ✗ {step}")
        
        # Add current resources
        current_resources = status.get('current_resources', {})
        if current_resources:
            report_lines.extend([
                "",
                "CURRENT RESOURCE USAGE:",
                f"  Process Memory: {current_resources.get('process_memory_mb', 0):.1f} MB",
                f"  Process CPU: {current_resources.get('process_cpu_percent', 0):.1f}%",
                f"  System Memory: {current_resources.get('system_memory_percent', 0):.1f}%",
                f"  Available Memory: {current_resources.get('system_memory_available_gb', 0):.1f} GB"
            ])
        
        # Add performance summary
        perf_summary = status.get('performance_summary', {})
        if perf_summary:
            report_lines.extend([
                "",
                "PERFORMANCE SUMMARY:",
                f"  Total Execution Time: {perf_summary.get('total_execution_time', 0):.2f} seconds",
                f"  Average Step Time: {perf_summary.get('average_step_time', 0):.2f} seconds",
                f"  Peak Memory Usage: {perf_summary.get('peak_memory_usage', 0):.1f} MB"
            ])
        
        # Add memory optimization summary
        if self.performance_metrics['memory_optimization']:
            total_freed = sum(opt.get('memory_freed_mb', 0) for opt in self.performance_metrics['memory_optimization'].values())
            total_collected = sum(opt.get('objects_collected', 0) for opt in self.performance_metrics['memory_optimization'].values())
            
            report_lines.extend([
                "",
                "MEMORY OPTIMIZATION:",
                f"  Total Memory Freed: {total_freed:.1f} MB",
                f"  Total Objects Collected: {total_collected}"
            ])
        
        report_lines.append("=" * 80)
        
        # Write report to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            self.logger.info(f"Status report created: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error creating status report: {e}")
            raise


def main():
    """Main entry point for the inflation prediction pipeline."""
    try:
        # Create and run pipeline
        pipeline = InflationPredictionPipeline()
        results = pipeline.run_pipeline()
        
        # Exit with appropriate code
        if results['status'] == 'success':
            print("\n🎉 Pipeline completed successfully!")
            sys.exit(0)
        elif results['status'] == 'partial_success':
            print("\n⚠️  Pipeline completed with some failures.")
            sys.exit(1)
        else:
            print("\n❌ Pipeline failed.")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Pipeline crashed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()