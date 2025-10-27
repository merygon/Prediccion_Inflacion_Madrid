"""
Prediction and forecasting module for inflation prediction system.
Handles loading trained models and generating future predictions with confidence intervals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from pathlib import Path
import pickle
import yaml
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
try:
    from scipy import stats
    from scipy.stats import norm
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ML imports
try:
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMAResults
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False


class Predictor:
    """
    Handles prediction and forecasting for inflation prediction system.
    
    This class loads trained models and generates future predictions with
    confidence intervals and uncertainty quantification.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the Predictor with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Prediction parameters from config
        self.prediction_params = self.config.get('prediction', {})
        self.horizon_months = self.prediction_params.get('horizon_months', 12)
        self.confidence_level = self.prediction_params.get('confidence_level', 0.95)
        
        # Model storage
        self.loaded_model = None
        self.model_info = None
        self.model_type = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self.logger.info(f"Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}. Using default parameters.")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading config: {e}. Using default parameters.")
            return {}
    
    def load_best_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load the best trained model for prediction.
        
        Loads a previously trained and saved model from disk, including all
        necessary metadata for making predictions.
        
        Args:
            model_path (str): Path to the saved model file
            
        Returns:
            Dict[str, Any]: Loaded model information and metadata
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model format is invalid
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.logger.info(f"Loading model from {model_path}")
            
            # Load model using pickle
            with open(model_path, 'rb') as f:
                model_info = pickle.load(f)
            
            # Validate model structure
            required_keys = ['model', 'model_type']
            for key in required_keys:
                if key not in model_info:
                    raise ValueError(f"Invalid model file: missing '{key}' key")
            
            # Store model information
            self.model_info = model_info
            self.loaded_model = model_info['model']
            self.model_type = model_info['model_type']
            
            # Log model details
            self.logger.info(f"Loaded {self.model_type} model successfully")
            
            # Add model-specific validation
            if self.model_type == 'ARIMA':
                self._validate_arima_model()
            elif self.model_type == 'RandomForest':
                self._validate_rf_model()
            elif self.model_type == 'LSTM':
                self._validate_lstm_model()
            
            return self.model_info
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _validate_arima_model(self) -> None:
        """Validate ARIMA model structure."""
        if not _STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA predictions")
        
        if not hasattr(self.loaded_model, 'forecast'):
            raise ValueError("Invalid ARIMA model: missing forecast method")
        
        self.logger.debug("ARIMA model validation passed")
    
    def _validate_rf_model(self) -> None:
        """Validate Random Forest model structure."""
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Random Forest predictions")
        
        if not hasattr(self.loaded_model, 'predict'):
            raise ValueError("Invalid Random Forest model: missing predict method")
        
        if 'scaler' not in self.model_info:
            self.logger.warning("No scaler found for Random Forest model")
        
        self.logger.debug("Random Forest model validation passed")
    
    def _validate_lstm_model(self) -> None:
        """Validate LSTM model structure."""
        if not _TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM predictions")
        
        if not hasattr(self.loaded_model, 'predict'):
            raise ValueError("Invalid LSTM model: missing predict method")
        
        required_keys = ['scaler', 'sequence_length']
        for key in required_keys:
            if key not in self.model_info:
                raise ValueError(f"Invalid LSTM model: missing '{key}' key")
        
        self.logger.debug("LSTM model validation passed")
    
    def generate_predictions(self, horizon: Optional[int] = None, 
                           input_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate predictions for the specified horizon.
        
        Creates forecasts for the next N periods using the loaded model,
        with appropriate handling for different model types.
        
        Args:
            horizon (int, optional): Number of periods to forecast. Uses config default if None.
            input_data (pd.DataFrame, optional): Input data for prediction. Required for ML models.
            
        Returns:
            pd.DataFrame: DataFrame with predictions and metadata
            
        Raises:
            ValueError: If no model is loaded or input data is invalid
        """
        if self.loaded_model is None:
            raise ValueError("No model loaded. Call load_best_model() first.")
        
        if horizon is None:
            horizon = self.horizon_months
        
        self.logger.info(f"Generating {horizon}-period predictions using {self.model_type} model")
        
        try:
            if self.model_type == 'ARIMA':
                predictions_df = self._generate_arima_predictions(horizon)
            elif self.model_type == 'RandomForest':
                if input_data is None:
                    raise ValueError("Input data required for Random Forest predictions")
                predictions_df = self._generate_rf_predictions(horizon, input_data)
            elif self.model_type == 'LSTM':
                if input_data is None:
                    raise ValueError("Input data required for LSTM predictions")
                predictions_df = self._generate_lstm_predictions(horizon, input_data)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.logger.info(f"Generated {len(predictions_df)} predictions successfully")
            return predictions_df
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            raise
    
    def _generate_arima_predictions(self, horizon: int) -> pd.DataFrame:
        """
        Generate predictions using ARIMA model.
        
        Args:
            horizon (int): Number of periods to forecast
            
        Returns:
            pd.DataFrame: ARIMA predictions with confidence intervals
        """
        try:
            # Generate forecast
            forecast = self.loaded_model.forecast(steps=horizon)
            forecast_ci = self.loaded_model.get_forecast(steps=horizon).conf_int(alpha=1-self.confidence_level)
            
            # Create date index for predictions
            last_date = self.loaded_model.data.dates[-1] if hasattr(self.loaded_model.data, 'dates') else None
            if last_date is not None:
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                           periods=horizon, freq='MS')
            else:
                future_dates = pd.date_range(start='2024-01-01', periods=horizon, freq='MS')
            
            # Create predictions DataFrame
            predictions_df = pd.DataFrame({
                'fecha': future_dates,
                'predicted_inflation': forecast.values,
                'confidence_lower': forecast_ci.iloc[:, 0].values,
                'confidence_upper': forecast_ci.iloc[:, 1].values,
                'model_used': 'ARIMA',
                'confidence_level': self.confidence_level
            })
            
            return predictions_df
            
        except Exception as e:
            self.logger.error(f"Error in ARIMA prediction: {e}")
            raise
    
    def _generate_rf_predictions(self, horizon: int, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using Random Forest model.
        
        Args:
            horizon (int): Number of periods to forecast
            input_data (pd.DataFrame): Input features for prediction
            
        Returns:
            pd.DataFrame: Random Forest predictions with confidence intervals
        """
        try:
            # Prepare features for prediction
            feature_names = self.model_info.get('feature_names', [])
            if not all(col in input_data.columns for col in feature_names):
                missing_cols = [col for col in feature_names if col not in input_data.columns]
                raise ValueError(f"Missing required features: {missing_cols}")
            
            # Get the most recent data for iterative prediction
            X_features = input_data[feature_names].tail(horizon).copy()
            
            # Scale features if scaler is available
            scaler = self.model_info.get('scaler')
            if scaler is not None:
                X_scaled = scaler.transform(X_features)
            else:
                X_scaled = X_features.values
            
            # Generate predictions
            predictions = self.loaded_model.predict(X_scaled)
            
            # Calculate confidence intervals using prediction intervals
            confidence_intervals = self._calculate_rf_confidence_intervals(X_scaled, predictions)
            
            # Create date index
            if isinstance(input_data.index, pd.DatetimeIndex):
                last_date = input_data.index[-1]
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                           periods=horizon, freq='MS')
            else:
                future_dates = pd.date_range(start='2024-01-01', periods=horizon, freq='MS')
            
            # Create predictions DataFrame
            predictions_df = pd.DataFrame({
                'fecha': future_dates,
                'predicted_inflation': predictions,
                'confidence_lower': confidence_intervals['lower'],
                'confidence_upper': confidence_intervals['upper'],
                'model_used': 'RandomForest',
                'confidence_level': self.confidence_level
            })
            
            return predictions_df
            
        except Exception as e:
            self.logger.error(f"Error in Random Forest prediction: {e}")
            raise
    
    def _generate_lstm_predictions(self, horizon: int, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using LSTM model.
        
        Args:
            horizon (int): Number of periods to forecast
            input_data (pd.DataFrame): Input time series data
            
        Returns:
            pd.DataFrame: LSTM predictions with confidence intervals
        """
        try:
            sequence_length = self.model_info['sequence_length']
            target_column = self.model_info['target_column']
            scaler = self.model_info['scaler']
            
            if target_column not in input_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in input data")
            
            # Prepare input sequence
            target_series = input_data[target_column].fillna(method='ffill').fillna(method='bfill')
            
            # Scale the input data
            scaled_data = scaler.transform(target_series.values.reshape(-1, 1))
            
            # Get the last sequence for prediction
            if len(scaled_data) < sequence_length:
                raise ValueError(f"Input data too short. Need at least {sequence_length} points.")
            
            last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
            
            # Generate iterative predictions
            predictions_scaled = []
            current_sequence = last_sequence.copy()
            
            for _ in range(horizon):
                # Predict next value
                next_pred = self.loaded_model.predict(current_sequence, verbose=0)
                predictions_scaled.append(next_pred[0, 0])
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_pred[0, 0]
            
            # Inverse transform predictions
            predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions_scaled).flatten()
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_lstm_confidence_intervals(predictions)
            
            # Create date index
            if isinstance(input_data.index, pd.DatetimeIndex):
                last_date = input_data.index[-1]
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                           periods=horizon, freq='MS')
            else:
                future_dates = pd.date_range(start='2024-01-01', periods=horizon, freq='MS')
            
            # Create predictions DataFrame
            predictions_df = pd.DataFrame({
                'fecha': future_dates,
                'predicted_inflation': predictions,
                'confidence_lower': confidence_intervals['lower'],
                'confidence_upper': confidence_intervals['upper'],
                'model_used': 'LSTM',
                'confidence_level': self.confidence_level
            })
            
            return predictions_df
            
        except Exception as e:
            self.logger.error(f"Error in LSTM prediction: {e}")
            raise
    
    def _calculate_rf_confidence_intervals(self, X: np.ndarray, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate confidence intervals for Random Forest predictions.
        
        Args:
            X (np.ndarray): Input features
            predictions (np.ndarray): Point predictions
            
        Returns:
            Dict[str, np.ndarray]: Lower and upper confidence bounds
        """
        try:
            # Use prediction intervals from individual trees
            if hasattr(self.loaded_model, 'estimators_'):
                # Get predictions from all trees
                tree_predictions = np.array([tree.predict(X) for tree in self.loaded_model.estimators_])
                
                # Calculate percentiles for confidence intervals
                alpha = 1 - self.confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                confidence_lower = np.percentile(tree_predictions, lower_percentile, axis=0)
                confidence_upper = np.percentile(tree_predictions, upper_percentile, axis=0)
            else:
                # Fallback: use standard deviation-based intervals
                std_dev = np.std(predictions) if len(predictions) > 1 else 0.1
                z_score = norm.ppf(1 - (1 - self.confidence_level) / 2) if _SCIPY_AVAILABLE else 1.96
                
                confidence_lower = predictions - z_score * std_dev
                confidence_upper = predictions + z_score * std_dev
            
            return {
                'lower': confidence_lower,
                'upper': confidence_upper
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating RF confidence intervals: {e}")
            # Fallback to simple intervals
            std_dev = 0.1
            return {
                'lower': predictions - std_dev,
                'upper': predictions + std_dev
            }
    
    def _calculate_lstm_confidence_intervals(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate confidence intervals for LSTM predictions.
        
        Args:
            predictions (np.ndarray): Point predictions
            
        Returns:
            Dict[str, np.ndarray]: Lower and upper confidence bounds
        """
        try:
            # Use Monte Carlo dropout for uncertainty estimation
            # For now, use a simple approach based on prediction variance
            
            # Estimate uncertainty based on model validation performance
            validation_results = self.model_info.get('validation', {})
            rmse = validation_results.get('rmse', 0.1)
            
            # Calculate confidence intervals using RMSE as uncertainty measure
            z_score = norm.ppf(1 - (1 - self.confidence_level) / 2) if _SCIPY_AVAILABLE else 1.96
            margin_of_error = z_score * rmse
            
            confidence_lower = predictions - margin_of_error
            confidence_upper = predictions + margin_of_error
            
            return {
                'lower': confidence_lower,
                'upper': confidence_upper
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating LSTM confidence intervals: {e}")
            # Fallback to simple intervals
            margin = 0.1
            return {
                'lower': predictions - margin,
                'upper': predictions + margin
            }
    
    def calculate_confidence_intervals(self, predictions: pd.DataFrame, 
                                     method: str = 'auto') -> pd.DataFrame:
        """
        Calculate or recalculate confidence intervals for predictions.
        
        Provides uncertainty quantification for predictions using various methods
        depending on the model type and available information.
        
        Args:
            predictions (pd.DataFrame): DataFrame with predictions
            method (str): Method for calculating intervals ('auto', 'bootstrap', 'normal')
            
        Returns:
            pd.DataFrame: Predictions with updated confidence intervals
        """
        if 'predicted_inflation' not in predictions.columns:
            raise ValueError("Predictions DataFrame must contain 'predicted_inflation' column")
        
        self.logger.info(f"Calculating confidence intervals using method: {method}")
        
        try:
            pred_values = predictions['predicted_inflation'].values
            
            if method == 'auto':
                # Use model-specific method
                if self.model_type == 'ARIMA':
                    # ARIMA already provides confidence intervals
                    return predictions
                elif self.model_type == 'RandomForest':
                    intervals = self._calculate_bootstrap_intervals(pred_values)
                elif self.model_type == 'LSTM':
                    intervals = self._calculate_normal_intervals(pred_values)
                else:
                    intervals = self._calculate_normal_intervals(pred_values)
            
            elif method == 'bootstrap':
                intervals = self._calculate_bootstrap_intervals(pred_values)
            
            elif method == 'normal':
                intervals = self._calculate_normal_intervals(pred_values)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Update confidence intervals
            predictions = predictions.copy()
            predictions['confidence_lower'] = intervals['lower']
            predictions['confidence_upper'] = intervals['upper']
            predictions['confidence_level'] = self.confidence_level
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence intervals: {e}")
            raise
    
    def _calculate_bootstrap_intervals(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate confidence intervals using bootstrap method.
        
        Args:
            predictions (np.ndarray): Point predictions
            
        Returns:
            Dict[str, np.ndarray]: Lower and upper confidence bounds
        """
        # Simple bootstrap-like approach using prediction variance
        if len(predictions) > 1:
            std_dev = np.std(predictions)
        else:
            # Use model validation RMSE if available
            validation_results = self.model_info.get('validation', {}) if self.model_info else {}
            std_dev = validation_results.get('rmse', 0.1)
        
        z_score = norm.ppf(1 - (1 - self.confidence_level) / 2) if _SCIPY_AVAILABLE else 1.96
        margin_of_error = z_score * std_dev
        
        return {
            'lower': predictions - margin_of_error,
            'upper': predictions + margin_of_error
        }
    
    def _calculate_normal_intervals(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate confidence intervals assuming normal distribution.
        
        Args:
            predictions (np.ndarray): Point predictions
            
        Returns:
            Dict[str, np.ndarray]: Lower and upper confidence bounds
        """
        # Use model validation performance for uncertainty estimation
        validation_results = self.model_info.get('validation', {}) if self.model_info else {}
        std_error = validation_results.get('rmse', 0.1)
        
        z_score = norm.ppf(1 - (1 - self.confidence_level) / 2) if _SCIPY_AVAILABLE else 1.96
        margin_of_error = z_score * std_error
        
        return {
            'lower': predictions - margin_of_error,
            'upper': predictions + margin_of_error
        }
    
    def get_prediction_summary(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for predictions.
        
        Args:
            predictions (pd.DataFrame): DataFrame with predictions
            
        Returns:
            Dict[str, Any]: Summary statistics and metadata
        """
        if predictions.empty:
            return {'message': 'No predictions available'}
        
        try:
            pred_values = predictions['predicted_inflation'].values
            
            summary = {
                'prediction_count': len(predictions),
                'prediction_period': {
                    'start': predictions['fecha'].min().strftime('%Y-%m-%d') if 'fecha' in predictions.columns else 'N/A',
                    'end': predictions['fecha'].max().strftime('%Y-%m-%d') if 'fecha' in predictions.columns else 'N/A'
                },
                'statistics': {
                    'mean_prediction': float(np.mean(pred_values)),
                    'median_prediction': float(np.median(pred_values)),
                    'std_prediction': float(np.std(pred_values)),
                    'min_prediction': float(np.min(pred_values)),
                    'max_prediction': float(np.max(pred_values))
                },
                'model_info': {
                    'model_type': self.model_type,
                    'confidence_level': self.confidence_level
                }
            }
            
            # Add confidence interval statistics if available
            if 'confidence_lower' in predictions.columns and 'confidence_upper' in predictions.columns:
                ci_width = predictions['confidence_upper'] - predictions['confidence_lower']
                summary['confidence_intervals'] = {
                    'mean_width': float(np.mean(ci_width)),
                    'median_width': float(np.median(ci_width)),
                    'min_width': float(np.min(ci_width)),
                    'max_width': float(np.max(ci_width))
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating prediction summary: {e}")
            return {'error': str(e)}
    
    def load_model_by_name(self, model_name: str, models_dir: str = "models/") -> Dict[str, Any]:
        """
        Load a specific model by name from the models directory.
        
        Args:
            model_name (str): Name of the model to load
            models_dir (str): Directory containing saved models
            
        Returns:
            Dict[str, Any]: Loaded model information
        """
        models_dir = Path(models_dir)
        model_file = models_dir / f"{model_name}_model.pkl"
        
        return self.load_best_model(str(model_file))
    
    def get_available_models(self, models_dir: str = "models/") -> List[str]:
        """
        Get list of available saved models.
        
        Args:
            models_dir (str): Directory containing saved models
            
        Returns:
            List[str]: List of available model names
        """
        models_dir = Path(models_dir)
        
        if not models_dir.exists():
            self.logger.warning(f"Models directory not found: {models_dir}")
            return []
        
        model_files = list(models_dir.glob("*_model.pkl"))
        model_names = [f.stem.replace('_model', '') for f in model_files]
        
        self.logger.info(f"Found {len(model_names)} available models: {model_names}")
        return model_names
    
    def validate_predictions(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate predictions for sanity checks.
        
        Performs various sanity checks on predictions to ensure they are
        reasonable and within expected bounds for inflation data.
        
        Args:
            predictions (pd.DataFrame): DataFrame with predictions to validate
            
        Returns:
            Dict[str, Any]: Validation results with checks and warnings
        """
        if predictions.empty:
            return {
                'valid': False,
                'error': 'No predictions to validate'
            }
        
        self.logger.info("Validating predictions for sanity checks")
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'checks': {}
        }
        
        try:
            pred_values = predictions['predicted_inflation'].values
            
            # Check 1: No NaN or infinite values
            if np.any(np.isnan(pred_values)) or np.any(np.isinf(pred_values)):
                validation_results['errors'].append("Predictions contain NaN or infinite values")
                validation_results['valid'] = False
            
            validation_results['checks']['no_invalid_values'] = not (np.any(np.isnan(pred_values)) or np.any(np.isinf(pred_values)))
            
            # Check 2: Reasonable inflation range (-10% to +20%)
            reasonable_min, reasonable_max = -10.0, 20.0
            out_of_range = np.any((pred_values < reasonable_min) | (pred_values > reasonable_max))
            
            if out_of_range:
                extreme_values = pred_values[(pred_values < reasonable_min) | (pred_values > reasonable_max)]
                validation_results['warnings'].append(
                    f"Some predictions outside reasonable range ({reasonable_min}% to {reasonable_max}%): "
                    f"min={np.min(extreme_values):.2f}%, max={np.max(extreme_values):.2f}%"
                )
            
            validation_results['checks']['reasonable_range'] = not out_of_range
            
            # Check 3: Volatility check (standard deviation should be reasonable)
            pred_std = np.std(pred_values)
            max_reasonable_std = 5.0  # 5% standard deviation seems reasonable for inflation
            
            if pred_std > max_reasonable_std:
                validation_results['warnings'].append(
                    f"High prediction volatility detected (std={pred_std:.2f}%). "
                    f"Consider reviewing model stability."
                )
            
            validation_results['checks']['reasonable_volatility'] = pred_std <= max_reasonable_std
            
            # Check 4: Trend consistency (no extreme jumps between consecutive predictions)
            if len(pred_values) > 1:
                differences = np.diff(pred_values)
                max_jump = np.max(np.abs(differences))
                reasonable_max_jump = 3.0  # 3% jump between consecutive months
                
                if max_jump > reasonable_max_jump:
                    validation_results['warnings'].append(
                        f"Large jump detected between consecutive predictions (max={max_jump:.2f}%). "
                        f"Consider smoothing or model review."
                    )
                
                validation_results['checks']['no_extreme_jumps'] = max_jump <= reasonable_max_jump
            else:
                validation_results['checks']['no_extreme_jumps'] = True
            
            # Check 5: Confidence intervals validation
            if 'confidence_lower' in predictions.columns and 'confidence_upper' in predictions.columns:
                ci_lower = predictions['confidence_lower'].values
                ci_upper = predictions['confidence_upper'].values
                
                # Check that lower < prediction < upper
                ci_valid = np.all((ci_lower <= pred_values) & (pred_values <= ci_upper))
                if not ci_valid:
                    validation_results['errors'].append("Predictions not within confidence intervals")
                    validation_results['valid'] = False
                
                # Check reasonable confidence interval widths
                ci_widths = ci_upper - ci_lower
                mean_ci_width = np.mean(ci_widths)
                if mean_ci_width > 10.0:  # Very wide intervals
                    validation_results['warnings'].append(
                        f"Very wide confidence intervals detected (mean width={mean_ci_width:.2f}%). "
                        f"High prediction uncertainty."
                    )
                
                validation_results['checks']['valid_confidence_intervals'] = ci_valid
                validation_results['checks']['reasonable_ci_width'] = mean_ci_width <= 10.0
            
            # Check 6: Date sequence validation
            if 'fecha' in predictions.columns:
                dates = pd.to_datetime(predictions['fecha'])
                if not dates.is_monotonic_increasing:
                    validation_results['errors'].append("Prediction dates are not in chronological order")
                    validation_results['valid'] = False
                
                # Check for reasonable date gaps (should be monthly)
                if len(dates) > 1:
                    date_diffs = dates.diff().dropna()
                    expected_diff = pd.Timedelta(days=30)  # Approximately monthly
                    irregular_gaps = date_diffs[(date_diffs < pd.Timedelta(days=25)) | 
                                               (date_diffs > pd.Timedelta(days=35))]
                    
                    if len(irregular_gaps) > 0:
                        validation_results['warnings'].append(
                            f"Irregular date gaps detected in {len(irregular_gaps)} places"
                        )
                
                validation_results['checks']['chronological_dates'] = dates.is_monotonic_increasing
            
            # Summary statistics for validation report
            validation_results['statistics'] = {
                'prediction_count': len(pred_values),
                'mean_prediction': float(np.mean(pred_values)),
                'std_prediction': float(np.std(pred_values)),
                'min_prediction': float(np.min(pred_values)),
                'max_prediction': float(np.max(pred_values)),
                'range': float(np.max(pred_values) - np.min(pred_values))
            }
            
            # Overall validation status
            total_checks = len(validation_results['checks'])
            passed_checks = sum(validation_results['checks'].values())
            validation_results['check_summary'] = {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'pass_rate': passed_checks / total_checks if total_checks > 0 else 0
            }
            
            self.logger.info(f"Validation completed: {passed_checks}/{total_checks} checks passed, "
                           f"{len(validation_results['warnings'])} warnings, "
                           f"{len(validation_results['errors'])} errors")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error during prediction validation: {e}")
            return {
                'valid': False,
                'error': str(e),
                'checks': {},
                'warnings': [],
                'errors': [f"Validation failed: {str(e)}"]
            }
    
    def export_predictions_csv(self, predictions: pd.DataFrame, 
                              filepath: str, include_metadata: bool = True) -> str:
        """
        Export predictions to CSV format.
        
        Args:
            predictions (pd.DataFrame): Predictions to export
            filepath (str): Output file path
            include_metadata (bool): Whether to include metadata in the file
            
        Returns:
            str: Path to the exported file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare data for export
            export_data = predictions.copy()
            
            # Format dates if present
            if 'fecha' in export_data.columns:
                export_data['fecha'] = pd.to_datetime(export_data['fecha']).dt.strftime('%Y-%m-%d')
            
            # Round numeric columns
            numeric_columns = export_data.select_dtypes(include=[np.number]).columns
            decimal_places = self.config.get('output', {}).get('decimal_places', 4)
            export_data[numeric_columns] = export_data[numeric_columns].round(decimal_places)
            
            # Export to CSV
            encoding = self.config.get('output', {}).get('csv_encoding', 'utf-8')
            export_data.to_csv(filepath, index=False, encoding=encoding)
            
            # Add metadata as comments if requested
            if include_metadata:
                self._add_csv_metadata(filepath, predictions)
            
            self.logger.info(f"Predictions exported to CSV: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error exporting predictions to CSV: {e}")
            raise
    
    def export_predictions_json(self, predictions: pd.DataFrame, 
                               filepath: str, include_metadata: bool = True) -> str:
        """
        Export predictions to JSON format.
        
        Args:
            predictions (pd.DataFrame): Predictions to export
            filepath (str): Output file path
            include_metadata (bool): Whether to include metadata in the JSON
            
        Returns:
            str: Path to the exported file
        """
        import json
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert DataFrame to dictionary
            export_data = predictions.copy()
            
            # Convert dates to strings
            if 'fecha' in export_data.columns:
                export_data['fecha'] = pd.to_datetime(export_data['fecha']).dt.strftime('%Y-%m-%d')
            
            # Round numeric values
            numeric_columns = export_data.select_dtypes(include=[np.number]).columns
            decimal_places = self.config.get('output', {}).get('decimal_places', 4)
            export_data[numeric_columns] = export_data[numeric_columns].round(decimal_places)
            
            # Create JSON structure
            json_data = {
                'predictions': export_data.to_dict('records')
            }
            
            # Add metadata if requested
            if include_metadata:
                json_data['metadata'] = {
                    'model_type': self.model_type,
                    'prediction_count': len(predictions),
                    'confidence_level': self.confidence_level,
                    'export_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'summary': self.get_prediction_summary(predictions)
                }
            
            # Export to JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Predictions exported to JSON: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error exporting predictions to JSON: {e}")
            raise
    
    def _add_csv_metadata(self, filepath: Path, predictions: pd.DataFrame) -> None:
        """
        Add metadata as comments to CSV file.
        
        Args:
            filepath (Path): Path to CSV file
            predictions (pd.DataFrame): Original predictions data
        """
        try:
            # Read existing content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create metadata header
            metadata_lines = [
                f"# Inflation Predictions Export",
                f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"# Model Type: {self.model_type}",
                f"# Prediction Count: {len(predictions)}",
                f"# Confidence Level: {self.confidence_level}",
                f"#"
            ]
            
            # Write metadata + content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(metadata_lines) + '\n')
                f.write(content)
                
        except Exception as e:
            self.logger.warning(f"Could not add metadata to CSV: {e}")
    
    def create_prediction_visualizations(self, predictions: pd.DataFrame, 
                                       historical_data: Optional[pd.DataFrame] = None,
                                       output_dir: str = "reports/") -> Dict[str, str]:
        """
        Create visualizations for predictions.
        
        Args:
            predictions (pd.DataFrame): Predictions to visualize
            historical_data (pd.DataFrame, optional): Historical data for context
            output_dir (str): Directory to save visualizations
            
        Returns:
            Dict[str, str]: Dictionary mapping plot types to file paths
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.figure import Figure
        except ImportError:
            self.logger.error("matplotlib is required for visualizations. Install with: pip install matplotlib")
            raise ImportError("matplotlib is required for creating visualizations")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        chart_style = self.config.get('reports', {}).get('chart_style', 'seaborn-v0_8')
        try:
            plt.style.use(chart_style)
        except:
            plt.style.use('default')
        
        chart_dpi = self.config.get('reports', {}).get('chart_dpi', 300)
        created_plots = {}
        
        try:
            # Plot 1: Predictions with confidence intervals
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if 'fecha' in predictions.columns:
                dates = pd.to_datetime(predictions['fecha'])
                ax.plot(dates, predictions['predicted_inflation'], 
                       label='Predicted Inflation', linewidth=2, color='blue')
                
                # Add confidence intervals if available
                if 'confidence_lower' in predictions.columns and 'confidence_upper' in predictions.columns:
                    ax.fill_between(dates, 
                                   predictions['confidence_lower'], 
                                   predictions['confidence_upper'],
                                   alpha=0.3, color='blue', 
                                   label=f'{int(self.confidence_level*100)}% Confidence Interval')
                
                # Add historical data if provided
                if historical_data is not None and not historical_data.empty:
                    # Try to find inflation rate column
                    hist_col = None
                    for col in ['inflation_rate_annual', 'inflacion_anual', 'ipc_annual_rate', 'annual_rate']:
                        if col in historical_data.columns:
                            hist_col = col
                            break
                    
                    if hist_col and isinstance(historical_data.index, pd.DatetimeIndex):
                        ax.plot(historical_data.index, historical_data[hist_col], 
                               label='Historical Inflation', linewidth=1, color='gray', alpha=0.7)
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Inflation Rate (%)')
                ax.set_title(f'Inflation Predictions - {self.model_type} Model')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Format x-axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plot_path = output_dir / f"predictions_{self.model_type.lower()}.png"
            plt.savefig(plot_path, dpi=chart_dpi, bbox_inches='tight')
            plt.close()
            created_plots['predictions'] = str(plot_path)
            
            # Plot 2: Prediction distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(predictions['predicted_inflation'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Predicted Inflation Rate (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Predicted Inflation Rates')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_pred = predictions['predicted_inflation'].mean()
            std_pred = predictions['predicted_inflation'].std()
            ax.axvline(mean_pred, color='red', linestyle='--', label=f'Mean: {mean_pred:.2f}%')
            ax.axvline(mean_pred + std_pred, color='orange', linestyle=':', alpha=0.7, label=f'+1 Std: {mean_pred + std_pred:.2f}%')
            ax.axvline(mean_pred - std_pred, color='orange', linestyle=':', alpha=0.7, label=f'-1 Std: {mean_pred - std_pred:.2f}%')
            ax.legend()
            
            plt.tight_layout()
            plot_path = output_dir / f"prediction_distribution_{self.model_type.lower()}.png"
            plt.savefig(plot_path, dpi=chart_dpi, bbox_inches='tight')
            plt.close()
            created_plots['distribution'] = str(plot_path)
            
            # Plot 3: Confidence interval width over time
            if 'confidence_lower' in predictions.columns and 'confidence_upper' in predictions.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ci_width = predictions['confidence_upper'] - predictions['confidence_lower']
                if 'fecha' in predictions.columns:
                    dates = pd.to_datetime(predictions['fecha'])
                    ax.plot(dates, ci_width, linewidth=2, color='green')
                    ax.set_xlabel('Date')
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                    plt.xticks(rotation=45)
                else:
                    ax.plot(ci_width, linewidth=2, color='green')
                    ax.set_xlabel('Prediction Period')
                
                ax.set_ylabel('Confidence Interval Width (%)')
                ax.set_title('Prediction Uncertainty Over Time')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = output_dir / f"uncertainty_{self.model_type.lower()}.png"
                plt.savefig(plot_path, dpi=chart_dpi, bbox_inches='tight')
                plt.close()
                created_plots['uncertainty'] = str(plot_path)
            
            self.logger.info(f"Created {len(created_plots)} visualization plots in {output_dir}")
            return created_plots
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            raise
    
    def export_prediction_report(self, predictions: pd.DataFrame, 
                               validation_results: Optional[Dict[str, Any]] = None,
                               output_path: str = "reports/prediction_report.json") -> str:
        """
        Export comprehensive prediction report.
        
        Args:
            predictions (pd.DataFrame): Predictions data
            validation_results (Dict[str, Any], optional): Validation results
            output_path (str): Path for the report file
            
        Returns:
            str: Path to the exported report
        """
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Generate comprehensive report
            report = {
                'report_metadata': {
                    'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_type': self.model_type,
                    'confidence_level': self.confidence_level,
                    'prediction_horizon': len(predictions)
                },
                'prediction_summary': self.get_prediction_summary(predictions),
                'predictions': predictions.round(4).to_dict('records') if not predictions.empty else [],
                'validation_results': validation_results or {},
                'model_information': {
                    'model_type': self.model_type,
                    'loaded_model_info': {
                        key: value for key, value in (self.model_info or {}).items() 
                        if key not in ['model', 'scaler']  # Exclude non-serializable objects
                    }
                }
            }
            
            # Convert dates to strings in predictions
            for pred in report['predictions']:
                if 'fecha' in pred and pred['fecha']:
                    pred['fecha'] = pd.to_datetime(pred['fecha']).strftime('%Y-%m-%d')
            
            # Export report
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Prediction report exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting prediction report: {e}")
            raise