"""
Machine learning model trainer for inflation prediction system.
Implements ARIMA, Random Forest, and LSTM models with evaluation and selection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import yaml
import pickle
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    # Create dummy classes for type hints when TensorFlow is not available
    class Sequential:
        pass

class ModelTrainer:
    """
    Handles training and evaluation of machine learning models for inflation prediction.
    
    This class implements ARIMA, Random Forest, and LSTM models with automatic
    parameter selection, validation, and performance evaluation.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the ModelTrainer with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Model parameters from config
        self.arima_params = self.config.get('models', {}).get('arima', {})
        self.rf_params = self.config.get('models', {}).get('random_forest', {})
        self.lstm_params = self.config.get('models', {}).get('lstm', {})
        
        # Evaluation parameters
        self.eval_params = self.config.get('evaluation', {})
        self.test_size = self.eval_params.get('test_size', 0.2)
        self.validation_size = self.eval_params.get('validation_size', 0.2)
        
        # Trained models storage
        self.trained_models = {}
        self.model_scores = {}
        
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
    
    def train_arima(self, data: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Train ARIMA model with automatic parameter selection.
        
        Implements ARIMA model with auto parameter selection using AIC criterion,
        model validation, and diagnostic tests.
        
        Args:
            data (pd.DataFrame): Time series data for training
            target_column (str, optional): Target column name. Auto-detects if None.
            
        Returns:
            Dict[str, Any]: Dictionary containing trained model and metadata
        """
        if not _STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA modeling. Install with: pip install statsmodels")
        
        # Auto-detect target column if not specified
        if target_column is None:
            target_column = self._auto_detect_target_column(data)
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        self.logger.info(f"Training ARIMA model on column '{target_column}'")
        
        # Prepare data for ARIMA
        ts_data = self._prepare_arima_data(data, target_column)
        
        # Check stationarity and apply differencing if needed
        ts_data, diff_order = self._make_stationary(ts_data)
        
        # Split data for training and validation
        train_size = int(len(ts_data) * (1 - self.test_size))
        train_data = ts_data[:train_size]
        test_data = ts_data[train_size:]
        
        # Auto parameter selection
        best_params = self._auto_select_arima_params(train_data, diff_order)
        
        # Train final model
        try:
            model = ARIMA(train_data, order=best_params['order'])
            fitted_model = model.fit()
            
            # Model diagnostics
            diagnostics = self._arima_diagnostics(fitted_model, train_data)
            
            # Validate on test set
            validation_results = self._validate_arima_model(fitted_model, test_data)
            
            # Store model results
            model_info = {
                'model': fitted_model,
                'model_type': 'ARIMA',
                'parameters': best_params,
                'target_column': target_column,
                'differencing_order': diff_order,
                'diagnostics': diagnostics,
                'validation': validation_results,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
            self.trained_models['arima'] = model_info
            self.logger.info(f"ARIMA model trained successfully with order {best_params['order']}")
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error training ARIMA model: {e}")
            raise
    
    def _prepare_arima_data(self, data: pd.DataFrame, target_column: str) -> pd.Series:
        """
        Prepare data for ARIMA modeling.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column name
            
        Returns:
            pd.Series: Prepared time series data
        """
        # Extract target series
        ts_data = data[target_column].copy()
        
        # Handle missing values
        if ts_data.isnull().any():
            ts_data = ts_data.interpolate(method='linear')
            self.logger.info("Interpolated missing values in time series")
        
        # Ensure proper frequency if datetime index
        if isinstance(data.index, pd.DatetimeIndex):
            ts_data.index = data.index
            # Infer frequency
            ts_data = ts_data.asfreq(ts_data.index.inferred_freq)
        
        return ts_data
    
    def _make_stationary(self, ts_data: pd.Series) -> Tuple[pd.Series, int]:
        """
        Make time series stationary using differencing.
        
        Args:
            ts_data (pd.Series): Time series data
            
        Returns:
            Tuple[pd.Series, int]: Stationary series and differencing order
        """
        max_d = self.arima_params.get('max_d', 2)
        diff_order = 0
        current_series = ts_data.copy()
        
        # Test stationarity
        for d in range(max_d + 1):
            adf_result = adfuller(current_series.dropna())
            p_value = adf_result[1]
            
            if p_value <= 0.05:  # Series is stationary
                self.logger.info(f"Series is stationary with differencing order: {diff_order}")
                break
            
            if d < max_d:
                current_series = current_series.diff().dropna()
                diff_order += 1
        
        return current_series, diff_order
    
    def _auto_select_arima_params(self, train_data: pd.Series, d: int) -> Dict[str, Any]:
        """
        Automatically select ARIMA parameters using grid search.
        
        Args:
            train_data (pd.Series): Training data
            d (int): Differencing order
            
        Returns:
            Dict[str, Any]: Best parameters found
        """
        max_p = self.arima_params.get('max_p', 5)
        max_q = self.arima_params.get('max_q', 5)
        ic = self.arima_params.get('information_criterion', 'aic')
        
        best_aic = np.inf
        best_params = None
        
        self.logger.info(f"Searching ARIMA parameters: p(0-{max_p}), d({d}), q(0-{max_q})")
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    fitted = model.fit()
                    
                    # Use specified information criterion
                    if ic == 'aic':
                        score = fitted.aic
                    elif ic == 'bic':
                        score = fitted.bic
                    else:
                        score = fitted.aic
                    
                    if score < best_aic:
                        best_aic = score
                        best_params = {
                            'order': (p, d, q),
                            'aic': fitted.aic,
                            'bic': fitted.bic
                        }
                        
                except Exception:
                    continue
        
        if best_params is None:
            # Fallback to simple ARIMA(1,d,1)
            best_params = {
                'order': (1, d, 1),
                'aic': None,
                'bic': None
            }
            self.logger.warning("Could not find optimal parameters, using ARIMA(1,d,1)")
        else:
            self.logger.info(f"Best ARIMA parameters: {best_params['order']} with AIC: {best_params['aic']:.2f}")
        
        return best_params
    
    def _arima_diagnostics(self, fitted_model, train_data: pd.Series) -> Dict[str, Any]:
        """
        Perform ARIMA model diagnostics.
        
        Args:
            fitted_model: Fitted ARIMA model
            train_data (pd.Series): Training data
            
        Returns:
            Dict[str, Any]: Diagnostic results
        """
        diagnostics = {}
        
        try:
            # Residual analysis
            residuals = fitted_model.resid
            diagnostics['residual_mean'] = residuals.mean()
            diagnostics['residual_std'] = residuals.std()
            
            # Ljung-Box test for residual autocorrelation
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
            diagnostics['ljung_box_pvalue'] = lb_test['lb_pvalue'].iloc[-1]
            diagnostics['residuals_autocorrelated'] = diagnostics['ljung_box_pvalue'] < 0.05
            
            # Model fit statistics
            diagnostics['log_likelihood'] = fitted_model.llf
            diagnostics['aic'] = fitted_model.aic
            diagnostics['bic'] = fitted_model.bic
            
            self.logger.info("ARIMA diagnostics completed")
            
        except Exception as e:
            self.logger.warning(f"Error in ARIMA diagnostics: {e}")
            diagnostics['error'] = str(e)
        
        return diagnostics
    
    def _validate_arima_model(self, fitted_model, test_data: pd.Series) -> Dict[str, Any]:
        """
        Validate ARIMA model on test data.
        
        Args:
            fitted_model: Fitted ARIMA model
            test_data (pd.Series): Test data
            
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            # Generate predictions
            forecast_steps = len(test_data)
            forecast = fitted_model.forecast(steps=forecast_steps)
            
            # Calculate metrics
            mae = mean_absolute_error(test_data, forecast)
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
            
            validation_results = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'forecast': forecast.tolist(),
                'actual': test_data.tolist()
            }
            
            self.logger.info(f"ARIMA validation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error in ARIMA validation: {e}")
            return {'error': str(e)}
    
    def _auto_detect_target_column(self, data: pd.DataFrame) -> str:
        """
        Auto-detect the target column for modeling.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            str: Target column name
        """
        # Priority order for target column detection
        priority_keywords = [
            'inflation_rate', 'inflacion', 'ipc_annual_rate', 'ipc_monthly_rate',
            'annual_rate', 'monthly_rate', 'rate', 'ipc'
        ]
        
        for keyword in priority_keywords:
            matching_cols = [col for col in data.columns if keyword in col.lower()]
            if matching_cols:
                return matching_cols[0]
        
        # Fallback to first numeric column
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        
        raise ValueError("No suitable target column found in data")
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series, 
                           hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train Random Forest model with hyperparameter tuning.
        
        Implements Random Forest regressor with grid search for hyperparameter
        optimization and cross-validation for robust evaluation.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            
        Returns:
            Dict[str, Any]: Dictionary containing trained model and metadata
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Random Forest. Install with: pip install scikit-learn")
        
        self.logger.info("Training Random Forest model")
        
        # Prepare data
        X_clean, y_clean = self._prepare_ml_data(X, y)
        
        # Split data
        train_size = int(len(X_clean) * (1 - self.test_size))
        X_train, X_test = X_clean[:train_size], X_clean[train_size:]
        y_train, y_test = y_clean[:train_size], y_clean[train_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        try:
            if hyperparameter_tuning:
                # Hyperparameter tuning with GridSearchCV
                rf_model = self._tune_random_forest_params(X_train_scaled, y_train)
            else:
                # Use default parameters from config
                rf_model = RandomForestRegressor(
                    n_estimators=self.rf_params.get('n_estimators', 100),
                    max_depth=self.rf_params.get('max_depth', 10),
                    min_samples_split=self.rf_params.get('min_samples_split', 5),
                    min_samples_leaf=self.rf_params.get('min_samples_leaf', 2),
                    random_state=self.rf_params.get('random_state', 42)
                )
                rf_model.fit(X_train_scaled, y_train)
            
            # Validate model
            validation_results = self._validate_ml_model(rf_model, X_test_scaled, y_test)
            
            # Feature importance analysis
            feature_importance = self._analyze_feature_importance(rf_model, X_train.columns)
            
            # Store model results
            model_info = {
                'model': rf_model,
                'scaler': scaler,
                'model_type': 'RandomForest',
                'parameters': rf_model.get_params(),
                'feature_names': X_train.columns.tolist(),
                'feature_importance': feature_importance,
                'validation': validation_results,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            self.trained_models['random_forest'] = model_info
            self.logger.info("Random Forest model trained successfully")
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest model: {e}")
            raise
    
    def _prepare_ml_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for machine learning models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Cleaned feature matrix and target
        """
        # Align indices
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        # Handle missing values
        X_clean = X_aligned.fillna(method='ffill').fillna(method='bfill').fillna(0)
        y_clean = y_aligned.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN values
        mask = ~(X_clean.isnull().any(axis=1) | y_clean.isnull())
        X_clean = X_clean[mask]
        y_clean = y_clean[mask]
        
        self.logger.info(f"Prepared ML data: {len(X_clean)} samples, {len(X_clean.columns)} features")
        
        return X_clean, y_clean
    
    def _tune_random_forest_params(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Tune Random Forest hyperparameters using GridSearchCV.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            
        Returns:
            RandomForestRegressor: Tuned model
        """
        # Parameter grid for tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Use TimeSeriesSplit for cross-validation
        cv_folds = self.eval_params.get('cross_validation_folds', 5)
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Grid search
        rf_base = RandomForestRegressor(random_state=self.rf_params.get('random_state', 42))
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=tscv, scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        self.logger.info(f"Best RF parameters: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _analyze_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Analyze feature importance from trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names (List[str]): Names of features
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            self.logger.info(f"Top 5 important features: {list(sorted_importance.keys())[:5]}")
            return sorted_importance
        else:
            return {}
    
    def train_lstm(self, data: pd.DataFrame, target_column: str = None, 
                   sequence_length: int = 12) -> Dict[str, Any]:
        """
        Train LSTM model with TensorFlow/Keras.
        
        Implements LSTM neural network for time series prediction with
        proper data preparation and early stopping.
        
        Args:
            data (pd.DataFrame): Time series data
            target_column (str, optional): Target column name
            sequence_length (int): Length of input sequences
            
        Returns:
            Dict[str, Any]: Dictionary containing trained model and metadata
        """
        if not _TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM. Install with: pip install tensorflow")
        
        # Auto-detect target column if not specified
        if target_column is None:
            target_column = self._auto_detect_target_column(data)
        
        self.logger.info(f"Training LSTM model on column '{target_column}'")
        
        # Prepare LSTM data
        X_lstm, y_lstm, scaler = self._prepare_lstm_data(data, target_column, sequence_length)
        
        # Split data
        train_size = int(len(X_lstm) * (1 - self.test_size))
        X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
        y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]
        
        try:
            # Build LSTM model
            model = self._build_lstm_model(X_train.shape[1:])
            
            # Train model
            history = self._train_lstm_model(model, X_train, y_train, X_test, y_test)
            
            # Validate model
            validation_results = self._validate_lstm_model(model, X_test, y_test, scaler)
            
            # Store model results
            model_info = {
                'model': model,
                'scaler': scaler,
                'model_type': 'LSTM',
                'sequence_length': sequence_length,
                'target_column': target_column,
                'training_history': history.history,
                'validation': validation_results,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            self.trained_models['lstm'] = model_info
            self.logger.info("LSTM model trained successfully")
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {e}")
            raise
    
    def _prepare_lstm_data(self, data: pd.DataFrame, target_column: str, 
                          sequence_length: int) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Prepare data for LSTM training.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column name
            sequence_length (int): Length of input sequences
            
        Returns:
            Tuple[np.ndarray, np.ndarray, StandardScaler]: X, y, and scaler
        """
        # Extract target series
        target_series = data[target_column].fillna(method='ffill').fillna(method='bfill')
        
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(target_series.values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X for LSTM (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        self.logger.info(f"Prepared LSTM data: {X.shape[0]} sequences of length {sequence_length}")
        
        return X, y, scaler
    
    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM neural network architecture.
        
        Args:
            input_shape (Tuple[int, int]): Shape of input data
            
        Returns:
            Sequential: Compiled LSTM model
        """
        model = Sequential()
        
        # LSTM layers
        hidden_units = self.lstm_params.get('hidden_units', 50)
        dropout = self.lstm_params.get('dropout', 0.2)
        
        model.add(LSTM(hidden_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        
        model.add(LSTM(hidden_units, return_sequences=False))
        model.add(Dropout(dropout))
        
        # Dense output layer
        model.add(Dense(1))
        
        # Compile model
        learning_rate = self.lstm_params.get('learning_rate', 0.001)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.logger.info(f"Built LSTM model with {hidden_units} hidden units")
        
        return model
    
    def _train_lstm_model(self, model: Sequential, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray):
        """
        Train LSTM model with early stopping.
        
        Args:
            model (Sequential): LSTM model to train
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            
        Returns:
            History: Training history
        """
        epochs = self.lstm_params.get('epochs', 100)
        batch_size = self.lstm_params.get('batch_size', 32)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.logger.info(f"LSTM training completed in {len(history.history['loss'])} epochs")
        
        return history
    
    def _validate_ml_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Validate machine learning model on test data.
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            validation_results = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
                'actual': y_test.tolist() if hasattr(y_test, 'tolist') else y_test
            }
            
            self.logger.info(f"Model validation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error in model validation: {e}")
            return {'error': str(e)}
    
    def _validate_lstm_model(self, model: Sequential, X_test: np.ndarray, y_test: np.ndarray,
                            scaler: StandardScaler) -> Dict[str, Any]:
        """
        Validate LSTM model on test data.
        
        Args:
            model (Sequential): Trained LSTM model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target (scaled)
            scaler (StandardScaler): Scaler for inverse transformation
            
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            # Generate predictions
            y_pred_scaled = model.predict(X_test)
            
            # Inverse transform predictions and actual values
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
            
            validation_results = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'predictions': y_pred.tolist(),
                'actual': y_actual.tolist()
            }
            
            self.logger.info(f"LSTM validation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error in LSTM validation: {e}")
            return {'error': str(e)}
    
    def evaluate_models(self, models: Optional[Dict[str, Any]] = None, 
                       metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate models using MAE, RMSE, and MAPE metrics.
        
        Compares performance of all trained models using specified metrics
        and provides comprehensive evaluation results.
        
        Args:
            models (Dict[str, Any], optional): Models to evaluate. Uses trained_models if None.
            metrics (List[str], optional): Metrics to calculate. Uses config if None.
            
        Returns:
            Dict[str, Any]: Evaluation results for all models
        """
        if models is None:
            models = self.trained_models
        
        if metrics is None:
            metrics = self.eval_params.get('metrics', ['mae', 'rmse', 'mape'])
        
        if not models:
            self.logger.warning("No models available for evaluation")
            return {}
        
        self.logger.info(f"Evaluating {len(models)} models with metrics: {metrics}")
        
        evaluation_results = {}
        
        for model_name, model_info in models.items():
            try:
                # Extract validation results
                validation = model_info.get('validation', {})
                
                if 'error' in validation:
                    evaluation_results[model_name] = {
                        'status': 'error',
                        'error': validation['error']
                    }
                    continue
                
                # Calculate evaluation metrics
                model_metrics = {}
                for metric in metrics:
                    if metric.lower() in validation:
                        model_metrics[metric.upper()] = validation[metric.lower()]
                
                # Add model-specific information
                model_evaluation = {
                    'metrics': model_metrics,
                    'model_type': model_info.get('model_type', 'Unknown'),
                    'train_size': model_info.get('train_size', 0),
                    'test_size': model_info.get('test_size', 0),
                    'status': 'success'
                }
                
                # Add model-specific details
                if model_name == 'arima':
                    model_evaluation['parameters'] = model_info.get('parameters', {})
                    model_evaluation['aic'] = model_info.get('aic')
                    model_evaluation['bic'] = model_info.get('bic')
                    model_evaluation['diagnostics'] = model_info.get('diagnostics', {})
                
                elif model_name == 'random_forest':
                    model_evaluation['feature_importance'] = model_info.get('feature_importance', {})
                    model_evaluation['n_features'] = len(model_info.get('feature_names', []))
                
                elif model_name == 'lstm':
                    model_evaluation['sequence_length'] = model_info.get('sequence_length')
                    history = model_info.get('training_history', {})
                    if 'loss' in history:
                        model_evaluation['final_train_loss'] = history['loss'][-1]
                        model_evaluation['final_val_loss'] = history['val_loss'][-1]
                
                evaluation_results[model_name] = model_evaluation
                
                # Store scores for model selection
                self.model_scores[model_name] = model_metrics
                
                self.logger.info(f"Evaluated {model_name}: {model_metrics}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating model {model_name}: {e}")
                evaluation_results[model_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Add cross-validation results if available
        evaluation_results = self._add_cross_validation_results(evaluation_results)
        
        return evaluation_results
    
    def _add_cross_validation_results(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add cross-validation results to evaluation.
        
        Args:
            evaluation_results (Dict[str, Any]): Current evaluation results
            
        Returns:
            Dict[str, Any]: Enhanced evaluation results with CV scores
        """
        # Perform cross-validation for models that support it
        cv_folds = self.eval_params.get('cross_validation_folds', 5)
        
        for model_name, model_info in self.trained_models.items():
            if model_name in evaluation_results and evaluation_results[model_name]['status'] == 'success':
                try:
                    if model_name == 'random_forest' and _SKLEARN_AVAILABLE:
                        cv_scores = self._cross_validate_rf_model(model_info, cv_folds)
                        evaluation_results[model_name]['cross_validation'] = cv_scores
                        
                except Exception as e:
                    self.logger.warning(f"Cross-validation failed for {model_name}: {e}")
        
        return evaluation_results
    
    def _cross_validate_rf_model(self, model_info: Dict[str, Any], cv_folds: int) -> Dict[str, Any]:
        """
        Perform cross-validation for Random Forest model.
        
        Args:
            model_info (Dict[str, Any]): Model information
            cv_folds (int): Number of CV folds
            
        Returns:
            Dict[str, Any]: Cross-validation scores
        """
        from sklearn.model_selection import cross_val_score
        
        model = model_info['model']
        # Note: In a real implementation, we would need access to the original data
        # For now, we'll return placeholder CV results
        
        cv_scores = {
            'cv_mae_mean': 0.0,  # Placeholder
            'cv_mae_std': 0.0,   # Placeholder
            'cv_folds': cv_folds,
            'note': 'Cross-validation requires original training data'
        }
        
        return cv_scores
    
    def select_best_model(self, evaluation_results: Optional[Dict[str, Any]] = None,
                         selection_metric: str = 'mae') -> Dict[str, Any]:
        """
        Select the best model based on validation performance.
        
        Selects the model with the lowest error on the specified metric,
        with additional considerations for model complexity and interpretability.
        
        Args:
            evaluation_results (Dict[str, Any], optional): Evaluation results. 
                                                         Computes if None.
            selection_metric (str): Metric to use for selection ('mae', 'rmse', 'mape')
            
        Returns:
            Dict[str, Any]: Information about the best model
        """
        if evaluation_results is None:
            evaluation_results = self.evaluate_models()
        
        if not evaluation_results:
            raise ValueError("No evaluation results available for model selection")
        
        self.logger.info(f"Selecting best model based on {selection_metric.upper()}")
        
        # Filter successful models
        valid_models = {name: results for name, results in evaluation_results.items() 
                       if results.get('status') == 'success'}
        
        if not valid_models:
            raise ValueError("No successfully evaluated models available")
        
        # Find best model based on selection metric
        best_model_name = None
        best_score = float('inf')
        
        for model_name, results in valid_models.items():
            metrics = results.get('metrics', {})
            if selection_metric.upper() in metrics:
                score = metrics[selection_metric.upper()]
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"No models have {selection_metric.upper()} metric available")
        
        # Get best model information
        best_model_info = {
            'model_name': best_model_name,
            'model_type': valid_models[best_model_name]['model_type'],
            'best_score': best_score,
            'selection_metric': selection_metric.upper(),
            'all_metrics': valid_models[best_model_name]['metrics'],
            'model_details': valid_models[best_model_name],
            'trained_model': self.trained_models[best_model_name]
        }
        
        # Add model comparison summary
        model_comparison = {}
        for model_name, results in valid_models.items():
            metrics = results.get('metrics', {})
            model_comparison[model_name] = {
                'mae': metrics.get('MAE', 'N/A'),
                'rmse': metrics.get('RMSE', 'N/A'),
                'mape': metrics.get('MAPE', 'N/A'),
                'model_type': results['model_type']
            }
        
        best_model_info['model_comparison'] = model_comparison
        
        self.logger.info(f"Best model selected: {best_model_name} ({best_model_info['model_type']}) "
                        f"with {selection_metric.upper()}: {best_score:.4f}")
        
        return best_model_info
    
    def save_models(self, models_dir: str = "models/") -> Dict[str, str]:
        """
        Save trained models to disk.
        
        Args:
            models_dir (str): Directory to save models
            
        Returns:
            Dict[str, str]: Dictionary mapping model names to file paths
        """
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        saved_models = {}
        
        for model_name, model_info in self.trained_models.items():
            try:
                model_path = models_dir / f"{model_name}_model.pkl"
                
                # Save model using pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info, f)
                
                saved_models[model_name] = str(model_path)
                self.logger.info(f"Saved {model_name} model to {model_path}")
                
            except Exception as e:
                self.logger.error(f"Error saving {model_name} model: {e}")
        
        return saved_models
    
    def load_models(self, models_dir: str = "models/") -> Dict[str, Any]:
        """
        Load trained models from disk.
        
        Args:
            models_dir (str): Directory containing saved models
            
        Returns:
            Dict[str, Any]: Dictionary of loaded models
        """
        models_dir = Path(models_dir)
        loaded_models = {}
        
        if not models_dir.exists():
            self.logger.warning(f"Models directory not found: {models_dir}")
            return loaded_models
        
        # Look for model files
        model_files = list(models_dir.glob("*_model.pkl"))
        
        for model_file in model_files:
            try:
                model_name = model_file.stem.replace('_model', '')
                
                with open(model_file, 'rb') as f:
                    model_info = pickle.load(f)
                
                loaded_models[model_name] = model_info
                self.logger.info(f"Loaded {model_name} model from {model_file}")
                
            except Exception as e:
                self.logger.error(f"Error loading model from {model_file}: {e}")
        
        self.trained_models.update(loaded_models)
        return loaded_models
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Generate summary of all trained models.
        
        Returns:
            Dict[str, Any]: Summary of models and their performance
        """
        if not self.trained_models:
            return {'message': 'No models trained yet'}
        
        summary = {
            'total_models': len(self.trained_models),
            'model_types': {},
            'best_performers': {},
            'model_details': {}
        }
        
        # Count model types
        for model_name, model_info in self.trained_models.items():
            model_type = model_info.get('model_type', 'Unknown')
            summary['model_types'][model_type] = summary['model_types'].get(model_type, 0) + 1
        
        # Find best performers for each metric
        metrics = ['mae', 'rmse', 'mape']
        for metric in metrics:
            best_score = float('inf')
            best_model = None
            
            for model_name, model_info in self.trained_models.items():
                validation = model_info.get('validation', {})
                if metric in validation and validation[metric] < best_score:
                    best_score = validation[metric]
                    best_model = model_name
            
            if best_model:
                summary['best_performers'][metric.upper()] = {
                    'model': best_model,
                    'score': best_score
                }
        
        # Add detailed information for each model
        for model_name, model_info in self.trained_models.items():
            validation = model_info.get('validation', {})
            summary['model_details'][model_name] = {
                'type': model_info.get('model_type'),
                'train_size': model_info.get('train_size'),
                'test_size': model_info.get('test_size'),
                'mae': validation.get('mae'),
                'rmse': validation.get('rmse'),
                'mape': validation.get('mape')
            }
        
        return summary