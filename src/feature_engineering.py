"""
Feature engineering module for inflation prediction system.
Creates lag features, rolling statistics, seasonal components, and economic indicators.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import yaml

# Optional sklearn imports for feature selection
try:
    from sklearn.feature_selection import mutual_info_regression, SelectFromModel
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from pandas.api.types import is_numeric_dtype
    _SKLEARN_AVAILABLE = True
except Exception:
    # sklearn may not be installed in all environments; fall back gracefully
    _SKLEARN_AVAILABLE = False

class FeatureEngineer:
    """
    Handles feature engineering operations for inflation prediction models.
    
    This class provides methods for creating lag features, rolling statistics,
    seasonal components, and economic indicators from time series data.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the FeatureEngineer with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Default feature parameters from config
        self.default_lags = self.config.get('features', {}).get('lags', [1, 3, 6, 12])
        self.default_windows = self.config.get('features', {}).get('rolling_windows', [3, 6, 12])
        self.seasonal_periods = self.config.get('features', {}).get('seasonal_periods', [12])
        
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
    
    def create_lag_features(self, data: pd.DataFrame, lags: Optional[List[int]] = None, 
                           target_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create lag features for specified columns.
        
        Creates lagged versions of target columns for time series modeling.
        Supports 1, 3, 6, 12 month lags as specified in requirements.
        
        Args:
            data (pd.DataFrame): Input data with time series
            lags (List[int], optional): List of lag periods. Defaults to config values.
            target_columns (List[str], optional): Columns to create lags for. 
                                                If None, uses numeric columns.
        
        Returns:
            pd.DataFrame: Data with added lag features
        """
        if lags is None:
            lags = self.default_lags
            
        # Create a copy to avoid modifying original data
        lagged_data = data.copy()
        
        # Determine target columns
        if target_columns is None:
            # Use numeric columns, prioritizing IPC-related columns
            numeric_cols = lagged_data.select_dtypes(include=[np.number]).columns.tolist()
            ipc_cols = [col for col in numeric_cols if 'ipc' in col.lower()]
            rate_cols = [col for col in numeric_cols if 'rate' in col.lower()]
            
            # Prioritize IPC and rate columns, then other numeric columns
            target_columns = ipc_cols + rate_cols + [col for col in numeric_cols 
                                                   if col not in ipc_cols + rate_cols]
        
        # Create lag features
        for col in target_columns:
            if col in lagged_data.columns and lagged_data[col].dtype in ['float64', 'int64']:
                for lag in lags:
                    lag_col_name = f"{col}_lag_{lag}"
                    lagged_data[lag_col_name] = lagged_data[col].shift(lag)
                    
                self.logger.info(f"Created lag features for column '{col}' with lags: {lags}")
        
        # Validate feature consistency
        self._validate_lag_features(lagged_data, lags, target_columns)
        
        return lagged_data
    
    def create_rolling_features(self, data: pd.DataFrame, windows: Optional[List[int]] = None,
                               target_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create rolling window features (moving averages, std, min, max).
        
        Creates rolling statistics for specified window sizes to capture
        short and medium-term trends in the data.
        
        Args:
            data (pd.DataFrame): Input data with time series
            windows (List[int], optional): List of window sizes. Defaults to config values.
            target_columns (List[str], optional): Columns to create rolling features for.
                                                If None, uses numeric columns.
        
        Returns:
            pd.DataFrame: Data with added rolling features
        """
        if windows is None:
            windows = self.default_windows
            
        # Create a copy to avoid modifying original data
        rolling_data = data.copy()
        
        # Determine target columns
        if target_columns is None:
            # Use numeric columns, prioritizing IPC-related columns
            numeric_cols = rolling_data.select_dtypes(include=[np.number]).columns.tolist()
            ipc_cols = [col for col in numeric_cols if 'ipc' in col.lower()]
            rate_cols = [col for col in numeric_cols if 'rate' in col.lower()]
            
            # Prioritize IPC and rate columns, then other numeric columns
            target_columns = ipc_cols + rate_cols + [col for col in numeric_cols 
                                                   if col not in ipc_cols + rate_cols]
        
        # Create rolling features
        for col in target_columns:
            if col in rolling_data.columns and rolling_data[col].dtype in ['float64', 'int64']:
                for window in windows:
                    # Moving average
                    ma_col_name = f"{col}_ma_{window}"
                    rolling_data[ma_col_name] = rolling_data[col].rolling(
                        window=window, min_periods=1
                    ).mean()
                    
                    # Rolling standard deviation
                    std_col_name = f"{col}_std_{window}"
                    rolling_data[std_col_name] = rolling_data[col].rolling(
                        window=window, min_periods=1
                    ).std()
                    
                    # Rolling minimum
                    min_col_name = f"{col}_min_{window}"
                    rolling_data[min_col_name] = rolling_data[col].rolling(
                        window=window, min_periods=1
                    ).min()
                    
                    # Rolling maximum
                    max_col_name = f"{col}_max_{window}"
                    rolling_data[max_col_name] = rolling_data[col].rolling(
                        window=window, min_periods=1
                    ).max()
                    
                self.logger.info(f"Created rolling features for column '{col}' with windows: {windows}")
        
        return rolling_data
    
    def _validate_lag_features(self, data: pd.DataFrame, lags: List[int], 
                              target_columns: List[str]) -> None:
        """
        Validate consistency of lag features.
        
        Checks that lag features were created correctly and logs any issues.
        
        Args:
            data (pd.DataFrame): Data with lag features
            lags (List[int]): List of lag periods used
            target_columns (List[str]): Original target columns
        """
        validation_issues = []
        
        for col in target_columns:
            if col in data.columns:
                for lag in lags:
                    lag_col_name = f"{col}_lag_{lag}"
                    
                    if lag_col_name not in data.columns:
                        validation_issues.append(f"Missing lag feature: {lag_col_name}")
                        continue
                    
                    # Check if lag feature has expected number of NaN values at the beginning
                    expected_nans = lag
                    actual_nans = data[lag_col_name].head(lag).isnull().sum()
                    
                    if actual_nans != expected_nans:
                        validation_issues.append(
                            f"Lag feature {lag_col_name} has {actual_nans} NaN values, "
                            f"expected {expected_nans}"
                        )
                    
                    # Check if non-NaN values match original column shifted by lag
                    if len(data) > lag:
                        original_value = data[col].iloc[lag]
                        lag_value = data[lag_col_name].iloc[0]
                        
                        if not pd.isna(original_value) and not pd.isna(lag_value):
                            if not np.isclose(original_value, lag_value, rtol=1e-10):
                                validation_issues.append(
                                    f"Lag feature {lag_col_name} values don't match expected shift"
                                )
        
        if validation_issues:
            for issue in validation_issues:
                self.logger.warning(f"Validation issue: {issue}")
        else:
            self.logger.info("All lag features passed validation checks")
    
    def create_seasonal_features(self, data: pd.DataFrame, 
                                date_column: Optional[str] = None) -> pd.DataFrame:
        """
        Create seasonal features for monthly patterns.
        
        Extracts seasonal components including month, quarter, and cyclical
        features to capture seasonal inflation patterns.
        
        Args:
            data (pd.DataFrame): Input data with datetime information
            date_column (str, optional): Name of date column. If None, auto-detects.
        
        Returns:
            pd.DataFrame: Data with added seasonal features
        """
        seasonal_data = data.copy()
        
        # Find date column if not specified
        if date_column is None:
            date_columns = seasonal_data.select_dtypes(include=['datetime64']).columns
            if len(date_columns) == 0:
                # Try to find date-like columns
                potential_date_cols = [col for col in seasonal_data.columns 
                                     if any(keyword in col.lower() 
                                           for keyword in ['fecha', 'date', 'periodo', 'period'])]
                if potential_date_cols:
                    date_column = potential_date_cols[0]
                    # Convert to datetime if not already
                    seasonal_data[date_column] = pd.to_datetime(seasonal_data[date_column])
                else:
                    self.logger.error("No date column found for seasonal feature creation")
                    return seasonal_data
            else:
                date_column = date_columns[0]
        
        # Ensure date column is datetime
        if seasonal_data[date_column].dtype != 'datetime64[ns]':
            seasonal_data[date_column] = pd.to_datetime(seasonal_data[date_column])
        
        # Extract basic seasonal components
        seasonal_data['month'] = seasonal_data[date_column].dt.month
        seasonal_data['quarter'] = seasonal_data[date_column].dt.quarter
        seasonal_data['year'] = seasonal_data[date_column].dt.year
        seasonal_data['day_of_year'] = seasonal_data[date_column].dt.dayofyear
        
        # Create cyclical features for month (sine/cosine encoding)
        seasonal_data['month_sin'] = np.sin(2 * np.pi * seasonal_data['month'] / 12)
        seasonal_data['month_cos'] = np.cos(2 * np.pi * seasonal_data['month'] / 12)
        
        # Create cyclical features for quarter
        seasonal_data['quarter_sin'] = np.sin(2 * np.pi * seasonal_data['quarter'] / 4)
        seasonal_data['quarter_cos'] = np.cos(2 * np.pi * seasonal_data['quarter'] / 4)
        
        # Create seasonal dummy variables for months
        month_dummies = pd.get_dummies(seasonal_data['month'], prefix='month')
        seasonal_data = pd.concat([seasonal_data, month_dummies], axis=1)
        
        # Create seasonal dummy variables for quarters
        quarter_dummies = pd.get_dummies(seasonal_data['quarter'], prefix='quarter')
        seasonal_data = pd.concat([seasonal_data, quarter_dummies], axis=1)
        
        # Create seasonal indicators for specific economic periods
        # Spanish economic calendar considerations
        seasonal_data['is_january'] = (seasonal_data['month'] == 1).astype(int)  # New year effects
        seasonal_data['is_summer'] = seasonal_data['month'].isin([7, 8]).astype(int)  # Summer season
        seasonal_data['is_christmas'] = (seasonal_data['month'] == 12).astype(int)  # Christmas effects
        seasonal_data['is_back_to_school'] = (seasonal_data['month'] == 9).astype(int)  # September effects
        
        self.logger.info(f"Created seasonal features based on date column '{date_column}'")
        return seasonal_data
    
    def create_economic_indicators(self, data: pd.DataFrame, 
                                  target_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create economic indicators for trend analysis.
        
        Generates economic indicators including trend components, volatility measures,
        and momentum indicators for inflation analysis.
        
        Args:
            data (pd.DataFrame): Input data with inflation metrics
            target_columns (List[str], optional): Columns to create indicators for.
                                                If None, uses IPC-related columns.
        
        Returns:
            pd.DataFrame: Data with added economic indicators
        """
        indicators_data = data.copy()
        
        # Determine target columns
        if target_columns is None:
            # Focus on IPC and rate columns
            numeric_cols = indicators_data.select_dtypes(include=[np.number]).columns.tolist()
            ipc_cols = [col for col in numeric_cols if 'ipc' in col.lower()]
            rate_cols = [col for col in numeric_cols if 'rate' in col.lower()]
            target_columns = ipc_cols + rate_cols
            
            if not target_columns:
                # Fallback to all numeric columns
                target_columns = numeric_cols
        
        for col in target_columns:
            if col in indicators_data.columns and indicators_data[col].dtype in ['float64', 'int64']:
                
                # Trend indicators
                # Linear trend (slope over rolling window)
                for window in [6, 12, 24]:
                    trend_col = f"{col}_trend_{window}"
                    indicators_data[trend_col] = indicators_data[col].rolling(
                        window=window, min_periods=2
                    ).apply(self._calculate_trend_slope, raw=False)
                
                # Momentum indicators
                # Rate of change over different periods
                for period in [1, 3, 6, 12]:
                    roc_col = f"{col}_roc_{period}"
                    indicators_data[roc_col] = indicators_data[col].pct_change(periods=period) * 100
                
                # Acceleration (second derivative)
                accel_col = f"{col}_acceleration"
                indicators_data[accel_col] = indicators_data[col].diff().diff()
                
                # Volatility indicators
                # Rolling standard deviation (volatility)
                for window in [3, 6, 12]:
                    vol_col = f"{col}_volatility_{window}"
                    indicators_data[vol_col] = indicators_data[col].rolling(
                        window=window, min_periods=1
                    ).std()
                
                # Coefficient of variation (normalized volatility)
                for window in [6, 12]:
                    cv_col = f"{col}_cv_{window}"
                    rolling_mean = indicators_data[col].rolling(window=window, min_periods=1).mean()
                    rolling_std = indicators_data[col].rolling(window=window, min_periods=1).std()
                    indicators_data[cv_col] = rolling_std / rolling_mean
                
                # Relative position indicators
                # Position within recent range (0-1 scale)
                for window in [6, 12, 24]:
                    pos_col = f"{col}_position_{window}"
                    rolling_min = indicators_data[col].rolling(window=window, min_periods=1).min()
                    rolling_max = indicators_data[col].rolling(window=window, min_periods=1).max()
                    range_size = rolling_max - rolling_min
                    # Avoid division by zero
                    indicators_data[pos_col] = np.where(
                        range_size != 0,
                        (indicators_data[col] - rolling_min) / range_size,
                        0.5  # Middle position when no range
                    )
                
                # Deviation from moving average
                for window in [6, 12]:
                    dev_col = f"{col}_ma_deviation_{window}"
                    ma = indicators_data[col].rolling(window=window, min_periods=1).mean()
                    indicators_data[dev_col] = indicators_data[col] - ma
                    
                    # Normalized deviation (z-score)
                    zscore_col = f"{col}_zscore_{window}"
                    ma_std = indicators_data[col].rolling(window=window, min_periods=1).std()
                    indicators_data[zscore_col] = np.where(
                        ma_std != 0,
                        (indicators_data[col] - ma) / ma_std,
                        0
                    )
                
                self.logger.info(f"Created economic indicators for column '{col}'")
        
        return indicators_data
    
    def _calculate_trend_slope(self, series: pd.Series) -> float:
        """
        Calculate linear trend slope for a time series window.
        
        Args:
            series (pd.Series): Time series data window
            
        Returns:
            float: Slope of linear trend
        """
        if len(series) < 2 or series.isnull().all():
            return np.nan
        
        # Remove NaN values
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return np.nan
        
        # Create x values (time indices)
        x = np.arange(len(clean_series))
        y = clean_series.values
        
        # Calculate slope using least squares
        try:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        except:
            return np.nan
    
    def create_feature_selection_methods(self, data: pd.DataFrame, 
                                       target_column: str,
                                       method: str = 'correlation') -> List[str]:
        """
        Create feature selection methods to identify most relevant features.
        
        Selects features based on correlation, mutual information, or variance.
        
        Args:
            data (pd.DataFrame): Data with features
            target_column (str): Target variable for feature selection
            method (str): Selection method ('correlation', 'variance', 'mutual_info')
        
        Returns:
            List[str]: List of selected feature names
        """
        if target_column not in data.columns:
            self.logger.error(f"Target column '{target_column}' not found in data")
            return []

        # Get feature columns (exclude target and non-numeric columns)
        # Use a robust numeric check to include other numeric dtypes
        feature_columns = [col for col in data.columns 
                          if col != target_column and 
                          is_numeric_dtype(data[col])]

        if not feature_columns:
            self.logger.warning("No numeric feature columns found for selection")
            return []

        selected_features = []

        if method == 'correlation':
            # Select features based on correlation with target
            correlations = data[feature_columns + [target_column]].corr()[target_column].abs()
            correlations = correlations.drop(target_column).sort_values(ascending=False)

            # Select features with correlation > 0.1
            selected_features = correlations[correlations > 0.1].index.tolist()

        elif method == 'variance':
            # Select features with sufficient variance
            variances = data[feature_columns].var()
            # Select features with variance > 1% of maximum variance
            threshold = variances.max() * 0.01
            selected_features = variances[variances > threshold].index.tolist()

        elif method == 'mutual_info':
            if not _SKLEARN_AVAILABLE:
                self.logger.warning("sklearn not available: falling back to correlation method for mutual_info")
                return self.create_feature_selection_methods(data, target_column, 'correlation')

            # Use mutual information regression to score features
            X = data[feature_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
            y = data[target_column].values
            try:
                mi = mutual_info_regression(X, y, discrete_features=False)
                mi_series = pd.Series(mi, index=feature_columns).sort_values(ascending=False)
                # Select features with MI greater than the median
                selected_features = mi_series[mi_series > mi_series.median()].index.tolist()
            except Exception as e:
                self.logger.warning(f"Mutual information computation failed: {e}. Falling back to correlation.")
                return self.create_feature_selection_methods(data, target_column, 'correlation')

        elif method == 'correlation_matrix':
            # Remove multicollinear features above a threshold (default 0.9)
            selected_features = self.select_features_correlation_threshold(data[feature_columns], threshold=0.9)

        elif method == 'model_based':
            if not _SKLEARN_AVAILABLE:
                self.logger.error("sklearn not available: cannot perform model-based selection")
                return feature_columns
            # Use a simple random forest selector
            X = data[feature_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
            y = data[target_column].values
            selected_features = self.select_features_model_based(X, y)

        else:
            self.logger.error(f"Unknown feature selection method: {method}")
            return feature_columns

        self.logger.info(f"Selected {len(selected_features)} features using {method} method")
        return selected_features

    def select_features_correlation_threshold(self, data: pd.DataFrame, threshold: float = 0.9) -> List[str]:
        """
        Drop features that are highly correlated with each other above a threshold.

        Keeps the feature with the highest variance among correlated pairs.
        """
        if data.shape[1] == 0:
            return []

        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        selected = [col for col in data.columns if col not in to_drop]
        self.logger.info(f"Dropped {len(to_drop)} features with correlation above {threshold}")
        return selected

    def select_features_model_based(self, X: pd.DataFrame, y: np.ndarray, 
                                    model: Optional[Any] = None, threshold: Any = 'median') -> List[str]:
        """
        Model-based feature selection using SelectFromModel.

        By default uses a RandomForestRegressor to compute importances.
        """
        if model is None:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Scale features to help some models (optional)
        scaler = StandardScaler()
        try:
            X_scaled = scaler.fit_transform(X)
        except Exception:
            X_scaled = X.values

        try:
            model.fit(X_scaled, y)
            selector = SelectFromModel(model, threshold=threshold, prefit=True)
            mask = selector.get_support()
            selected = list(X.columns[mask])
            self.logger.info(f"Model-based selection chose {len(selected)} features")
            return selected
        except Exception as e:
            self.logger.warning(f"Model-based selection failed: {e}. Returning all features.")
            return list(X.columns)
    
    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary of created features.
        
        Args:
            data (pd.DataFrame): Data with engineered features
            
        Returns:
            Dict[str, Any]: Summary of features created
        """
        summary = {
            'total_features': len(data.columns),
            'feature_types': {},
            'missing_values': {},
            'feature_categories': {
                'lag_features': [],
                'rolling_features': [],
                'seasonal_features': [],
                'economic_indicators': []
            }
        }
        
        # Categorize features
        for col in data.columns:
            if '_lag_' in col:
                summary['feature_categories']['lag_features'].append(col)
            elif any(keyword in col for keyword in ['_ma_', '_std_', '_min_', '_max_']):
                summary['feature_categories']['rolling_features'].append(col)
            elif any(keyword in col for keyword in ['month', 'quarter', 'seasonal', 'sin', 'cos']):
                summary['feature_categories']['seasonal_features'].append(col)
            elif any(keyword in col for keyword in ['_trend_', '_roc_', '_volatility_', '_cv_', 
                                                   '_position_', '_deviation_', '_zscore_', '_acceleration']):
                summary['feature_categories']['economic_indicators'].append(col)
        
        # Feature type analysis
        summary['feature_types'] = {
            'numeric': len(data.select_dtypes(include=[np.number]).columns),
            'categorical': len(data.select_dtypes(include=['object', 'category']).columns),
            'datetime': len(data.select_dtypes(include=['datetime64']).columns)
        }
        
        # Missing values analysis
        missing_counts = data.isnull().sum()
        summary['missing_values'] = {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentage': (missing_counts.sum() / data.size) * 100
        }
        
        # Feature category counts
        for category, features in summary['feature_categories'].items():
            summary[f'{category}_count'] = len(features)
        
        self.logger.info("Generated feature engineering summary")
        return summary