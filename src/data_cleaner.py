"""
Data cleaning and processing module for inflation prediction system.
Handles CSV file reading, missing values, outlier detection, and data validation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from pathlib import Path

class DataProcessor:
    """
    Handles data cleaning and processing operations for inflation data.
    
    This class provides methods for loading raw data, handling missing values,
    detecting outliers, and preparing data for analysis.
    """
    
    def __init__(self):
        """Initialize the DataProcessor with logging configuration."""
        self.logger = logging.getLogger(__name__)
        
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw data from CSV file with proper error handling.
        
        Args:
            filepath (str): Path to the CSV file to load
            
        Returns:
            pd.DataFrame: Loaded data with proper column types
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
            pd.errors.ParserError: If the file format is invalid
        """
        try:
            # Check if file exists
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            # Load CSV with proper settings for Spanish data
            data = pd.read_csv(
                filepath,
                encoding='utf-8',
                sep=',',
                decimal='.'
            )
            
            # Validate that data is not empty
            if data.empty:
                raise pd.errors.EmptyDataError(f"No data found in file: {filepath}")
            
            self.logger.info(f"Successfully loaded {len(data)} rows from {filepath}")
            return data
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            self.logger.error(f"Empty data file: {e}")
            raise
        except pd.errors.ParserError as e:
            self.logger.error(f"Error parsing CSV file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading data: {e}")
            raise
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using interpolation techniques.
        
        Uses linear interpolation for numeric columns and forward fill
        for categorical columns. Logs the number of missing values handled.
        
        Args:
            data (pd.DataFrame): Input data with potential missing values
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        # Create a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Log initial missing values
        initial_missing = cleaned_data.isnull().sum().sum()
        if initial_missing > 0:
            self.logger.info(f"Found {initial_missing} missing values to handle")
        
        # Handle numeric columns with linear interpolation
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            missing_count = cleaned_data[col].isnull().sum()
            if missing_count > 0:
                # Use linear interpolation for time series data
                cleaned_data[col] = cleaned_data[col].interpolate(
                    method='linear',
                    limit_direction='both'
                )
                self.logger.info(f"Interpolated {missing_count} missing values in column '{col}'")
        
        # Handle categorical columns with forward fill
        categorical_columns = cleaned_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            missing_count = cleaned_data[col].isnull().sum()
            if missing_count > 0:
                cleaned_data[col] = cleaned_data[col].fillna(method='ffill')
                # If still missing values at the beginning, use backward fill
                cleaned_data[col] = cleaned_data[col].fillna(method='bfill')
                self.logger.info(f"Forward/backward filled {missing_count} missing values in column '{col}'")
        
        # Final check for any remaining missing values
        final_missing = cleaned_data.isnull().sum().sum()
        if final_missing > 0:
            self.logger.warning(f"Still have {final_missing} missing values after cleaning")
        else:
            self.logger.info("All missing values successfully handled")
        
        return cleaned_data
    
    def detect_outliers(self, data: pd.DataFrame) -> List[int]:
        """
        Detect outliers using IQR and Z-score methods.
        
        Combines both IQR (Interquartile Range) and Z-score methods to identify
        outliers in numeric columns. Returns indices of rows containing outliers.
        
        Args:
            data (pd.DataFrame): Input data to analyze for outliers
            
        Returns:
            List[int]: List of row indices that contain outliers
        """
        outlier_indices = set()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Skip columns with all NaN values
            if data[col].isnull().all():
                continue
                
            # IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds (1.5 * IQR is standard)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find IQR outliers
            iqr_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
            outlier_indices.update(iqr_outliers)
            
            # Z-score method (threshold of 3 is standard)
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            z_outliers = data[z_scores > 3].index
            outlier_indices.update(z_outliers)
            
            # Log findings for this column
            iqr_count = len(iqr_outliers)
            z_count = len(z_outliers)
            if iqr_count > 0 or z_count > 0:
                self.logger.info(f"Column '{col}': {iqr_count} IQR outliers, {z_count} Z-score outliers")
        
        outlier_list = sorted(list(outlier_indices))
        self.logger.info(f"Total unique outlier rows detected: {len(outlier_list)}")
        
        return outlier_list
    
    def normalize_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize dates to consistent datetime formatting.
        
        Converts date columns to pandas datetime format and ensures
        consistent formatting across the dataset.
        
        Args:
            data (pd.DataFrame): Input data with date columns to normalize
            
        Returns:
            pd.DataFrame: Data with normalized datetime columns
        """
        normalized_data = data.copy()
        
        # Common date column names to look for
        date_columns = ['fecha', 'date', 'periodo', 'period']
        
        for col in normalized_data.columns:
            # Check if column name suggests it's a date column
            if any(date_col in col.lower() for date_col in date_columns):
                try:
                    # Convert to datetime with flexible parsing
                    normalized_data[col] = pd.to_datetime(
                        normalized_data[col],
                        infer_datetime_format=True,
                        errors='coerce'
                    )
                    
                    # Check for any failed conversions
                    failed_conversions = normalized_data[col].isnull().sum()
                    if failed_conversions > 0:
                        self.logger.warning(f"Failed to convert {failed_conversions} dates in column '{col}'")
                    else:
                        self.logger.info(f"Successfully normalized dates in column '{col}'")
                        
                except Exception as e:
                    self.logger.error(f"Error normalizing dates in column '{col}': {e}")
            
            # Also check if column contains date-like strings
            elif normalized_data[col].dtype == 'object':
                # Sample a few values to check if they look like dates
                sample_values = normalized_data[col].dropna().head(5)
                if len(sample_values) > 0:
                    # Try to parse first few values as dates
                    try:
                        pd.to_datetime(sample_values.iloc[0])
                        # If successful, convert the entire column
                        normalized_data[col] = pd.to_datetime(
                            normalized_data[col],
                            infer_datetime_format=True,
                            errors='coerce'
                        )
                        self.logger.info(f"Auto-detected and normalized dates in column '{col}'")
                    except:
                        # Not a date column, skip
                        pass
        
        return normalized_data
    
    def calculate_inflation_rates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monthly and annual inflation rates.
        
        Computes inflation rates based on IPC values, adding new columns
        for monthly and annual percentage changes.
        
        Args:
            data (pd.DataFrame): Input data with IPC values
            
        Returns:
            pd.DataFrame: Data with added inflation rate columns
        """
        rates_data = data.copy()
        
        # Look for IPC-related columns
        ipc_columns = [col for col in rates_data.columns if 'ipc' in col.lower()]
        
        for col in ipc_columns:
            if rates_data[col].dtype in ['float64', 'int64']:
                # Calculate monthly inflation rate (month-over-month percentage change)
                monthly_col = f"{col}_monthly_rate"
                rates_data[monthly_col] = rates_data[col].pct_change() * 100
                
                # Calculate annual inflation rate (year-over-year percentage change)
                annual_col = f"{col}_annual_rate"
                rates_data[annual_col] = rates_data[col].pct_change(periods=12) * 100
                
                self.logger.info(f"Calculated inflation rates for column '{col}'")
        
        # If no IPC columns found, look for any numeric column that might represent price index
        if not ipc_columns:
            numeric_columns = rates_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if any(keyword in col.lower() for keyword in ['index', 'indice', 'precio', 'price']):
                    monthly_col = f"{col}_monthly_rate"
                    annual_col = f"{col}_annual_rate"
                    
                    rates_data[monthly_col] = rates_data[col].pct_change() * 100
                    rates_data[annual_col] = rates_data[col].pct_change(periods=12) * 100
                    
                    self.logger.info(f"Calculated inflation rates for price index column '{col}'")
        
        return rates_data
    
    def generate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate descriptive statistics for the dataset.
        
        Creates comprehensive statistical summary including basic statistics,
        missing value counts, and data quality metrics.
        
        Args:
            data (pd.DataFrame): Input data to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing statistical summaries
        """
        stats = {}
        
        # Basic dataset information
        stats['dataset_info'] = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'date_range': self._get_date_range(data)
        }
        
        # Missing values analysis
        missing_values = data.isnull().sum()
        stats['missing_values'] = {
            'total_missing': missing_values.sum(),
            'missing_percentage': (missing_values.sum() / (len(data) * len(data.columns))) * 100,
            'columns_with_missing': missing_values[missing_values > 0].to_dict()
        }
        
        # Numeric columns statistics
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            stats['numeric_statistics'] = {
                'summary': numeric_data.describe().to_dict(),
                'correlation_matrix': numeric_data.corr().to_dict() if len(numeric_data.columns) > 1 else {},
                'skewness': numeric_data.skew().to_dict(),
                'kurtosis': numeric_data.kurtosis().to_dict()
            }
        
        # Categorical columns statistics
        categorical_data = data.select_dtypes(include=['object', 'category'])
        if not categorical_data.empty:
            stats['categorical_statistics'] = {}
            for col in categorical_data.columns:
                stats['categorical_statistics'][col] = {
                    'unique_values': categorical_data[col].nunique(),
                    'most_frequent': categorical_data[col].mode().iloc[0] if not categorical_data[col].mode().empty else None,
                    'value_counts': categorical_data[col].value_counts().head(10).to_dict()
                }
        
        # Data quality metrics
        stats['data_quality'] = {
            'duplicate_rows': data.duplicated().sum(),
            'duplicate_percentage': (data.duplicated().sum() / len(data)) * 100,
            'completeness_score': ((data.size - data.isnull().sum().sum()) / data.size) * 100
        }
        
        # Inflation-specific statistics (if applicable)
        inflation_columns = [col for col in data.columns if 'rate' in col.lower() or 'inflacion' in col.lower()]
        if inflation_columns:
            stats['inflation_statistics'] = {}
            for col in inflation_columns:
                if data[col].dtype in ['float64', 'int64']:
                    stats['inflation_statistics'][col] = {
                        'mean_rate': data[col].mean(),
                        'median_rate': data[col].median(),
                        'std_rate': data[col].std(),
                        'min_rate': data[col].min(),
                        'max_rate': data[col].max(),
                        'periods_above_2_percent': (data[col] > 2.0).sum(),
                        'periods_deflation': (data[col] < 0.0).sum()
                    }
        
        self.logger.info("Generated comprehensive dataset statistics")
        return stats
    
    def _get_date_range(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Helper method to extract date range from dataset.
        
        Args:
            data (pd.DataFrame): Input data to analyze
            
        Returns:
            Dict[str, str]: Dictionary with start and end dates
        """
        date_columns = data.select_dtypes(include=['datetime64']).columns
        
        if len(date_columns) > 0:
            # Use the first date column found
            date_col = date_columns[0]
            return {
                'start_date': str(data[date_col].min()),
                'end_date': str(data[date_col].max()),
                'date_column': date_col
            }
        else:
            return {
                'start_date': 'No date column found',
                'end_date': 'No date column found',
                'date_column': None
            }