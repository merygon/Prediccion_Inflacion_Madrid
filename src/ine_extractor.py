"""
INE Data Extractor Module

This module provides functionality to extract inflation data from the Spanish
National Statistics Institute (INE) API. It handles different types of IPC data
including general index, sectoral groups, and harmonized European index (IPCA).
"""

import requests
import pandas as pd
import time
import logging
from typing import Dict, Optional, List
from datetime import datetime
import yaml
import json


class INEExtractor:
    """
    Extractor class for downloading inflation data from INE API.
    
    Handles HTTP connections with retry logic and exponential backoff
    for robust data extraction from different IPC data sources.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize INE Extractor with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.session = self._create_session()
        
        # Extract configuration values
        self.base_url = self.config['data']['ine_base_url']
        self.urls = self.config['data']['urls']
        self.max_attempts = self.config['data']['retry']['max_attempts']
        self.backoff_factor = self.config['data']['retry']['backoff_factor']
        self.timeout = self.config['data']['retry']['timeout']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing configuration file: {e}")
            raise
            
    def _create_session(self) -> requests.Session:
        """
        Create HTTP session with appropriate headers for INE API.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Spanish-Inflation-Predictor/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        return session
        
    def _make_request_with_retry(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        """
        Make HTTP request with exponential backoff retry logic.
        
        Args:
            url: Target URL for the request
            params: Optional query parameters
            
        Returns:
            HTTP response object
            
        Raises:
            requests.RequestException: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                self.logger.info(f"Attempting request to {url} (attempt {attempt + 1}/{self.max_attempts})")
                
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                self.logger.info(f"Request successful on attempt {attempt + 1}")
                return response
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                self.logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_attempts - 1:
                    wait_time = self.backoff_factor ** attempt
                    self.logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    
        # If we get here, all attempts failed
        self.logger.error(f"All {self.max_attempts} attempts failed for URL: {url}")
        raise last_exception
        
    def _parse_ine_response(self, response: requests.Response) -> pd.DataFrame:
        """
        Parse INE API response and convert to pandas DataFrame with adaptive strategies.
        
        Args:
            response: HTTP response from INE API
            
        Returns:
            Parsed data as pandas DataFrame
        """
        try:
            # INE API returns JSON data
            data = response.json()
            self.logger.debug(f"API response type: {type(data)}, length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")

            # Strategy 1: Handle INE's typical array-of-objects format
            df = self._extract_from_ine_array_format(data)
            if df is not None:
                self.logger.info(f"Successfully parsed using INE array format strategy: {len(df)} records")
                return df

            # Strategy 2: Handle direct list response format
            df = self._extract_from_direct_list(data)
            if df is not None:
                self.logger.info(f"Successfully parsed using direct list strategy: {len(df)} records")
                return df

            # Strategy 3: Search for data in common INE keys
            df = self._extract_from_common_keys(data)
            if df is not None:
                self.logger.info(f"Successfully parsed using common keys strategy: {len(df)} records")
                return df

            # Strategy 4: Scan all properties for list-type data
            df = self._extract_from_any_list_property(data)
            if df is not None:
                self.logger.info(f"Successfully parsed using property scan strategy: {len(df)} records")
                return df

            # Strategy 5: Try json_normalize for complex nested structures
            df = self._extract_using_json_normalize(data)
            if df is not None:
                self.logger.info(f"Successfully parsed using json_normalize strategy: {len(df)} records")
                return df

            # If all strategies fail, save debug info and raise error
            self._save_debug_response(data, "parsing_failed")
            raise ValueError("Could not extract data using any parsing strategy")

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Error parsing INE response: {e}")
            self.logger.debug(f"Response content (first 1000 chars): {response.text[:1000]}...")
            raise

    def _extract_from_ine_array_format(self, data) -> Optional[pd.DataFrame]:
        """Extract data from INE's typical array-of-objects format where each object has a 'Data' property."""
        try:
            if isinstance(data, list) and len(data) > 0:
                # Look for objects with 'Data' property containing time series
                for item in data:
                    if isinstance(item, dict) and 'Data' in item:
                        data_array = item['Data']
                        if isinstance(data_array, list) and len(data_array) > 0:
                            df = pd.DataFrame(data_array)
                            # Convert timestamp to readable date
                            if 'Fecha' in df.columns:
                                df['Fecha'] = pd.to_datetime(df['Fecha'], unit='ms')
                            self.logger.debug(f"Extracted {len(df)} records from INE array format")
                            return df
        except Exception as e:
            self.logger.debug(f"INE array format extraction failed: {e}")
        return None

    def _extract_from_direct_list(self, data) -> Optional[pd.DataFrame]:
        """Extract data from direct list format."""
        try:
            if isinstance(data, list) and len(data) > 0:
                # Check if it's a simple list of records
                first_item = data[0]
                if isinstance(first_item, dict) and not any(key in first_item for key in ['Data', 'Datos']):
                    df = pd.DataFrame(data)
                    self.logger.debug(f"Extracted {len(df)} records from direct list format")
                    return df
        except Exception as e:
            self.logger.debug(f"Direct list extraction failed: {e}")
        return None

    def _extract_from_common_keys(self, data) -> Optional[pd.DataFrame]:
        """Extract data from common INE data keys."""
        try:
            if isinstance(data, dict):
                # Common keys that may contain the records
                possible_keys = ['Data', 'data', 'Datos', 'datos', 'Valores', 'valores', 'Series', 'series']
                for key in possible_keys:
                    if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                        df = pd.DataFrame(data[key])
                        self.logger.debug(f"Extracted {len(df)} records using key '{key}'")
                        return df
        except Exception as e:
            self.logger.debug(f"Common keys extraction failed: {e}")
        return None

    def _extract_from_any_list_property(self, data) -> Optional[pd.DataFrame]:
        """Scan all object properties for list-type data."""
        try:
            if isinstance(data, dict):
                best_candidate = None
                best_score = 0
                
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        # Score based on list length and content structure
                        score = len(value)
                        if isinstance(value[0], dict):
                            score += 10  # Prefer lists of objects
                        
                        if score > best_score:
                            best_candidate = value
                            best_score = score
                
                if best_candidate:
                    df = pd.DataFrame(best_candidate)
                    self.logger.debug(f"Extracted {len(df)} records from best candidate property")
                    return df
        except Exception as e:
            self.logger.debug(f"Property scan extraction failed: {e}")
        return None

    def _extract_using_json_normalize(self, data) -> Optional[pd.DataFrame]:
        """Use pandas json_normalize for complex nested structures."""
        try:
            if isinstance(data, (dict, list)):
                df = pd.json_normalize(data)
                if len(df) > 0:
                    self.logger.debug(f"Extracted {len(df)} records using json_normalize")
                    return df
        except Exception as e:
            self.logger.debug(f"json_normalize extraction failed: {e}")
        return None

    def _save_debug_response(self, data, reason: str) -> None:
        """Save raw API response for debugging purposes."""
        try:
            logs_dir = self.config.get('paths', {}).get('logs', 'logs/')
            import os
            os.makedirs(logs_dir, exist_ok=True)
            
            timestamp = int(time.time())
            raw_path = os.path.join(logs_dir, f"raw_ine_response_{reason}_{timestamp}.json")
            
            with open(raw_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.error(f"Raw API response saved to {raw_path} (reason: {reason})")
            
            # Also log structure summary
            if isinstance(data, list):
                self.logger.debug(f"Response is list with {len(data)} items")
                if len(data) > 0:
                    self.logger.debug(f"First item type: {type(data[0])}, keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}")
            elif isinstance(data, dict):
                self.logger.debug(f"Response is dict with keys: {list(data.keys())}")
                
        except Exception as e:
            self.logger.debug(f"Failed to save debug response: {e}")
            
    def get_ipc_general_connection(self) -> bool:
        """
        Test connection to IPC General data endpoint.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            url = self.urls['ipc_general']
            response = self._make_request_with_retry(url)
            self.logger.info("IPC General connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"IPC General connection test failed: {e}")
            return False
            
    def get_ipc_groups_connection(self) -> bool:
        """
        Test connection to IPC Groups data endpoint.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            url = self.urls['ipc_groups']
            response = self._make_request_with_retry(url)
            self.logger.info("IPC Groups connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"IPC Groups connection test failed: {e}")
            return False
            
    def get_ipca_connection(self) -> bool:
        """
        Test connection to IPCA (Harmonized European Index) data endpoint.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            url = self.urls['ipca']
            response = self._make_request_with_retry(url)
            self.logger.info("IPCA connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"IPCA connection test failed: {e}")
            return False
            
    def test_all_connections(self) -> Dict[str, bool]:
        """
        Test connections to all INE data endpoints.
        
        Returns:
            Dictionary with connection test results for each endpoint
        """
        results = {
            'ipc_general': self.get_ipc_general_connection(),
            'ipc_groups': self.get_ipc_groups_connection(),
            'ipca': self.get_ipca_connection()
        }
        
        successful_connections = sum(results.values())
        total_connections = len(results)
        
        self.logger.info(f"Connection tests completed: {successful_connections}/{total_connections} successful")
        
        return results
        
    def download_ipc_general(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download IPC General (main inflation index) data from INE.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with IPC General time series data
            
        Raises:
            requests.RequestException: If download fails after all retries
            ValueError: If date format is invalid
        """
        self.logger.info(f"Downloading IPC General data from {start_date} to {end_date}")
        
        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            raise ValueError("Dates must be in YYYY-MM-DD format")
            
        # Prepare request parameters
        url = self.urls['ipc_general']
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'format': 'json'
        }
        
        try:
            # Make request with retry logic
            response = self._make_request_with_retry(url, params)
            
            # Parse response to DataFrame
            df = self._parse_ine_response(response)
            
            # Standardize column names for IPC General
            df = self._standardize_ipc_general_columns(df)
            
            self.logger.info(f"Successfully downloaded {len(df)} IPC General records")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to download IPC General data: {e}")
            raise
            
    def download_ipc_groups(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download IPC Groups (sectoral inflation data) from INE.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with IPC Groups time series data by sector
            
        Raises:
            requests.RequestException: If download fails after all retries
            ValueError: If date format is invalid
        """
        self.logger.info(f"Downloading IPC Groups data from {start_date} to {end_date}")
        
        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            raise ValueError("Dates must be in YYYY-MM-DD format")
            
        # Prepare request parameters
        url = self.urls['ipc_groups']
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'format': 'json'
        }
        
        try:
            # Make request with retry logic
            response = self._make_request_with_retry(url, params)
            
            # Parse response to DataFrame
            df = self._parse_ine_response(response)
            
            # Standardize column names for IPC Groups
            df = self._standardize_ipc_groups_columns(df)
            
            self.logger.info(f"Successfully downloaded {len(df)} IPC Groups records")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to download IPC Groups data: {e}")
            raise
            
    def download_ipca(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download IPCA (Harmonized European Index) data from INE.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with IPCA time series data
            
        Raises:
            requests.RequestException: If download fails after all retries
            ValueError: If date format is invalid
        """
        self.logger.info(f"Downloading IPCA data from {start_date} to {end_date}")
        
        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            raise ValueError("Dates must be in YYYY-MM-DD format")
            
        # Prepare request parameters
        url = self.urls['ipca']
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'format': 'json'
        }
        
        try:
            # Make request with retry logic
            response = self._make_request_with_retry(url, params)
            
            # Parse response to DataFrame
            df = self._parse_ine_response(response)
            
            # Standardize column names for IPCA
            df = self._standardize_ipca_columns(df)
            
            self.logger.info(f"Successfully downloaded {len(df)} IPCA records")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to download IPCA data: {e}")
            raise
            
    def _standardize_ipc_general_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for IPC General data with flexible mapping strategies.
        
        Args:
            df: Raw DataFrame from INE API
            
        Returns:
            DataFrame with standardized column names
        """
        self.logger.debug(f"Standardizing IPC General columns. Original columns: {list(df.columns)}")
        
        df_standardized = df.copy()
        
        # Strategy 1: Exact column name mapping
        column_mapping = {
            'Fecha': 'fecha',
            'Date': 'fecha', 
            'fecha': 'fecha',
            'Valor': 'ipc_general',
            'Value': 'ipc_general',
            'valor': 'ipc_general',
            'IPC': 'ipc_general',
            'Indice': 'ipc_general',
            'Index': 'ipc_general'
        }
        
        mapped_columns = {}
        for old_name, new_name in column_mapping.items():
            if old_name in df_standardized.columns:
                mapped_columns[old_name] = new_name
        
        if mapped_columns:
            df_standardized = df_standardized.rename(columns=mapped_columns)
            self.logger.debug(f"Applied exact mappings: {mapped_columns}")
        
        # Strategy 2: Pattern-based mapping for date columns
        if 'fecha' not in df_standardized.columns:
            date_column = self._find_date_column(df_standardized)
            if date_column:
                df_standardized = df_standardized.rename(columns={date_column: 'fecha'})
                self.logger.debug(f"Mapped date column '{date_column}' to 'fecha'")
        
        # Strategy 3: Content-based mapping for numeric value columns
        if 'ipc_general' not in df_standardized.columns:
            value_column = self._find_numeric_value_column(df_standardized)
            if value_column:
                df_standardized = df_standardized.rename(columns={value_column: 'ipc_general'})
                self.logger.debug(f"Mapped value column '{value_column}' to 'ipc_general'")
        
        # Strategy 4: Position-based inference as last resort
        required_columns = ['fecha', 'ipc_general']
        missing_columns = [col for col in required_columns if col not in df_standardized.columns]
        
        if missing_columns and len(df_standardized.columns) >= 2:
            self.logger.warning(f"Using position-based column inference for missing columns: {missing_columns}")
            # Assume first column is date, second is value
            new_column_names = ['fecha', 'ipc_general'] + [f'col_{i}' for i in range(2, len(df_standardized.columns))]
            df_standardized.columns = new_column_names[:len(df_standardized.columns)]
        
        self.logger.info(f"Final standardized columns: {list(df_standardized.columns)}")
        return df_standardized

    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find a column that likely contains date information."""
        for col in df.columns:
            col_lower = str(col).lower()
            # Check column name patterns
            if any(pattern in col_lower for pattern in ['fecha', 'date', 'time', 'periodo']):
                return col
            
            # Check column content (sample first few non-null values)
            try:
                sample_values = df[col].dropna().head(3)
                if len(sample_values) > 0:
                    # Try to detect timestamp (large numbers) or date strings
                    first_val = sample_values.iloc[0]
                    if isinstance(first_val, (int, float)) and first_val > 1000000000:  # Likely timestamp
                        return col
                    elif isinstance(first_val, str) and any(char.isdigit() for char in first_val):
                        # Try parsing as date
                        try:
                            pd.to_datetime(first_val)
                            return col
                        except:
                            pass
            except:
                pass
        return None

    def _find_numeric_value_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find a column that likely contains the main numeric values."""
        numeric_columns = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
        
        for col in numeric_columns:
            col_lower = str(col).lower()
            # Check for value-related patterns
            if any(pattern in col_lower for pattern in ['valor', 'value', 'indice', 'index', 'ipc']):
                return col
        
        # If no pattern match, return the first numeric column that's not likely a date
        for col in numeric_columns:
            # Skip columns that might be dates (large numbers suggesting timestamps)
            try:
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    mean_val = sample_values.mean()
                    # Skip if values look like timestamps
                    if mean_val > 1000000000:  # Likely timestamp
                        continue
                    return col
            except:
                pass
        
        return numeric_columns[0] if len(numeric_columns) > 0 else None
        
    def _standardize_ipc_groups_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for IPC Groups data.
        
        Args:
            df: Raw DataFrame from INE API
            
        Returns:
            DataFrame with standardized column names
        """
        # Common INE column mappings for IPC Groups
        column_mapping = {
            'Fecha': 'fecha',
            'Date': 'fecha',
            'fecha': 'fecha',
            'Alimentos y bebidas no alcohólicas': 'ipc_alimentacion',
            'Bebidas alcohólicas y tabaco': 'ipc_bebidas_tabaco',
            'Vestido y calzado': 'ipc_vestido',
            'Vivienda': 'ipc_vivienda',
            'Menaje': 'ipc_menaje',
            'Medicina': 'ipc_medicina',
            'Transporte': 'ipc_transporte',
            'Comunicaciones': 'ipc_comunicaciones',
            'Ocio y cultura': 'ipc_ocio',
            'Enseñanza': 'ipc_ensenanza',
            'Hoteles, cafés y restaurantes': 'ipc_hoteles',
            'Otros': 'ipc_otros'
        }
        
        # Apply column mapping
        df_standardized = df.copy()
        for old_name, new_name in column_mapping.items():
            if old_name in df_standardized.columns:
                df_standardized = df_standardized.rename(columns={old_name: new_name})
                
        return df_standardized
        
    def _standardize_ipca_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for IPCA data.
        
        Args:
            df: Raw DataFrame from INE API
            
        Returns:
            DataFrame with standardized column names
        """
        # Common INE column mappings for IPCA
        column_mapping = {
            'Fecha': 'fecha',
            'Date': 'fecha',
            'fecha': 'fecha',
            'Valor': 'ipca',
            'Value': 'ipca',
            'valor': 'ipca',
            'IPCA': 'ipca',
            'Indice': 'ipca',
            'Index': 'ipca'
        }
        
        # Apply column mapping
        df_standardized = df.copy()
        for old_name, new_name in column_mapping.items():
            if old_name in df_standardized.columns:
                df_standardized = df_standardized.rename(columns={old_name: new_name})
                
        # Ensure we have the required columns
        required_columns = ['fecha', 'ipca']
        missing_columns = [col for col in required_columns if col not in df_standardized.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing required columns in IPCA data: {missing_columns}")
            # Try to infer from available columns
            if len(df_standardized.columns) >= 2:
                df_standardized.columns = ['fecha', 'ipca'] + list(df_standardized.columns[2:])
                
        return df_standardized
        
    def save_to_csv(self, data: pd.DataFrame, filename: str, data_type: str = "general") -> None:
        """
        Save DataFrame to CSV with proper formatting and validation.
        
        Args:
            data: DataFrame to save
            filename: Output filename (without extension)
            data_type: Type of data being saved ("general", "groups", "ipca")
            
        Raises:
            ValueError: If data validation fails
            IOError: If file cannot be written
        """
        self.logger.info(f"Saving {data_type} data to CSV: {filename}")
        
        # Validate data before saving
        if not self._validate_data_for_export(data, data_type):
            raise ValueError(f"Data validation failed for {data_type} export")
            
        # Prepare data for export
        df_export = self._prepare_data_for_export(data, data_type)
        
        # Construct full file path
        raw_data_path = self.config['paths']['data']['raw']
        full_path = f"{raw_data_path}{filename}.csv"
        
        try:
            # Ensure directory exists
            import os
            os.makedirs(raw_data_path, exist_ok=True)
            
            # Save to CSV with configuration settings
            encoding = self.config['output']['csv_encoding']
            decimal_places = self.config['output']['decimal_places']
            
            # Round numeric columns to specified decimal places
            numeric_columns = df_export.select_dtypes(include=['float64', 'int64']).columns
            df_export[numeric_columns] = df_export[numeric_columns].round(decimal_places)
            
            # Save to CSV
            df_export.to_csv(
                full_path,
                index=False,
                encoding=encoding,
                date_format=self.config['output']['date_format']
            )
            
            self.logger.info(f"Successfully saved {len(df_export)} records to {full_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data to CSV: {e}")
            raise IOError(f"Could not write file {full_path}: {e}")
            
    def _validate_data_for_export(self, data: pd.DataFrame, data_type: str) -> bool:
        """
        Enhanced data validation with detailed error reporting and recovery options.
        
        Args:
            data: DataFrame to validate
            data_type: Type of data being validated
            
        Returns:
            True if data is valid for export, False otherwise
        """
        validation_errors = []
        validation_warnings = []
        
        # Basic data checks
        if data.empty:
            validation_errors.append("DataFrame is empty")
            self.logger.error("Cannot export empty DataFrame")
            return False
        
        self.logger.info(f"Validating {data_type} data: {len(data)} records, columns: {list(data.columns)}")
        
        # Check for required columns based on data type
        required_columns = {
            'general': ['fecha', 'ipc_general'],
            'groups': ['fecha'],  # Groups can have variable columns
            'ipca': ['fecha', 'ipca']
        }
        
        if data_type in required_columns:
            missing_columns = [col for col in required_columns[data_type] 
                             if col not in data.columns]
            if missing_columns:
                validation_errors.append(f"Missing required columns: {missing_columns}")
                self.logger.error(f"Missing required columns for {data_type}: {missing_columns}")
                self.logger.error(f"Available columns: {list(data.columns)}")
                
                # Try to suggest column mappings
                self._suggest_column_mappings(data.columns, required_columns[data_type])
                return False
        
        # Enhanced date column validation
        if 'fecha' in data.columns:
            date_validation_result = self._validate_date_column(data['fecha'])
            if not date_validation_result['is_valid']:
                validation_errors.extend(date_validation_result['errors'])
                validation_warnings.extend(date_validation_result['warnings'])
                
                # If date validation fails completely, it's a critical error
                if date_validation_result['errors']:
                    self.logger.error(f"Date column validation failed: {date_validation_result['errors']}")
                    return False
        
        # Enhanced numeric columns validation
        numeric_columns = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
        for col in numeric_columns:
            if data[col].isna().all():
                validation_warnings.append(f"Column {col} contains only NaN values")
                self.logger.warning(f"Column {col} contains only NaN values")
            elif data[col].isna().sum() > len(data) * 0.8:
                validation_warnings.append(f"Column {col} has more than 80% missing values")
                self.logger.warning(f"Column {col} has more than 80% missing values")
            elif data[col].isna().sum() > len(data) * 0.5:
                validation_warnings.append(f"Column {col} has more than 50% missing values")
                self.logger.warning(f"Column {col} has more than 50% missing values")
        
        # Data quality checks
        data_quality_score = self._calculate_data_quality_score(data)
        self.logger.info(f"Data quality score: {data_quality_score:.2f}/100")
        
        if data_quality_score < 50:
            validation_warnings.append(f"Low data quality score: {data_quality_score:.1f}/100")
        
        # Log validation summary
        if validation_warnings:
            self.logger.warning(f"Validation warnings for {data_type}: {validation_warnings}")
        
        if validation_errors:
            self.logger.error(f"Validation errors for {data_type}: {validation_errors}")
            return False
        
        self.logger.info(f"Data validation passed for {data_type} export")
        return True

    def _validate_date_column(self, date_series: pd.Series) -> Dict:
        """Validate date column with multiple format support."""
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'format_detected': None
        }
        
        try:
            # Sample some values for testing
            sample_values = date_series.dropna().head(10)
            if len(sample_values) == 0:
                result['errors'].append("Date column contains no valid values")
                result['is_valid'] = False
                return result
            
            first_value = sample_values.iloc[0]
            self.logger.debug(f"Date column sample value: {first_value} (type: {type(first_value)})")
            
            # Try different parsing strategies
            if isinstance(first_value, (int, float)):
                # Likely timestamp - try milliseconds first, then seconds
                try:
                    if first_value > 1e12:  # Milliseconds timestamp
                        pd.to_datetime(date_series, unit='ms')
                        result['format_detected'] = 'timestamp_ms'
                    else:  # Seconds timestamp
                        pd.to_datetime(date_series, unit='s')
                        result['format_detected'] = 'timestamp_s'
                except Exception as e:
                    result['errors'].append(f"Cannot parse numeric dates as timestamps: {e}")
                    result['is_valid'] = False
            else:
                # Try standard datetime parsing
                try:
                    pd.to_datetime(date_series)
                    result['format_detected'] = 'datetime_string'
                except Exception as e:
                    result['errors'].append(f"Cannot parse date strings: {e}")
                    result['is_valid'] = False
            
        except Exception as e:
            result['errors'].append(f"Date validation error: {e}")
            result['is_valid'] = False
        
        return result

    def _suggest_column_mappings(self, available_columns: List[str], required_columns: List[str]) -> None:
        """Suggest possible column mappings when required columns are missing."""
        self.logger.info("Suggesting possible column mappings:")
        
        for required_col in required_columns:
            suggestions = []
            for available_col in available_columns:
                available_lower = str(available_col).lower()
                required_lower = required_col.lower()
                
                # Check for partial matches
                if required_lower in available_lower or available_lower in required_lower:
                    suggestions.append(available_col)
                
                # Check for common patterns
                if required_col == 'fecha' and any(pattern in available_lower for pattern in ['date', 'time', 'periodo']):
                    suggestions.append(available_col)
                elif 'ipc' in required_lower and any(pattern in available_lower for pattern in ['valor', 'value', 'indice']):
                    suggestions.append(available_col)
            
            if suggestions:
                self.logger.info(f"  {required_col} -> possible matches: {suggestions}")
            else:
                self.logger.info(f"  {required_col} -> no obvious matches found")

    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate a data quality score (0-100) based on completeness and consistency."""
        score = 100.0
        
        # Penalize for missing data
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isna().sum().sum()
        if total_cells > 0:
            missing_percentage = (missing_cells / total_cells) * 100
            score -= missing_percentage
        
        # Penalize for completely empty columns
        empty_columns = data.columns[data.isna().all()].tolist()
        score -= len(empty_columns) * 10
        
        # Bonus for having expected data types
        numeric_columns = data.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            score += 5
        
        return max(0.0, min(100.0, score))
        
    def _prepare_data_for_export(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Prepare data for CSV export with standardized formatting and enhanced date handling.
        
        Args:
            data: DataFrame to prepare
            data_type: Type of data being prepared
            
        Returns:
            Prepared DataFrame ready for export
        """
        df_prepared = data.copy()
        
        # Enhanced fecha column conversion
        if 'fecha' in df_prepared.columns:
            df_prepared = self._convert_fecha_column(df_prepared)
            
        # Sort by date if possible
        if 'fecha' in df_prepared.columns:
            try:
                df_prepared = df_prepared.sort_values('fecha')
            except Exception as e:
                self.logger.warning(f"Could not sort by fecha column: {e}")
            
        # Reset index
        df_prepared = df_prepared.reset_index(drop=True)
        
        # Add metadata columns
        df_prepared['data_type'] = data_type
        df_prepared['extraction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.logger.info(f"Data prepared for {data_type} export: {len(df_prepared)} records")
        return df_prepared

    def _convert_fecha_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert fecha column to standardized date format with multiple parsing strategies."""
        try:
            fecha_series = df['fecha']
            sample_value = fecha_series.dropna().iloc[0] if len(fecha_series.dropna()) > 0 else None
            
            if sample_value is None:
                self.logger.warning("Fecha column contains no valid values")
                return df
            
            self.logger.debug(f"Converting fecha column, sample value: {sample_value} (type: {type(sample_value)})")
            
            # Strategy 1: Handle timestamp values (milliseconds)
            if isinstance(sample_value, (int, float)) and sample_value > 1e12:
                df['fecha'] = pd.to_datetime(df['fecha'], unit='ms')
                self.logger.debug("Converted fecha from millisecond timestamps")
            
            # Strategy 2: Handle timestamp values (seconds)
            elif isinstance(sample_value, (int, float)) and sample_value > 1e9:
                df['fecha'] = pd.to_datetime(df['fecha'], unit='s')
                self.logger.debug("Converted fecha from second timestamps")
            
            # Strategy 3: Handle string dates
            else:
                df['fecha'] = pd.to_datetime(df['fecha'])
                self.logger.debug("Converted fecha from datetime strings")
            
            # Format to standard string format
            date_format = self.config['output']['date_format']
            df['fecha'] = df['fecha'].dt.strftime(date_format)
            self.logger.debug(f"Formatted fecha to {date_format}")
            
        except Exception as e:
            self.logger.error(f"Error converting fecha column: {e}")
            # Keep original values if conversion fails
            
        return df
        
    def export_all_data(self, start_date: str, end_date: str) -> Dict[str, str]:
        """
        Download and export all types of IPC data to CSV files.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with file paths for each exported dataset
            
        Raises:
            Exception: If any download or export operation fails
        """
        self.logger.info(f"Starting complete data export from {start_date} to {end_date}")
        
        exported_files = {}
        
        try:
            # Download and export IPC General
            self.logger.info("Downloading IPC General data...")
            ipc_general = self.download_ipc_general(start_date, end_date)
            filename_general = f"ipc_general_{start_date}_{end_date}"
            self.save_to_csv(ipc_general, filename_general, "general")
            exported_files['ipc_general'] = f"{self.config['paths']['data']['raw']}{filename_general}.csv"
            
            # Download and export IPC Groups
            self.logger.info("Downloading IPC Groups data...")
            ipc_groups = self.download_ipc_groups(start_date, end_date)
            filename_groups = f"ipc_groups_{start_date}_{end_date}"
            self.save_to_csv(ipc_groups, filename_groups, "groups")
            exported_files['ipc_groups'] = f"{self.config['paths']['data']['raw']}{filename_groups}.csv"
            
            # Download and export IPCA
            self.logger.info("Downloading IPCA data...")
            ipca = self.download_ipca(start_date, end_date)
            filename_ipca = f"ipca_{start_date}_{end_date}"
            self.save_to_csv(ipca, filename_ipca, "ipca")
            exported_files['ipca'] = f"{self.config['paths']['data']['raw']}{filename_ipca}.csv"
            
            self.logger.info("Complete data export finished successfully")
            self.logger.info(f"Exported files: {list(exported_files.values())}")
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Data export failed: {e}")
            raise
            
    def get_data_summary(self, data: pd.DataFrame, data_type: str) -> Dict:
        """
        Generate summary statistics for downloaded data.
        
        Args:
            data: DataFrame to summarize
            data_type: Type of data being summarized
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'data_type': data_type,
            'total_records': len(data),
            'date_range': {},
            'columns': list(data.columns),
            'missing_values': {},
            'numeric_summary': {}
        }
        
        # Date range information
        if 'fecha' in data.columns:
            dates = pd.to_datetime(data['fecha'])
            summary['date_range'] = {
                'start_date': dates.min().strftime('%Y-%m-%d'),
                'end_date': dates.max().strftime('%Y-%m-%d'),
                'total_months': len(dates.unique())
            }
            
        # Missing values analysis
        for col in data.columns:
            missing_count = data[col].isna().sum()
            if missing_count > 0:
                summary['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_count / len(data) * 100, 2)
                }
                
        # Numeric columns summary
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            if not data[col].isna().all():
                summary['numeric_summary'][col] = {
                    'mean': round(data[col].mean(), 4),
                    'std': round(data[col].std(), 4),
                    'min': round(data[col].min(), 4),
                    'max': round(data[col].max(), 4)
                }
                
        return summary